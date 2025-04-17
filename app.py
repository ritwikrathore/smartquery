import os
import json
import base64
from datetime import datetime
from pathlib import Path
import io
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
from typing import Dict, List, Union, Optional, Literal, Any, AsyncGenerator, Tuple
import asyncio
import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.models.gemini import GeminiModel
# from pydantic_ai.format_as_xml import format_as_xml # Not used directly it seems
import plotly.express as px
import nest_asyncio
import streamlit.components.v1 as components # Import components for JS scroll
import re

# LangChain conversational memory
from langchain.memory import ConversationBufferMemory

# --- Apply nest_asyncio EARLY and GLOBALLY ---
# This patches asyncio to allow nested loops, often needed with Streamlit
nest_asyncio.apply()

# --- Set up logging ---
# Increased log level for less noise in production, but keep INFO for key events
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("nest_asyncio applied globally at startup.")
# logger.setLevel(logging.DEBUG) # Uncomment for very detailed debugging


# --- Google Generative AI Import and Configuration ---
try:
    import google.generativeai as genai
    logger.info("Successfully imported google.generativeai.")
except ImportError:
    logger.error("Google Generative AI SDK not installed.", exc_info=True)
    st.error("Google Generative AI SDK not installed. Please run `pip install google-generativeai`.")
    st.stop()

# Load API Key (Essential)
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not google_api_key:
    logger.error("🔴 GOOGLE_API_KEY not found in secrets.toml or environment variables!")
    st.error("🔴 GOOGLE_API_KEY not found. Please configure it in Streamlit secrets or environment variables.")
    st.stop()
else:
    logger.info("Google API Key loaded successfully.")
    # Configure the SDK globally once here is generally okay, but we'll re-confirm in async functions
    try:
        genai.configure(api_key=google_api_key)
        logger.info("Globally configured Google Generative AI SDK.")
    except Exception as config_err:
        logger.error(f"🔴 Failed initial global GenAI SDK configuration: {config_err}", exc_info=True)
        st.error(f"Failed initial AI configuration: {config_err}")
        # Don't stop here, maybe it works when configured locally later


# --- Configuration and Dependencies ---

# Default SQLite database path (ensure this file exists or is created)
DEFAULT_DB_PATH = st.secrets.get("DATABASE_PATH", "assets/data1.sqlite") # Example path

class AgentDependencies:
    """Manages dependencies like database connections."""
    def __init__(self):
        self.db_connection: Optional[sqlite3.Connection] = None

    @classmethod
    def create(cls) -> 'AgentDependencies':
        return cls()

    def with_db(self, db_path: str) -> 'AgentDependencies':
        """Establishes SQLite connection."""
        try:
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Connecting to database: {db_path}")
            # check_same_thread=False is crucial for multithreaded frameworks like Streamlit
            self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
            logger.info(f"Successfully connected to database: {db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {db_path}: {e}", exc_info=True)
            st.error(f"Failed to connect to the database: {e}")
            self.db_connection = None
        except Exception as e:
            logger.error(f"Unexpected error connecting to database {db_path}: {e}", exc_info=True)
            st.error(f"An unexpected error occurred while connecting to the database: {e}")
            self.db_connection = None
        return self

    async def cleanup(self):
        """Closes database connection asynchronously."""
        if self.db_connection:
            try:
                # Closing might not be strictly async, but keep await pattern if needed elsewhere
                self.db_connection.close()
                logger.info("Database connection closed.")
                self.db_connection = None
            except Exception as e:
                 logger.error(f"Error closing database connection: {e}", exc_info=True)


# Define Token Usage Limits for Gemini
DEFAULT_USAGE_LIMITS = UsageLimits()


# --- Pydantic Models for API Structure --- (Keep as is)

class SQLQueryResult(BaseModel):
    """Response when SQL could be successfully generated"""
    sql_query: str = Field(..., description="The SQL query to execute")
    explanation: str = Field("", description="Explanation of what the SQL query does")

class PythonCodeResult(BaseModel):
    """Response when Python code needs to be executed for analysis or visualization"""
    python_code: str = Field(..., description="The Python code to execute using pandas (df) and potentially matplotlib/plotly (plt/px) if absolutely needed, though prefer chart suggestions.")
    explanation: str = Field("", description="Explanation of what the Python code does")

class InvalidRequest(BaseModel):
    """Response when the request cannot be processed"""
    error_message: str = Field(..., description="Error message explaining why the request is invalid")

class SuggestedTables(BaseModel):
    """Lists tables suggested by the TableAgent."""
    table_names: List[str] = Field(..., description="List of table names deemed relevant to the user query.")
    reasoning: str = Field(..., description="Brief explanation for selecting these tables.")

class PrunedSchemaResult(BaseModel):
    """Contains the pruned schema string generated by the ColumnPruneAgent."""
    pruned_schema_string: str = Field(..., description="The database schema string containing only essential columns for the query.")
    explanation: str = Field(..., description="Explanation of which columns were kept and why.")

class QueryResponse(BaseModel):
    """Complete response from the agent, potentially including text, SQL, and Python code"""
    text_message: str = Field(..., description="Human-readable response explaining the action or findings")
    sql_result: SQLQueryResult = Field(..., description="SQL query generated by the SQL agent")
    python_result: Optional[PythonCodeResult] = Field(None, description="Python code details if Python was generated for visualization/analysis")

class DatabaseClassification(BaseModel):
    """Identifies the target database for a user query."""
    database_key: Literal["IFC", "MIGA", "IBRD", "UNKNOWN"] = Field(..., description="The database key ('IFC', 'MIGA', 'IBRD') the user query most likely refers to, based on keywords and the database descriptions provided. Use 'UNKNOWN' if the query is ambiguous.")
    reasoning: str = Field(..., description="Brief explanation for the classification (e.g., 'Query mentions IFC investments', 'Query mentions MIGA guarantees', 'Query mentions IBRD and IDA lending', 'Query is ambiguous/general').")

# --- Orchestrator Model and Blueprint ---
class OrchestratorResult(BaseModel):
    action: Literal["assistant", "database_query", "visualization"] = Field(
        ..., description="Type of action: 'assistant' for greetings, help, and general conversation, 'database_query' for DB queries, 'visualization' for chart requests."
    )
    response: str = Field(..., description="Assistant's response to the user.")
    chart_type: Optional[str] = Field(None, description="Type of chart requested (for visualization actions only).")

def create_orchestrator_agent_blueprint():
    return {
        "result_type": OrchestratorResult,
        "name": "Orchestrator Agent",
        "retries": 2,
        "system_prompt": '''You are an orchestrator agent for a database query system. Your job is to:

1. ASSISTANT MODE:
   - If the user message is a greeting, general question, or anything NOT related to database queries, respond with action='assistant'.
   - You are a helpful assistant that can greet users, answer general questions, and help users query the World Bank database systems, including IFC investment data and MIGA guarantee information about investments, projects, countries, sectors, and more.
   - If the user asks for clarification about the last response, help them as long as it is related to the World Bank database systems or the outputs of the previous database query.

2. DATABASE QUERY MODE:
   - ONLY set action='database_query' if the user wants to query the database, where SQL is the appropriate response.
   - DO NOT set action='database_query' if the user is asking for clarification about the last response or the previous database query or for a follow-up chart or visualization.
   - Provide a brief helpful response acknowledging the query, such as "I'll search the database for that information" or "Let me find that data for you."

3. VISUALIZATION MODE:
   - Set action='visualization' when the user is requesting to visualize or chart previously retrieved data.
   - When setting action='visualization', include the chart_type (e.g., "bar", "line", "pie", "scatter") in your response.
   - Provide a brief helpful response acknowledging the visualization request, such as "I'll create a bar chart with the data."

4. PYTHON AGENT USAGE:
   - You have access to a Python agent tool for data manipulation and visualization (e.g., using pandas, numpy, or plotting the last DataFrame as a chart).
   - Use this tool when the user makes a follow-up request to visualize, plot, or chart the previous results, or requests data manipulation in Python.
   - Call the Python agent with the requested operation and any relevant user context.
   - If the tool returns a successful result, respond to the user with the message and chart type or code from the tool's output. If there is an error, inform the user accordingly.
   - Only use this tool if there is previous data available to manipulate or visualize.

5. CONVERSATION MANAGEMENT:
   - Use the conversation history to maintain context.
   - If a follow-up question refers to previous results, treat it as a database query, Python data manipulation, or visualization request or clarification about the last response.
   - IMPORTANT: Short queries like "show me a bar chart" or "can you visualize this?" after a database query are almost always visualization requests, not new database queries.

Respond ONLY with the structured OrchestratorResult, including the appropriate action type and chart_type if relevant.'''
    }


# --- Define the Agent Blueprints (Prompts, Tools, Validators - Keep as is) ---

# System prompt generator function remains global
def generate_system_prompt() -> str:
    """Generates the full system prompt for the query agent."""
    prompt = f"""You are an expert SQL assistant designed to help users query the database.

TARGET DATABASE:
The target database (SQLite) details are provided in each request.

IMPORTANT OUTPUT STRUCTURE:
You MUST return your response as a valid QueryResponse object with these fields:
1. text_message: A human-readable response explaining your findings and analysis.
2. sql_result: For any data retrieval query, you MUST include this field with the SQL query string.
3. python_result: OPTIONAL. This field is ONLY populated by the output of the Python agent tool IF it was called.

CRITICAL RULES FOR SQL GENERATION:
1. For ANY question about data in the database (counts, totals, listings, comparisons, etc.), you MUST generate an appropriate SQLite query.
2. ALWAYS GENERATE SQL: If the user is asking about specific data, records, amounts, or numbers, you MUST generate SQL - even if you're unsure about exact column names. Missing SQL is a critical error.
3. PAY ATTENTION TO COLUMN NAMES: If a column name in the provided schema contains spaces or special characters, you MUST enclose it in double quotes (e.g., SELECT \"Total IFC Investment Amount\" FROM ...). Failure to quote such names will cause errors. Check for columns like \"IFC investment for Risk Management(Million USD)\", \"IFC investment for Guarantee(Million USD)\", etc.
4. AGGREGATIONS: For questions asking about totals, sums, or aggregations, use SQL aggregate functions (SUM, COUNT, AVG, etc.).
5. GROUPING: When a question mentions \"per\" some field (e.g., \"per product line\"), this requires a GROUP BY clause for that field.
6. SUM FOR TOTALS: Numerical fields asking for totals must use SUM() in your query. Ensure you select the correct column (e.g., \"IFC investment for Loan(Million USD)\" for loan sizes).
7. DATA TYPES: Be mindful that many numeric columns might be stored as TEXT (e.g., \"(Million USD)\" columns). You might need to CAST them to a numeric type (e.g., CAST(\"IFC investment for Loan(Million USD)\" AS REAL)) before performing calculations like AVG or SUM. Handle potential non-numeric values gracefully if possible (e.g., WHERE clause to filter them out before casting, or use `IFNULL(CAST(... AS REAL), 0)`).
8. SECURITY: ONLY generate SELECT queries. NEVER generate SQL statements that modify the database (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.).

PYTHON AGENT TOOL:
- You have access to a 'Python agent tool'.
- If the user's request requires any data manipulation, analysis, or visualization using Python (e.g., using pandas, numpy, plotting charts like bar, line, pie), you MUST call this tool.
- Pass the user's request and any necessary context (like the target database or previous results if applicable) to the Python agent tool.
- Do NOT attempt to generate or execute Python code yourself.
- The Python agent tool will handle the Python execution and visualization, and its result (code, explanation, visualization info) should be included in your final response if the tool was called.

RESPONSE STRUCTURE:
1. First, review the database schema provided in the message.
2. Understand the request: Does it require data retrieval (SQL)? Does it require Python manipulation/visualization (call Python agent tool)? Or just a textual answer?
3. Generate SQL: If data retrieval is needed, generate an accurate SQLite query string following the rules above.
4. Call Python Agent Tool (if needed): If the request involves Python analysis or visualization, call the Python agent tool. Include its results in the final response (potentially in the python_result field).
5. Explain Clearly: Explain the SQL query (including any casting). If the Python agent tool was called, summarize its findings or indicate that a visualization was prepared.
6. Format Output: Format your final response using the 'QueryResponse' structure. Include 'text_message', 'sql_result' (if SQL was generated), and 'python_result' (ONLY if the Python agent tool was called and returned relevant info).
7. **CRUCIAL**: Even if you use internal tools (like `execute_sql`) to find the answer or validate the query during your thought process, the final `QueryResponse` object you return MUST contain the generated SQL query string in the `sql_result` field if the original request required data retrieval from the database.
8. Safety: Focus ONLY on SELECT queries. Do not generate SQL/Python that modifies the database.
9. Efficiency: Write efficient SQL queries. Filter data (e.g., using WHERE clause) before aggregation.

FINAL OUTPUT FORMAT - VERY IMPORTANT:
Your final output MUST be the structured 'QueryResponse' object.
"""
    return prompt


# Agent Tool function remains global
async def execute_sql(ctx: RunContext[AgentDependencies], query: str) -> Union[List[Dict], str]:
    """Executes SQL and returns results."""
    logger.info(f"Executing SQL query: {query[:100]}...")
    if not ctx.deps or not ctx.deps.db_connection:
        logger.error("execute_sql called with no db_connection")
        return "Error: No database connection available."
    
    try:
        # Execute SQL
        cursor = ctx.deps.db_connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert to list of dicts for easier processing
        columns = [col[0] for col in cursor.description]
        result_dicts = [dict(zip(columns, row)) for row in results]
        
        # Save results to session state for visualization
        if result_dicts and len(result_dicts) > 0:
            import pandas as pd
            df = pd.DataFrame(result_dicts)
            st.session_state.last_dataframe = df
            
            # Store the database key if available
            if hasattr(ctx, 'deps') and hasattr(ctx.deps, 'db_key'):
                st.session_state.last_db_key = ctx.deps.db_key
            elif 'target_db_key' in st.session_state:
                st.session_state.last_db_key = st.session_state.target_db_key
            else:
                st.session_state.last_db_key = "unknown"
                
            logger.info(f"Stored query result in session state: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return result_dicts
    except Exception as e:
        error_msg = f"SQL Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# Agent Validation function remains global
async def validate_query_result(ctx: RunContext[AgentDependencies], result: QueryResponse) -> QueryResponse:
    """
    Validate the generated response. Checks SQL syntax, security, missing elements.
    Raises ModelRetry if validation fails.
    """
    user_message = ctx.prompt # The message sent to the agent (includes schema and user query)
    logger.info(f"Running result validation for prompt: {user_message[:100]}...")

    # --- SQL Validation ---
    if result.sql_result:
        if not ctx.deps or not ctx.deps.db_connection:
             logger.warning("SQL validation skipped: No database connection available in context.")
        else:
            original_sql = result.sql_result.sql_query
            cleaned_sql = original_sql.replace('\\', '').strip()
            if cleaned_sql != original_sql:
                logger.info(f"Cleaned SQL query. Original: '{original_sql[:100]}...', Cleaned: '{cleaned_sql[:100]}...'")
                result.sql_result.sql_query = cleaned_sql
            else:
                cleaned_sql = original_sql

            # --- Enhanced Security Validation (Duplicated from execute_sql for defense-in-depth) ---
            forbidden_commands = ['ALTER', 'CREATE', 'DELETE', 'DROP', 'INSERT', 'UPDATE', 'PRAGMA',
                                'ATTACH', 'DETACH', 'VACUUM', 'GRANT', 'REVOKE', 'EXECUTE', 'TRUNCATE']
            normalized_query = ' '.join([
                line for line in cleaned_sql.upper().split('\n')
                if not line.strip().startswith('--')
            ])

            if not normalized_query.strip().startswith("SELECT"):
                logger.warning(f"Non-SELECT query generated (validation): {cleaned_sql[:100]}...")
                raise ModelRetry("Only SELECT queries are allowed for security reasons. Please regenerate a proper SELECT query.")

            for cmd in forbidden_commands:
                 if f" {cmd} " in f" {normalized_query} " or normalized_query.startswith(f"{cmd} "):
                    logger.warning(f"Detected forbidden SQL command '{cmd}' in validation: {cleaned_sql[:100]}...")
                    raise ModelRetry(f"The SQL query contains a potentially harmful command '{cmd}'. Only SELECT statements are allowed. Please regenerate the query without this command.")

            if ';' in cleaned_sql.rstrip(';'):
                logger.warning(f"Multiple SQL statements detected (validation): {cleaned_sql[:100]}...")
                raise ModelRetry("Multiple SQL statements are not allowed. Please provide a single SELECT query.")

            # Validate SQL Syntax using EXPLAIN QUERY PLAN
            try:
                query_to_explain = cleaned_sql.rstrip(';')
                if query_to_explain:
                    cursor = ctx.deps.db_connection.cursor()
                    # Note: EXPLAIN itself doesn't execute the inner query, generally safe.
                    cursor.execute(f"EXPLAIN QUERY PLAN {query_to_explain}")
                    cursor.fetchall()
                    logger.info("Generated SQL query syntax validated successfully.")
                else:
                    logger.warning("Empty SQL query after cleaning, skipping EXPLAIN.")

            except sqlite3.Error as e:
                error_detail = f"SQL Syntax Validation Error: {e}. Query: {cleaned_sql[:100]}..."
                logger.error(error_detail, exc_info=False) # Less verbose trace for syntax error
                logger.warning(f"Raising ModelRetry due to SQL Syntax Error. Response: text='{result.text_message[:50]}...', sql='{cleaned_sql[:100]}...'")
                raise ModelRetry(f"The generated SQL query has invalid syntax: {str(e)}. Please check quoting of names, data types, and function usage, then correct the SQL query.") from e
            except Exception as e:
                error_detail = f"Unexpected SQL Validation Error: {e}. Query: {cleaned_sql[:100]}..."
                logger.error(error_detail, exc_info=True)
                logger.warning(f"Raising ModelRetry due to Unexpected SQL Error. Response details: text='{result.text_message[:50]}...', sql='{cleaned_sql[:100]}...'")
                raise ModelRetry(f"An unexpected error occurred during SQL validation: {str(e)}. Please try generating the SQL query again.") from e

    # --- Check for Missing SQL when Expected ---
    # (Keep the logic for checking keywords vs. actual SQL presence as is)
    data_query_keywords = ['total', 'sum', 'average', 'count', 'list', 'show', 'per', 'group', 'compare', 'what is', 'how many', 'which', 'top', 'approved', 'loan', 'loans', 'amount', 'india']
    user_question_marker = "User Request:"
    user_question_index = user_message.find(user_question_marker)
    original_user_question = user_message[user_question_index + len(user_question_marker):].strip() if user_question_index != -1 else user_message

    if not result.sql_result and any(keyword in original_user_question.lower() for keyword in data_query_keywords):
        is_greeting = any(greet in original_user_question.lower()[:15] for greet in ['hello', 'hi ', 'thanks', 'thank you'])
        is_meta_query = any(kw in original_user_question.lower() for kw in ['explain what is', 'how does', 'tell me more about', 'can you describe'])
        ai_gave_error_reason = "invalid request" in result.text_message.lower() or "cannot process" in result.text_message.lower()

        # Additional check for common database query patterns
        is_database_query = (
            ('ibrd' in original_user_question.lower() and ('loan' in original_user_question.lower() or 'india' in original_user_question.lower())) or
            ('show me' in original_user_question.lower()) or
            ('sum of' in original_user_question.lower()) or
            ('total' in original_user_question.lower() and ('amount' in original_user_question.lower() or 'loan' in original_user_question.lower()))
        )

        if (not is_greeting and not is_meta_query and not ai_gave_error_reason) or is_database_query:
            logger.warning(f"SQL result is missing, but keywords suggest it might be needed for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Missing SQL. Response: text='{result.text_message[:50]}...', sql=None")
            raise ModelRetry("The user's question seems to require data retrieval (based on keywords like 'compare', 'average', 'top', 'total', 'list', or specific database entities like 'IBRD', 'loans', 'India'), but no SQL query was generated. Please generate the appropriate SQL query.")


    # --- Visualization Validation ---
    # (Keep the logic for checking visualization requests vs. agent response as is)
    visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation', 'bar chart', 'pie chart', 'histogram', 'line graph', 'scatter plot']
    decline_phrases = ['cannot plot', 'unable to plot', 'cannot visualize', 'unable to visualize', 'cannot create chart', 'unable to create chart', 'do not have the ability to create plots']
    chart_suggestion_phrases = ["bar chart", "line chart", "pie chart", "scatter plot", "area chart"] # Expected suggestions
    is_visualization_request = any(keyword in original_user_question.lower() for keyword in visualization_keywords)
    has_declined_visualization = any(phrase in result.text_message.lower() for phrase in decline_phrases)
    has_suggested_chart = any(phrase in result.text_message.lower() for phrase in chart_suggestion_phrases)

    if is_visualization_request:
        if has_declined_visualization:
            logger.warning(f"Visualization requested, but AI declined capability for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Visualization Declined. Response: text='{result.text_message[:50]}...'")
            raise ModelRetry("The response incorrectly stated an inability to create visualizations. You MUST suggest an appropriate chart type (e.g., 'bar chart', 'line chart') in your text message and generate the necessary SQL query.")

        if not result.sql_result:
            logger.warning(f"Visualization requested, but SQL query is missing for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Missing SQL for Viz. Response: text='{result.text_message[:50]}...'")
            raise ModelRetry("The user requested a visualization, but the SQL query needed to fetch the data was missing. Please generate the appropriate SQL query and suggest a chart type in your text message.")

        if not has_suggested_chart:
            logger.warning(f"Visualization requested, SQL provided, but no chart type suggested in text for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Missing Chart Suggestion. Response: text='{result.text_message[:50]}...', sql='{result.sql_result.sql_query[:50] if result.sql_result else None}...', python='{'present' if result.python_result else 'absent'}'")
            if result.python_result:
                 raise ModelRetry("The user requested a visualization, and you provided SQL and Python data preparation code. However, you MUST also explicitly suggest the chart type (e.g., 'bar chart', 'line chart') in your text message.")
            else:
                 raise ModelRetry("The user requested a visualization, and you provided the SQL query. However, you MUST also explicitly suggest the chart type (e.g., 'bar chart', 'line chart') in your text message.")


    logger.info("Result validation completed successfully.")
    return result


# --- Blueprint Functions (Return Agent Instances) ---
# IMPORTANT: These functions now just define *how* to create an agent.
# The actual instantiation happens inside the async functions later.

def create_table_selection_agent_blueprint():
    """Returns the CONFIGURATION for the Table Selection Agent."""
    # Note: No model instance passed here anymore
    return {
        "result_type": SuggestedTables,
        "name": "Table Selection Agent",
        "retries": 2,
        "system_prompt": """You are an expert database assistant. Your task is to analyze a user's query and a list of available tables (with descriptions) in a specific database.
Identify the **minimum set of table names** from the provided list that are absolutely required to answer the user's query.
Consider the table names and their descriptions carefully.
Output ONLY the list of relevant table names and a brief reasoning."""
    }

def create_query_agent_blueprint():
    """Returns the CONFIGURATION for the main Query Agent."""
    # Note: No model instance passed here anymore
    return {
        "deps_type": AgentDependencies,
        "result_type": QueryResponse,
        "name": "SQL and Visualization Assistant",
        "retries": 3,
        "system_prompt_func": generate_system_prompt, # Pass the function itself
        "tools": [execute_sql, visualize_last_dataframe], # Add the visualizer tool here
        "result_validator_func": validate_query_result # Pass the validator function
    }

def create_column_prune_agent_blueprint():
    """Returns the CONFIGURATION for the Column Pruning Agent."""
    # Note: No model instance passed here anymore
    return {
        "result_type": PrunedSchemaResult,
        "name": "Column Pruning Agent",
        "retries": 2,
        "system_prompt": """You are an expert data analyst assistant. Your task is to prune the schema of a database to include only essential columns for a given user query.

IMPORTANT: The schema of the database will be provided at the beginning of each user message. Use this schema information to understand the database structure and generate an accurate pruned schema string. DO NOT respond that you need to know the table structure - it is already provided in the message.

CRITICAL RULES FOR SCHEMA PRUNING (Focus on identifying relevant columns):
1. Identify which columns are needed based on the user's query (e.g., columns mentioned, columns needed for filtering, aggregation, or grouping).
2. Pay attention to specific column names requested or implied by the query.
3. Understand the intent of the query and include columns that would be needed to make sense of the query.

YOUR GOAL is to output a concise schema string containing ONLY the necessary tables and columns.

RESPONSE STRUCTURE:
1. Review the full schema provided.
2. Analyze the user query to determine essential columns.
3. Generate the `pruned_schema_string` containing only the required columns.
4. Provide a brief `explanation` of why these columns were kept.
5. Format your final response using the 'PrunedSchemaResult' structure.

Do NOT generate SQL queries for the user's request. Do NOT generate Python code.
Your only task is SCHEMA PRUNING.
"""
    }


# --- Metadata and Schema Functions --- (Keep as is)
METADATA_PATH = Path(__file__).parent / "assets" / "database_metadata.json"

@st.cache_data
def load_db_metadata(path: Path = METADATA_PATH) -> Optional[Dict]:
    """Loads the database metadata from the specified JSON file."""
    if not path.exists():
        st.error(f"Metadata file not found: {path}")
        logger.error(f"Metadata file not found: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"Successfully loaded database metadata from {path}")
        return metadata
    except json.JSONDecodeError as e:
        st.error(f"Error decoding metadata JSON from {path}: {e}")
        logger.error(f"Error decoding metadata JSON from {path}: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"Error loading metadata file {path}: {e}")
        logger.error(f"Error loading metadata file {path}: {e}", exc_info=True)
        return None

def format_schema_for_selected_tables(metadata: Dict, db_key: str, selected_table_names: List[str]) -> str:
    """Formats the schema string for a specific list of tables within a database key."""
    if 'databases' not in metadata or db_key not in metadata['databases']:
        logger.error(f"Database key '{db_key}' not found in metadata.")
        return f"Error: Database key '{db_key}' not found in metadata."

    db_entry = metadata['databases'][db_key]
    schema_parts = []
    tables = db_entry.get("tables", {})
    if not tables:
         logger.warning(f"No tables found in metadata for database {db_key}.")
         return f"No tables found in metadata for database {db_key}." # Return error string

    found_tables = False
    for table_name in selected_table_names:
        if table_name in tables:
            found_tables = True
            table_info = tables[table_name]
            schema_parts.append(f"\nTable: {table_name}")
            table_desc = table_info.get("description")
            if table_desc:
                schema_parts.append(f"  (Description: {table_desc})") # Keep description concise

            columns = table_info.get("columns", {})
            if columns:
                schema_parts.append("  Columns:")
                for col_name, col_info in columns.items():
                    col_type = col_info.get("type", "TEXT") # Default to TEXT if missing
                    col_desc = col_info.get("description", "")
                    # Ensure column names with spaces/special chars are quoted for the LLM's reference
                    # The LLM needs to use these quotes in the generated SQL
                    schema_parts.append(f'    - "{col_name}" ({col_type}){" - " + col_desc if col_desc else ""}')
            else:
                schema_parts.append("    - (No columns found in metadata for this table)")
        else:
            logger.warning(f"Table '{table_name}' selected by TableAgent not found in metadata for DB '{db_key}'.")

    if not found_tables:
        logger.warning(f"None of the selected tables ({', '.join(selected_table_names)}) were found in metadata for DB '{db_key}'.")
        # Return error string if none found
        return f"Error: None of the selected tables ({', '.join(selected_table_names)}) were found in the metadata for database {db_key}."

    return "\n".join(schema_parts)


def get_table_descriptions_for_db(metadata: Dict, db_key: str) -> str:
    """Generates a string listing table names and descriptions for the TableAgent."""
    if 'databases' not in metadata or db_key not in metadata['databases']:
        logger.error(f"Database key '{db_key}' not found in metadata for descriptions.")
        return f"Error: Database key '{db_key}' not found in metadata."

    db_entry = metadata['databases'][db_key]
    tables = db_entry.get("tables", {})
    if not tables:
         logger.warning(f"No tables found in metadata for database {db_key} to describe.")
         return f"No tables found in metadata for database {db_key}." # Return error string

    desc_parts = [f"Available Tables in Database '{db_entry.get('database_name', db_key)}' ({db_key}):"]
    for table_name, table_info in tables.items():
        desc = table_info.get("description", "No description available.")
        # Quote table name for clarity in the prompt
        desc_parts.append(f'- "{table_name}": {desc}')

    return "\n".join(desc_parts)


# --- Async Functions for Core Logic ---

async def identify_target_database(user_query: str, metadata: Dict) -> Tuple[Optional[str], str, List[str]]:
    """
    Identifies which database (IFC or MIGA) the user query is most likely referring to.
    Returns a tuple (database_key, reasoning, available_keys).
    'database_key' can be a valid key, None (for critical errors), or "USER_SELECTION_NEEDED".
    Instantiates its own LLM and Agent.
    """
    logger.info(f"Attempting to identify target database for query: {user_query[:50]}...")
    # Extract database descriptions and keys
    if 'databases' not in metadata:
        logger.error("Metadata missing 'databases' key.")
        return None, "Error: 'databases' key missing in metadata configuration.", []
    descriptions = []
    valid_keys = []
    for key, db_info in metadata['databases'].items():
        desc = db_info.get('description', f'Database {key}')
        descriptions.append(f"- {key}: {desc}")
        valid_keys.append(key)
    if not descriptions:
         logger.error("No databases found in metadata to classify against.")
         return None, "Error: No databases found in metadata to classify against.", []
    descriptions_str = "\n".join(descriptions)
    valid_keys_str = ", ".join([f"'{k}'" for k in valid_keys]) + ", or 'UNKNOWN'"
    classification_prompt = f"""Given the user query and the descriptions of available databases, identify which database the query is most likely related to.

Available Databases:
{descriptions_str}

User Query: "{user_query}"

Based *only* on the query and the database descriptions, which database key ({valid_keys_str}) is the most relevant target? If the query is ambiguous, unrelated to these specific databases, or a general greeting/request (like 'hello', 'thank you'), classify it as 'UNKNOWN'.
"""
    logger.info("--- Sending classification request to LLM ---")
    logger.debug(f"Prompt:\\n{classification_prompt}") # DEBUG level if needed
    logger.info("--------------------------------------------")

    try:
        # --- Instantiate Model and Agent LOCALLY ---
        global google_api_key
        try:
            # Ensure configuration happens just before use in this async context
            genai.configure(api_key=google_api_key)
            logger.info("Configured GenAI SDK within identify_target_database.")
        except Exception as config_err:
             logger.error(f"Failed to configure GenAI SDK for classification: {config_err}", exc_info=True)
             # Return None for critical config errors
             return None, f"Internal Error: Failed to configure AI Service ({config_err}).", []

        gemini_model_name = st.secrets.get("GEMINI_CLASSIFICATION_MODEL", "gemini-1.5-flash")
        local_llm = GeminiModel(model_name=gemini_model_name)
        logger.info(f"Instantiated GeminiModel: {gemini_model_name} for classification.")

        classifier_agent = Agent(
            local_llm,
            result_type=DatabaseClassification,
            name="Database Classifier",
            retries=2, # Fewer retries for classification
            system_prompt="You are an AI assistant that classifies user queries based on provided database descriptions. Output ONLY the structured classification result."
        )
        logger.info("Classifier agent created locally.")
        # --- End Instantiation ---

        classification_result = None
        try:
            logger.info("Running database classifier agent")
            
            # Use run_async_task instead of directly awaiting to handle event loop issues
            async def run_classifier():
                return await classifier_agent.run(classification_prompt)
            
            classification_result = run_async_task(run_classifier)
            
            logger.info("Classification agent run completed.")
        except Exception as e:
            logger.error(f"Classification agent.run() failed: {str(e)}", exc_info=True)
            # If agent run fails, trigger user selection as we couldn't determine the DB
            return "USER_SELECTION_NEEDED", f"Could not automatically determine the database due to an internal error: {str(e)}. Please select one.", valid_keys

        # Process result
        if hasattr(classification_result, 'data') and isinstance(classification_result.output, DatabaseClassification):
            result_data: DatabaseClassification = classification_result.output
            logger.info("--- LLM Classification Result ---")
            logger.info(f"Key: {result_data.database_key}")
            logger.info(f"Reasoning: {result_data.reasoning}")
            logger.info("-------------------------------")

            if result_data.database_key == "UNKNOWN":
                logger.warning(f"LLM classified as UNKNOWN. Triggering user selection. Reasoning: {result_data.reasoning}")
                # Ask user to select
                return "USER_SELECTION_NEEDED", f"Could not automatically determine the target database (Reason: {result_data.reasoning}). Please select one:", valid_keys
            elif result_data.database_key in valid_keys:
                # Success
                return result_data.database_key, result_data.reasoning, valid_keys
            else:
                 logger.warning(f"LLM returned an invalid key: {result_data.database_key}. Triggering user selection.")
                 # Ask user to select
                 return "USER_SELECTION_NEEDED", f"AI classification returned an unexpected key '{result_data.database_key}'. Please select the correct database.", valid_keys
        else:
             logger.error(f"Classification call returned unexpected structure: {classification_result}. Triggering user selection.")
             # Ask user to select
             return "USER_SELECTION_NEEDED", "Failed to get a valid classification structure from the AI. Please select the database.", valid_keys

    except Exception as e:
        logger.exception("Error during database classification LLM call:")
        # General error during classification, trigger user selection
        return "USER_SELECTION_NEEDED", f"An error occurred during database classification: {str(e)}. Please select one.", valid_keys


def get_recent_turns_from_memory(memory, n=2):
    """Extract last n user/assistant turns from LangChain memory as a string."""
    history = memory.load_memory_variables({}).get("history", [])
    # If history is a string (older config), fallback to previous logic
    if isinstance(history, str):
        turns = [turn.strip() for turn in history.split('\n') if turn.strip()]
        return '\n'.join(turns[-2*n:])
    # If history is a list of LangChain message objects
    if not isinstance(history, list):
        return ""
    formatted = []
    for msg in history[-2*n:]:
        # LangChain message objects: HumanMessage, AIMessage, SystemMessage, etc.
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            formatted.append(f"User: {getattr(msg, 'content', '')}")
        elif msg_type == "AIMessage":
            formatted.append(f"Assistant: {getattr(msg, 'content', '')}")
    return "\n".join(formatted)

# --- Orchestrator Agent Routing ---
async def handle_user_message(message: str) -> None:
    """Handles user message, routes via orchestrator, then runs DB/Table selection."""
    start_time = time.time()
    logger.info(f"handle_user_message started for message: '{message[:50]}...'")
    
    # Add user message to history for context
    # st.session_state.chat_history.append({"role": "user", "content": message})  # <-- REMOVE THIS LINE

    # Check for visualization-specific keywords for follow-up requests
    visualization_keywords = [
        'chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization',
        'visualisation', 'bar chart', 'pie chart', 'histogram', 'line graph',
        'scatter plot', 'show me', 'display'
    ]
    
    # Only call handle_follow_up_chart for genuine follow-up requests:
    # - Short message
    # - Visualization keywords
    # - There is actually previous data to visualize
    is_short_message = len(message.split()) < 15
    last_is_assistant = (len(st.session_state.chat_history) > 1 and 
                         st.session_state.chat_history[-2].get("role") == "assistant")
    is_follow_up_chart_request = (
        is_short_message and
        last_is_assistant and
        any(keyword in message.lower() for keyword in visualization_keywords) and
        'last_chartable_data' in st.session_state and st.session_state.last_chartable_data is not None
    )

    # Shortcut for obvious visualization requests to bypass the classifier
    if is_follow_up_chart_request:
        logger.info("Detected follow-up visualization request directly")
        chart_type = "bar"  # Default
        for chart_keyword in ['bar', 'pie', 'line', 'scatter', 'histogram']:
            if chart_keyword in message.lower():
                chart_type = chart_keyword
                break
        await handle_follow_up_chart(message, chart_type)
        return

    # --- Orchestrator Agent Routing ---
    try:
        # Pass the last few user/assistant turns as context for better decision-making
        short_context = get_recent_turns_from_memory(st.session_state.lc_memory, n=3)  # Increase context to 3 turns
        global google_api_key
        import google.generativeai as genai
        try:
            genai.configure(api_key=google_api_key)
            logger.info("Configured GenAI SDK for orchestrator.")
        except Exception as config_err:
            logger.error(f"Failed to configure GenAI SDK for orchestrator: {config_err}", exc_info=True)
            raise RuntimeError(f"Internal Error: Failed to configure AI Service ({config_err}).") from config_err

        gemini_model_name = st.secrets.get("GEMINI_ORCHESTRATOR_MODEL", "gemini-2.0-flash")
        local_llm = GeminiModel(model_name=gemini_model_name)
        orchestrator_config = create_orchestrator_agent_blueprint()
        orchestrator_agent = Agent(local_llm, **orchestrator_config)
        # Register the Python agent tool function for the orchestrator
        orchestrator_agent.tool(call_python_agent)
        orchestrator_prompt = f"""Recent conversation:
{short_context}

User: {message}"""
        logger.info("Running orchestrator agent...")
        
        # Use run_async_task instead of directly awaiting to handle event loop issues
        async def run_orchestrator():
            return await orchestrator_agent.run(orchestrator_prompt)
        
        orchestrator_result = run_async_task(run_orchestrator)
        
        if hasattr(orchestrator_result, 'data'):
            # --- Modern LLM Orchestration: Combined SQL + Visualization Intent ---
            # Detect if the user query requests both data and visualization in one step
            if orchestrator_result.output.action == "visualization":
                logger.info("Orchestrator determined this is a visualization request")
                chart_type = orchestrator_result.output.chart_type or "bar"  # Default to bar if not specified

                # NEW: Check if the user query also contains data retrieval keywords (combined intent)
                data_query_keywords = [
                    'total', 'sum', 'average', 'count', 'list', 'show', 'per', 'group', 'compare', 'what is', 'how many', 'which', 'top', 'biggest', 'largest', 'most', 'least', 'highest', 'lowest', 'by', 'rank', 'distribution', 'breakdown'
                ]
                is_combined_intent = any(kw in message.lower() for kw in data_query_keywords)

                if is_combined_intent:
                    logger.info("Detected combined SQL + visualization intent. Running SQL and visualization in one workflow.")
                    # Run the full DB flow: DB selection, table selection, SQL, then visualization
                    # Set a flag in session state to indicate visualization is expected after SQL
                    st.session_state.combined_sql_visualization = {
                        'chart_type': chart_type,
                        'user_message': message
                    }
                    # Continue with the existing DB flow (will trigger visualization after SQL)
                    # (Do NOT return here)
                else:
                    # If not combined, handle as a follow-up visualization (requires previous data)
                    await handle_follow_up_chart(message, chart_type)
                    return
            elif orchestrator_result.output.action == "assistant":
                logger.info("Orchestrator determined this is an assistant/general help request")
                response = orchestrator_result.output.response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                st.session_state.lc_memory.save_context({"input": message}, {"output": response})
                logger.info(f"handle_user_message finished (assistant). Duration: {time.time() - start_time:.2f}s")
                return
            elif orchestrator_result.output.action == "database_query":
                logger.info("Orchestrator determined this is a database query")
                # Continue with the existing database query flow
            else:
                logger.warning(f"Unexpected action from orchestrator: {orchestrator_result.output.action}")
        else:
            logger.warning("No 'data' attribute in orchestrator result")
    except Exception as e:
        logger.error(f"Error in orchestrator processing: {str(e)}", exc_info=True)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"I'm sorry, I encountered an error understanding your request. Please try rephrasing your question. Error: {str(e)}"
        })
        return

    # --- Continue with existing DB flow (database classifier, etc) ---
    assistant_chat_message = None
    target_db_key = None
    try:
        logger.info("Step 1: Loading database metadata...")
        db_metadata = load_db_metadata()
        if not db_metadata:
            logger.error("Metadata loading failed. Cannot proceed.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Sorry, I couldn't load the database configuration. Please check the application setup."
            })
            return # Exit
        logger.info("Step 1: Database metadata loaded successfully.")
        logger.info("Step 2: Identifying target database...")
        # Pass only the current user query to the classifier agent
        identified_key, reasoning, available_keys = await identify_target_database(message, db_metadata)
        logger.info(f"Step 2: Database identification result - Key: {identified_key}, Reasoning: {reasoning}")
        if identified_key == "USER_SELECTION_NEEDED":
            logger.warning(f"Database identification requires user input. Reason: {reasoning}")
            st.session_state.db_selection_pending = True
            st.session_state.db_selection_reason = reasoning
            st.session_state.pending_db_keys = available_keys
            st.session_state.pending_user_message = message
            st.session_state.pending_db_metadata = db_metadata
            logger.info("Step 2: Pausing for database confirmation. State saved.")
            # DO NOT return here - let finally block run
        elif identified_key is None:
            logger.error(f"Critical error during database identification. Cannot proceed. Reason: {reasoning}")
            assistant_chat_message = {
                "role": "assistant",
                "content": f"Sorry, a critical error occurred while identifying the database: {reasoning}"
            }
            st.session_state.chat_history.append(assistant_chat_message)
            # DO NOT return here - let finally block run
        else:
            target_db_key = identified_key
            logger.info(f"Step 2: Target database confirmed as: {target_db_key}")
            st.session_state.last_db_key = target_db_key
            logger.info(f"Step 3: Proceeding to table selection stage for DB: {target_db_key}")
            # Await the table selection stage directly
            await run_table_selection_stage(message, target_db_key, db_metadata)
            logger.info("Step 3: run_table_selection_stage finished.")
            # run_table_selection_stage will set the state for confirmation pause
            # DO NOT return here - let finally block run

    except Exception as e:
        error_msg = f"A critical error occurred during initial message processing: {str(e)}"
        logger.exception("Critical error in handle_user_message setup:")
        if not assistant_chat_message: # Avoid overwriting specific identification error
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, a critical internal error occurred: {str(e)}"}
        st.session_state.chat_history.append(assistant_chat_message)
    finally:
        # This block runs regardless of whether the try block completed successfully,
        # raised an exception, or paused for confirmation.
        logger.info(f"handle_user_message finished initial stage (or error/pause). Duration: {time.time() - start_time:.2f}s")
        # Do not return here explicitly; control flow continues to Streamlit rerun naturally

# --- Table Selection Stage ---
async def run_table_selection_stage(
    user_message: str,
    target_db_key: str,
    db_metadata: Dict
):
    """Runs the table selection agent and sets state for user confirmation."""
    start_time = time.time()
    logger.info(f"run_table_selection_stage started for DB: {target_db_key}, Query: '{user_message[:50]}...'")
    assistant_chat_message = None # To hold potential error messages
    try:
        logger.info(f"Getting database path for key: {target_db_key}...")
        db_entry = db_metadata.get('databases', {}).get(target_db_key)
        if not db_entry or 'database_path' not in db_entry:
            error_msg = f"Metadata configuration error: Could not find path for database '{target_db_key}'."
            logger.error(error_msg)
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, internal configuration error for database {target_db_key}."}
            st.session_state.chat_history.append(assistant_chat_message)
            return # Exit on critical config error

        target_db_path = db_entry['database_path']
        logger.info(f"Database path found: {target_db_path}")
        logger.info(f"Running Table Selection Agent for DB: {target_db_key}...")
        table_descriptions = get_table_descriptions_for_db(db_metadata, target_db_key)

        if table_descriptions.startswith("Error:") or table_descriptions.startswith("No tables found"):
            logger.error(f"Could not get table descriptions for TableAgent: {table_descriptions}")
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, couldn't retrieve table list for the {target_db_key} database."}
            st.session_state.chat_history.append(assistant_chat_message)
            return # Exit on error getting descriptions

        table_agent_prompt = f"""User Query: "{user_message}"

Database: {target_db_key}
{table_descriptions}

Based *only* on the user query and the table descriptions, which of the listed table names (keys like "investment_services_projects") are required? Output just the list and reasoning. If no specific tables seem relevant based *only* on the query and descriptions, return an empty list.
"""
        selected_tables = []
        table_agent_reasoning = ""

        try:
            logger.info("Calling Table Selection Agent...")
            logger.info("Instantiating LLM/Agent for table selection...")
            global google_api_key
            import google.generativeai as genai
            try:
                # Configure just before use
                genai.configure(api_key=google_api_key)
                logger.info("Configured GenAI SDK within table selection.")
            except Exception as config_err:
                logger.error(f"Failed to configure GenAI SDK for table selection: {config_err}", exc_info=True)
                raise RuntimeError(f"Internal Error: Failed to configure AI Service ({config_err}).") from config_err

            gemini_model_name = st.secrets.get("GEMINI_TABLE_SELECTION_MODEL", "gemini-1.5-flash")
            local_llm = GeminiModel(model_name=gemini_model_name)
            logger.info(f"Instantiated GeminiModel: {gemini_model_name} for table selection.")
            table_selection_agent_blueprint = create_table_selection_agent_blueprint()
            agent_instance = Agent(local_llm, **table_selection_agent_blueprint)
            logger.info("Table selection agent created locally.")

            # Only pass the user query and table descriptions (no message history)
            logger.info("Running table selection agent without message history")
            
            # Use run_async_task instead of directly awaiting to handle event loop issues
            async def run_table_agent():
                return await agent_instance.run(table_agent_prompt)
            
            table_agent_result = run_async_task(run_table_agent)
            
            logger.info("Table selection agent run completed.")

            if table_agent_result and hasattr(table_agent_result, 'data') and isinstance(table_agent_result.output, SuggestedTables):
                selected_tables = table_agent_result.output.table_names
                table_agent_reasoning = table_agent_result.output.reasoning
                logger.info(f"Table Agent suggested tables: {selected_tables}. Reasoning: {table_agent_reasoning}")
            elif not selected_tables: # Handle case where agent returns empty list or unexpected result
                logger.warning(f"Table Selection Agent returned no specific tables or failed. Result: {table_agent_result}")
                if not table_agent_reasoning: # Provide default reasoning if none given
                    table_agent_reasoning = "Could not automatically identify specific tables for your query. Please select the tables needed."
                selected_tables = [] # Ensure it's an empty list

        except Exception as e:
            logger.exception("Error running Table Selection Agent:")
            table_agent_reasoning = f"An error occurred during table selection: {str(e)}. Please select the tables needed."
            selected_tables = [] # Ensure empty list on exception

        # --- Set State for Confirmation ---
        # This state will be picked up by the UI in the next Streamlit rerun
        st.session_state.table_confirmation_pending = True
        st.session_state.pending_user_message = user_message
        st.session_state.pending_db_metadata = db_metadata
        st.session_state.pending_target_db_key = target_db_key
        st.session_state.pending_target_db_path = target_db_path
        db_entry = db_metadata.get('databases', {}).get(target_db_key)
        all_tables = list(db_entry.get("tables", {}).keys()) if db_entry else []
        st.session_state.all_tables = all_tables
        # Set candidate tables based on agent result (might be empty)
        st.session_state.candidate_tables = selected_tables
        st.session_state.table_agent_reasoning = table_agent_reasoning
        logger.info("Pausing for table confirmation. State saved.")
        # No return needed here, state is set, Streamlit will rerun

    except Exception as e:
        error_msg = f"A critical error occurred during table selection stage: {str(e)}"
        logger.exception("Critical error in run_table_selection_stage:")
        # Ensure an error message is shown if something went wrong before setting confirmation state
        if not st.session_state.get("table_confirmation_pending"):
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, a critical internal error occurred during table selection: {str(e)}"}
            st.session_state.chat_history.append(assistant_chat_message)
            clear_pending_state() # Clear state on critical error
    finally:
        logger.info(f"run_table_selection_stage finished. Duration: {time.time() - start_time:.2f}s")


async def handle_follow_up_chart(message: str, chart_type: str = "bar"):
    """Handle a follow-up request to visualize the last query result as a chart."""
    logger.info(f"Handling follow-up chart request with chart type: {chart_type}")
    
    # Check if we have a last result to visualize
    if 'last_chartable_data' not in st.session_state or st.session_state.last_chartable_data is None:
        logger.warning("No previous query result found to visualize")
        response = "I don't have any data from a previous query to visualize. Please run a database query first, and then I can create a chart from the results."
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        return
    
    try:
        db_key = st.session_state.get('last_chartable_db_key', 'unknown')
        df = st.session_state.last_chartable_data
        # Generate DataFrame summary
        df_summary = summarize_dataframe(df)
        # Prepare context for the Python agent
        visualization_prompt = f"""Create a {chart_type} chart from the most recent query results.\nData summary:\n{df_summary}\nUser request: {message}\nDatabase: {db_key}"""
        class SimpleContext:
            def __init__(self, model, deps=None):
                self.model = model
                self.deps = deps
        class SimpleDeps:
            def __init__(self, db_key):
                self.db_key = db_key
        local_llm = GeminiModel(model_name=st.secrets.get("GEMINI_MAIN_MODEL", "gemini-1.5-flash"))
        simple_deps = SimpleDeps(db_key)
        ctx = SimpleContext(local_llm, simple_deps)
        
        # Use run_async_task instead of directly awaiting to handle event loop issues
        async def run_python_agent():
            return await call_python_agent(ctx, visualization_prompt)
        
        viz_result = run_async_task(run_python_agent)
        
        if hasattr(viz_result, 'data') and hasattr(viz_result.output, 'visualization') and viz_result.output.visualization:
            vis = viz_result.output.visualization
            if vis.success:
                response = f"Here's the {chart_type} chart of your data:"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "streamlit_chart": {
                        "type": chart_type,
                        "data": df  # Use the original dataframe
                    }
                })
            else:
                error_msg = vis.error or "Unknown error"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I couldn't create the {chart_type} chart. Error: {error_msg}"
                })
        else:
            content = "I've created a chart based on your data."
            if hasattr(viz_result, 'data') and hasattr(viz_result.output, 'explanation'):
                content = viz_result.output.explanation or content
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": content,
                "streamlit_chart": {
                    "type": chart_type,
                    "data": df  # Use the original dataframe
                }
            })
        logger.info(f"Chart response added to chat history")
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"I'm sorry, I couldn't create the {chart_type} chart due to an error: {str(e)}"
        })


async def run_agents_post_confirmation_inner(
    db_metadata: dict,
    selected_tables: list,
    target_db_key: str,
    target_db_path: str,
    user_message: str,
    agent_message_history: list # Note: This is currently passed but not used due to Gemini issues
) -> dict:
    """
    Runs the column pruning agent and main query agent after table confirmation.
    Returns a message dict to add to chat_history.
    Handles DB connection internally.
    """
    import pandas as pd
    import numpy as np
    deps = None
    final_assistant_message_dict = None
    start_inner_time = time.time()
    logger.info("run_agents_post_confirmation_inner started.")
    pruned_schema_string = None
    assistant_chat_message = None # For accumulating warnings/info before main result

    try:
        # Step 1: Format FULL schema for selected tables
        # (No spinner here, called from within async)
        logger.info(f"Formatting FULL schema for CONFIRMED tables: {selected_tables}...")
        full_schema_for_pruning = format_schema_for_selected_tables(db_metadata, target_db_key, selected_tables)

        if full_schema_for_pruning.startswith("Error:"):
            logger.error(f"Could not get valid full schemas for selected tables: {full_schema_for_pruning}")
            return {"role": "assistant", "content": f"Sorry, couldn't retrieve valid schema details for the selected tables ({', '.join(selected_tables)}) in the {target_db_key} database: {full_schema_for_pruning}"}

        # Step 2: Run Column Pruning Agent
        # (No spinner here)
        logger.info("Running Column Pruning Agent...")
        pruning_prompt = f'''User Query: "{user_message}"

Full Schema for Relevant Tables:
{full_schema_for_pruning}

Based *only* on the user query and the full schema provided, prune the schema string to include only essential columns. Output the pruned schema string and explanation.'''
        try:
            global google_api_key
            try:
                # Configure GenAI just before use
                genai.configure(api_key=google_api_key)
                logger.info("Configured GenAI SDK within column pruning.")
            except Exception as config_err:
                logger.error(f"Failed to configure GenAI SDK for column pruning: {config_err}", exc_info=True)
                raise RuntimeError(f"Internal Error: Failed to configure AI Service ({config_err}).") from config_err

            gemini_model_name = st.secrets.get("GEMINI_PRUNING_MODEL", "gemini-1.5-flash") # Use specific model if needed
            local_llm_prune = GeminiModel(model_name=gemini_model_name)
            agent_config_prune = create_column_prune_agent_blueprint()
            agent_instance_prune = Agent(local_llm_prune, **agent_config_prune)
            
            # Use run_async_task instead of directly awaiting to handle event loop issues
            async def run_prune_agent():
                return await agent_instance_prune.run(pruning_prompt)
            
            pruning_agent_result = run_async_task(run_prune_agent)

            if hasattr(pruning_agent_result, 'data') and isinstance(pruning_agent_result.output, PrunedSchemaResult):
                pruned_schema_string = pruning_agent_result.output.pruned_schema_string
                pruning_explanation = pruning_agent_result.output.explanation
                logger.info(f"Column Pruning Agent successful. Explanation: {pruning_explanation}")
            else:
                logger.warning(f"Column Pruning Agent returned unexpected structure: {pruning_agent_result}. Proceeding with FULL schema.")
                pruned_schema_string = full_schema_for_pruning
                # Add a note for the user in the final message
                assistant_chat_message = {"role": "assistant", "content": f"Note: Could not prune the schema effectively, proceeding with full schema for selected tables."}

        except Exception as prune_e:
            # Log the full exception details for debugging
            logger.exception("Error running Column Pruning Agent:")
            pruned_schema_string = full_schema_for_pruning # Fallback to full schema
            # Prepare a warning message for the user
            prune_error_warning = f"Warning: Encountered an error during schema pruning ({str(prune_e)}). Proceeding with the full schema for selected tables."
            if assistant_chat_message: # If a previous note exists, append
                 assistant_chat_message["content"] += f"\n{prune_error_warning}"
            else: # Otherwise, create the message
                 assistant_chat_message = {"role": "assistant", "content": prune_error_warning}

        # Ensure we have a schema string to proceed
        schema_to_use = pruned_schema_string if pruned_schema_string else full_schema_for_pruning
        if not schema_to_use:
             logger.error("Critical error: No schema string available after pruning attempt.")
             return {"role": "assistant", "content": "Sorry, a critical internal error occurred while preparing the database schema."}


        # Step 3: Run Main Query Agent with Pruned (or Full) Schema
        # (No spinner here)
        logger.info("Instantiating LLM/Agent for post-confirmation query...")
        try:
            # Configure GenAI just before use
            genai.configure(api_key=google_api_key)
            logger.info("Configured GenAI SDK within post-confirmation flow.")
        except Exception as config_err:
            logger.error(f"Failed to configure GenAI SDK post-confirmation: {config_err}", exc_info=True)
            return {"role": "assistant", "content": f"Internal Error: Failed to configure AI Service ({config_err})."}

        gemini_model_name = st.secrets.get("GEMINI_MAIN_MODEL", "gemini-1.5-flash") # Changed default to flash for consistency
        local_llm = GeminiModel(model_name=gemini_model_name)
        agent_config_bp = create_query_agent_blueprint()
        # Instantiate the agent
        local_query_agent = Agent(
            local_llm,
            deps_type=agent_config_bp["deps_type"],
            result_type=agent_config_bp["result_type"],
            name=agent_config_bp["name"],
            retries=agent_config_bp["retries"],
        )
        # Apply configurations from the blueprint
        local_query_agent.system_prompt(agent_config_bp["system_prompt_func"])
        for tool_func in agent_config_bp["tools"]:
            local_query_agent.tool(tool_func)
        # Register the Python agent tool function for the query agent
        local_query_agent.tool(call_python_agent)
        local_query_agent.result_validator(agent_config_bp["result_validator_func"])
        logger.info("Query agent created locally for post-confirmation flow.")


        # --- Database Connection ---
        logger.info(f"Connecting to database: {target_db_path} for key: {target_db_key}")
        deps = AgentDependencies.create().with_db(db_path=target_db_path)
        if not deps.db_connection:
            # deps.with_db already logged the error and st.error
            return {"role": "assistant", "content": f"Sorry, I couldn't connect to the {target_db_key} database at {target_db_path}."}
        logger.info("Database connection successful.")

        # --- Prepare Prompt ---
        usage = Usage() # Track token usage
        prompt_message = f'''Target Database: {target_db_key}
Pruned Database Schema (Only essential columns for the query):
{schema_to_use}

User Request: {user_message}'''

        visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation', 'bar chart', 'pie chart', 'histogram', 'line graph', 'scatter plot']
        is_visualization_request = any(keyword in user_message.lower() for keyword in visualization_keywords)

        if is_visualization_request:
            logger.info("Adding visualization instructions to prompt.")
            prompt_message += f"""
\nIMPORTANT: This is a visualization request for the {target_db_key} database.\n1. Generate the appropriate SQL query to retrieve the necessary data from the provided schema (remember to CAST text numbers if needed for aggregation).\n2. In your text response, you MUST suggest an appropriate chart type (e.g., \"bar chart\", \"line chart\", \"pie chart\") based on the user's request and the data.\n3. Do NOT generate Python code for plotting (e.g., using matplotlib or seaborn). Only generate Python code if data *preparation* (like setting index, renaming) beyond the SQL query is needed for the suggested chart.\n"""
        else:
            logger.info("Standard (non-visualization) request.")

        # --- LOG THE PROMPT ---
        logger.info(f"Prompt sent to LLM (main query agent):\n{prompt_message}")

        # --- Run Query Agent ---
        # IMPORTANT: DO NOT USE MESSAGE HISTORY WITH GEMINI - CAUSES INCOMPATIBILITY
        logger.info("Skipping message history for main query agent to avoid Gemini compatibility issues")
        agent_run_result = None
        try:
            logger.info("Running query agent without message history")
            
            # Use run_async_task instead of directly awaiting to handle event loop issues
            async def run_query_agent():
                return await local_query_agent.run(
                    prompt_message,
                    deps=deps,
                    usage=usage,
                    usage_limits=DEFAULT_USAGE_LIMITS
                    # No message_history parameter
                )
            
            agent_run_result = run_async_task(run_query_agent)
            
            run_duration = time.time() - start_inner_time # Recalculate duration accurately here
            logger.info(f"Query Agent call completed post-confirmation. Duration: {run_duration:.2f}s. Result type: {type(agent_run_result)}")
        except Exception as agent_run_e:
            # Catch specific agent run errors
            logger.error(f"Main Query Agent agent.run() failed: {str(agent_run_e)}", exc_info=True)
            # Return specific error message
            return {"role": "assistant", "content": f"Sorry, the AI agent failed to process your request for the {target_db_key} database. Error: {str(agent_run_e)}"}

        # --- Process Agent Result ---
        st.session_state.last_result = agent_run_result # For debugging if needed

        # Append new messages from agent run to internal history (still useful for logging/potential future use)
        if hasattr(agent_run_result, 'new_messages'):
            new_msgs = agent_run_result.new_messages()
            valid_new_msgs = filter_message_history(new_msgs) # Use validation helper
            st.session_state.agent_message_history.extend(valid_new_msgs)
            logger.info(f"Appended {len(valid_new_msgs)} validated new messages to agent_message_history (new total: {len(st.session_state.agent_message_history)}).")
        else:
            logger.warning("Agent result object does not have 'new_messages' attribute.")

        logger.info("Processing Query Agent response (after confirmation)...")
        if hasattr(agent_run_result, 'data') and isinstance(agent_run_result.output, QueryResponse):
            response: QueryResponse = agent_run_result.output
            logger.info("Agent response has expected QueryResponse structure.")
            logger.debug(f"AI Response Data: {response}") # Debug log

            # Start building the final message dict
            # Prepend any earlier warnings (e.g., from pruning)
            initial_content = f"[{target_db_key}] {response.text_message}"
            if assistant_chat_message and "content" in assistant_chat_message:
                initial_content = f"{assistant_chat_message['content']}\n\n{initial_content}"

            base_assistant_message = {"role": "assistant", "content": initial_content}
            logger.info(f"Base assistant message: {base_assistant_message['content'][:100]}...")

            sql_results_df = None
            sql_info = {}
            if response.sql_result:
                sql_query = response.sql_result.sql_query
                logger.info(f"SQL query generated: {sql_query[:100]}...")
                logger.info(f"Executing SQL query against {target_db_key}...")
                # Need a RunContext to call execute_sql tool function correctly
                sql_run_context = RunContext(
                    deps=deps, model=local_llm, usage=usage, prompt=sql_query
                )
                
                # Use run_async_task instead of directly awaiting to handle event loop issues
                async def run_sql_execution():
                    return await execute_sql(sql_run_context, sql_query)
                
                sql_execution_result = run_async_task(run_sql_execution)

                sql_info = {
                    "query": sql_query,
                    "explanation": response.sql_result.explanation
                }

                if isinstance(sql_execution_result, str) and sql_execution_result.startswith("Error:"):
                    sql_info["error"] = sql_execution_result
                    logger.error(f"SQL execution failed: {sql_execution_result}")
                    base_assistant_message["content"] += f"\n\n**Warning:** There was an error executing the SQL query: `{sql_execution_result}`"
                    sql_results_df = pd.DataFrame() # Ensure df is empty on error
                elif isinstance(sql_execution_result, list):
                    if sql_execution_result:
                        logger.info(f"SQL execution successful, {len(sql_execution_result)} rows returned.")
                        try:
                            sql_results_df = pd.DataFrame(sql_execution_result)
                            # Attempt numeric conversion (safer loop)
                            for col in sql_results_df.columns:
                                try:
                                    # Check if column looks numeric before trying conversion
                                    if sql_results_df[col].astype(str).str.match(r'^-?\d+(\.\d+)?$').all():
                                        sql_results_df[col] = pd.to_numeric(sql_results_df[col])
                                        logger.info(f"Converted column '{col}' to numeric.")
                                except (ValueError, TypeError, AttributeError) as conv_err:
                                     # Ignore columns that can't be converted or have mixed types
                                     logger.debug(f"Could not convert column '{col}' to numeric: {conv_err}. Keeping original type.")

                            sql_info["results"] = sql_results_df.to_dict('records')
                            sql_info["columns"] = list(sql_results_df.columns)
                            # Store data for potential follow-up charting
                            st.session_state.last_chartable_data = sql_results_df
                            st.session_state.last_chartable_db_key = target_db_key
                            logger.info(f"Stored chartable data (shape: {sql_results_df.shape}) for DB '{target_db_key}'.")
                        except Exception as df_e:
                            logger.error(f"Error creating/processing DataFrame from SQL results: {df_e}", exc_info=True)
                            sql_info["error"] = f"Error processing SQL results into DataFrame: {df_e}"
                            sql_results_df = pd.DataFrame() # Ensure empty df
                            st.session_state.last_chartable_data = None # Clear chartable data
                            st.session_state.last_chartable_db_key = None
                    else:
                        logger.info("SQL execution successful, 0 rows returned.")
                        sql_info["results"] = []
                        sql_results_df = pd.DataFrame() # Empty df
                        st.session_state.last_chartable_data = None # Clear chartable data
                        st.session_state.last_chartable_db_key = None
                else: # Should not happen if execute_sql returns correctly
                    sql_info["error"] = "Unexpected result type from SQL execution."
                    logger.error(f"{sql_info['error']} Type: {type(sql_execution_result)}")
                    sql_results_df = pd.DataFrame() # Empty df
                    st.session_state.last_chartable_data = None # Clear chartable data
                    st.session_state.last_chartable_db_key = None

                # Add sql_info block to the message dictionary
                base_assistant_message["sql_result"] = sql_info
                logger.info("Added sql_result block to assistant message.")
            else: # No SQL query generated
                logger.info("No SQL query was generated by the agent.")
                st.session_state.last_chartable_data = None
                st.session_state.last_chartable_db_key = None
                # Inform the user that no SQL query was generated
                base_assistant_message["content"] += "\n\n**Note:** No database query was executed. The answer is based on general knowledge or database schema information."


            # --- Python Code Execution (if present) ---
            df_for_chart = sql_results_df if sql_results_df is not None else pd.DataFrame() # Use potentially modified df
            python_info = {}
            if response.python_result:
                logger.info("Python code generated for data preparation...")
                python_code = response.python_result.python_code
                python_info = {
                    "code": python_code,
                    "explanation": response.python_result.explanation
                }
                logger.info(f"Python code to execute:\n{python_code}")
                # Execute in a restricted scope
                local_vars = {
                    'pd': pd,
                    'np': np,
                    'df': df_for_chart.copy() # Operate on a copy
                }
                try:
                    exec(python_code, {"pd": pd, "np": np}, local_vars) # Pass globals explicitly
                    # Check if 'df' was modified and is still a DataFrame
                    if 'df' in local_vars and isinstance(local_vars['df'], pd.DataFrame):
                        df_for_chart = local_vars['df'] # Update df_for_chart with result
                        logger.info(f"Python data preparation executed. DataFrame shape after exec: {df_for_chart.shape}")
                        # Update chartable data if Python modified it
                        st.session_state.last_chartable_data = df_for_chart
                        logger.info("Updated last_chartable_data with DataFrame from Python exec.")
                    else:
                        logger.warning(f"'df' variable after Python code is missing or not a DataFrame. Type: {type(local_vars.get('df'))}")
                        python_info["warning"] = "Python code did not produce or update a DataFrame named 'df'."
                except Exception as exec_e:
                    logger.error(f"Error executing Python data preparation code: {exec_e}\nCode:\n{python_code}", exc_info=True)
                    python_info["error"] = str(exec_e)
                    base_assistant_message["content"] += f"\n\n**Warning:** There was an error executing the provided Python code: `{exec_e}`"
                # Add python_info block to the message dictionary
                base_assistant_message["python_result"] = python_info
                logger.info("Added python_result block to assistant message.")


            # --- Chart Generation Check ---
            chart_type = None
            # Use the potentially modified df_for_chart
            if df_for_chart is not None and not df_for_chart.empty:
                text_lower = response.text_message.lower()
                # Check for chart suggestions
                if "bar chart" in text_lower: chart_type = "bar"
                elif "line chart" in text_lower: chart_type = "line"
                elif "area chart" in text_lower: chart_type = "area"
                elif "scatter plot" in text_lower or "scatter chart" in text_lower: chart_type = "scatter"
                elif "pie chart" in text_lower: chart_type = "pie"

                if chart_type:
                    logger.info(f"Detected chart type suggestion: {chart_type}")
                    df_display_chart = df_for_chart # Start with the latest df
                    try:
                        # Prepare DataFrame index for non-pie charts if suitable
                        if chart_type != "pie" and df_display_chart.index.name is None and len(df_display_chart.columns) > 1:
                            potential_index_col = df_display_chart.columns[0]
                            col_dtype = df_display_chart[potential_index_col].dtype
                            if pd.api.types.is_string_dtype(col_dtype) or \
                               pd.api.types.is_categorical_dtype(col_dtype) or \
                               pd.api.types.is_datetime64_any_dtype(col_dtype):
                                logger.info(f"Attempting to automatically set DataFrame index to '{potential_index_col}' for charting.")
                                df_display_chart = df_display_chart.copy().set_index(potential_index_col)
                                logger.info("Index set successfully for chart display.")
                            else:
                                logger.info(f"First column '{potential_index_col}' (type: {col_dtype}) not suitable for index, using original DataFrame for chart.")
                        else:
                             logger.info("Using original DataFrame for chart (index exists, single col, or pie).")

                        # Add chart info to the message dictionary
                        base_assistant_message["streamlit_chart"] = {
                            "type": chart_type,
                            "data": df_display_chart # Use the potentially indexed dataframe
                        }
                        logger.info(f"Added streamlit_chart block (type: {chart_type}) to assistant message.")
                    except Exception as chart_prep_e:
                        logger.warning(f"Could not prepare DataFrame for charting: {chart_prep_e}. Chart might not display correctly.", exc_info=True)
                        # Still add chart block but use original df and add warning
                        base_assistant_message["streamlit_chart"] = {
                            "type": chart_type,
                            "data": df_for_chart # Fallback df
                        }
                        base_assistant_message["content"] += f"\n\n**Note:** Could not automatically prepare data for {chart_type} chart: `{chart_prep_e}`"
                else:
                    logger.info("No specific chart type suggestion detected in AI response text.")
            else: # DataFrame is empty or None
                 logger.info("DataFrame is empty or None, skipping chart generation check.")


            final_assistant_message_dict = base_assistant_message
            logger.info("Query Agent response processing complete (post-confirmation).")

        else: # Agent result was not the expected QueryResponse structure
            error_msg = f"Received an unexpected response structure from main Query Agent. Type: {type(agent_run_result)}"
            logger.error(f"{error_msg}. Content: {agent_run_result}")
            return {"role": "assistant", "content": f"Sorry, internal issue processing the {target_db_key} database query results. Unexpected AI response format."}

        # Save the final response to LangChain memory before returning
        if final_assistant_message_dict and 'content' in final_assistant_message_dict:
            logger.info("Saving final query response to LangChain memory")
            if 'lc_memory' in st.session_state:
                try:
                    # Only save the user message that triggered this and the final text content
                    # Avoid saving complex objects or intermediate steps in lc_memory
                    # --- NEW: Append DataFrame summary if available ---
                    output_text = final_assistant_message_dict['content']
                    # Try to find the most recent DataFrame (sql_results_df or last_chartable_data)
                    df_to_summarize = None
                    if 'last_chartable_data' in st.session_state and st.session_state.last_chartable_data is not None:
                        df_to_summarize = st.session_state.last_chartable_data
                    elif 'sql_results_df' in locals() and sql_results_df is not None and not sql_results_df.empty:
                        df_to_summarize = sql_results_df
                    if df_to_summarize is not None and not df_to_summarize.empty:
                        df_summary = summarize_dataframe(df_to_summarize)
                        output_text += f"\n\n[DataFrame Summary]\n{df_summary}"
                    st.session_state.lc_memory.save_context(
                        {"input": user_message}, # The original user request
                        {"output": output_text} # The final text part of the answer + DataFrame summary if present
                    )
                    logger.info("Successfully saved final response context to LangChain memory (with DataFrame summary if present)")
                except Exception as mem_err:
                    logger.error(f"Error saving to LangChain memory: {str(mem_err)}", exc_info=True)

        # --- NEW: If combined SQL + visualization intent, trigger visualization immediately ---
        if st.session_state.get('combined_sql_visualization'):
            chart_type = st.session_state.combined_sql_visualization.get('chart_type', 'bar')
            user_message = st.session_state.combined_sql_visualization.get('user_message', '')
            # Use the most recent DataFrame from SQL result
            if 'last_chartable_data' in st.session_state and st.session_state.last_chartable_data is not None:
                df = st.session_state.last_chartable_data
                db_key = st.session_state.get('last_chartable_db_key', 'unknown')
                # Generate DataFrame summary
                df_summary = summarize_dataframe(df)
                visualization_prompt = f"Create a {chart_type} chart from the most recent query results.\nData summary:\n{df_summary}\nUser request: {user_message}\nDatabase: {db_key}"
                class SimpleContext:
                    def __init__(self, model, deps=None):
                        self.model = model
                        self.deps = deps
                class SimpleDeps:
                    def __init__(self, db_key):
                        self.db_key = db_key
                local_llm = GeminiModel(model_name=st.secrets.get("GEMINI_MAIN_MODEL", "gemini-1.5-flash"))
                simple_deps = SimpleDeps(db_key)
                ctx = SimpleContext(local_llm, simple_deps)
                async def run_python_agent():
                    return await call_python_agent(ctx, visualization_prompt)
                viz_result = run_async_task(run_python_agent)
                # Add the chart to the chat history
                if hasattr(viz_result, 'data') and hasattr(viz_result.output, 'visualization') and viz_result.output.visualization:
                    vis = viz_result.output.visualization
                    if vis.success:
                        response = f"Here's the {chart_type} chart of your data:"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "streamlit_chart": {
                                "type": chart_type,
                                "data": df
                            }
                        })
                    else:
                        error_msg = vis.error or "Unknown error"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"I couldn't create the {chart_type} chart. Error: {error_msg}"
                        })
                else:
                    content = "I've created a chart based on your data."
                    if hasattr(viz_result, 'data') and hasattr(viz_result.output, 'explanation'):
                        content = viz_result.output.explanation or content
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": content,
                        "streamlit_chart": {
                            "type": chart_type,
                            "data": df
                        }
                    })
                # Clear the combined intent flag
                del st.session_state['combined_sql_visualization']

    except Exception as outer_agent_e:
        # Catch errors in the outer try block (e.g., connection, schema formatting, agent instantiation)
        error_msg = f"An error occurred during main query agent processing: {str(outer_agent_e)}"
        logger.exception("Error during query agent execution or response processing (post-confirmation):")

        # Save error message context to LangChain memory
        if 'lc_memory' in st.session_state:
            try:
                error_response_content = f"Sorry, I encountered an error generating the response for the {target_db_key} database: {str(outer_agent_e)}"
                st.session_state.lc_memory.save_context({"input": user_message}, {"output": error_response_content})
                logger.info("Saved error context to LangChain memory")
            except Exception as mem_save_err:
                logger.error(f"Failed to save error message context to LangChain memory: {mem_save_err}")

        # Return the error message for display
        return {"role": "assistant", "content": f"Sorry, I encountered an error generating the response for the {target_db_key} database: {str(outer_agent_e)}"}
    finally:
        # --- Cleanup Database Connection ---
        if deps:
            logger.info("Cleaning up database connection from run_agents_post_confirmation_inner.")
            # Use run_async_task for the async cleanup method
            try:
                async def run_cleanup():
                    await deps.cleanup()
                
                run_async_task(run_cleanup)
                logger.info("Database connection cleanup completed successfully.")
            except Exception as cleanup_e:
                logger.error(f"Error during database connection cleanup: {str(cleanup_e)}", exc_info=True)
        logger.info(f"run_agents_post_confirmation_inner finished. Total duration: {time.time() - start_inner_time:.2f}s")

    # Return the final message dict if successful, otherwise an error dict should have been returned earlier
    if final_assistant_message_dict:
        return final_assistant_message_dict
    else:
        # This case should ideally not be reached if errors are handled above
        logger.error("run_agents_post_confirmation_inner finished unexpectedly without a result dictionary or error string being returned.")
        return {"role": "assistant", "content": "Sorry, an unexpected internal error occurred during processing."}


async def continue_after_table_confirmation():
    """
    Coordinates the logic flow *after* the user confirms table selection.
    Calls the inner async function to run the main query agent steps.
    Handles appending the result/error message to the chat history.
    This function itself is async and should be awaited or run via run_async.
    """
    start_time = time.time()
    logger.info("continue_after_table_confirmation called.")

    # --- Retrieve necessary state ---
    if st.session_state.get("confirmed_tables") is None:
        logger.error("continue_after_table_confirmation called without confirmed_tables in session state.")
        st.error("Internal error: No confirmed tables found to continue processing.")
        clear_pending_state() # Clean up
        return # Exit early

    db_metadata = st.session_state.get("pending_db_metadata")
    target_db_key = st.session_state.get("pending_target_db_key")
    target_db_path = st.session_state.get("pending_target_db_path")
    message = st.session_state.get("pending_user_message")
    selected_tables = st.session_state.get("confirmed_tables", [])
    # Retrieve agent history (though not currently used in agent call)
    agent_history = st.session_state.get("agent_message_history", [])

    # --- Validate state ---
    if not all([db_metadata, target_db_key, target_db_path, message, selected_tables is not None]):
        logger.error("Missing required data in session state for continue_after_table_confirmation.")
        st.error("Internal error: Missing context to continue processing your request.")
        clear_pending_state()
        return # Exit early

    logger.info(f"Continuing with DB: {target_db_key}, Path: {target_db_path}, Tables: {selected_tables}, Query: '{message[:50]}...'")
    assistant_chat_message = None # Initialize

    # --- Execute the core post-confirmation logic ---
    try:
        # No spinner here as this is already async
        # *** KEY CHANGE: Use await instead of run_async ***
        result_dict = await run_agents_post_confirmation_inner(
            db_metadata=db_metadata,
            selected_tables=selected_tables,
            target_db_key=target_db_key,
            target_db_path=target_db_path,
            user_message=message,
            agent_message_history=agent_history # Pass history for logging/potential future use
        )
        assistant_chat_message = result_dict
        # --- NEW: If combined SQL + visualization intent, trigger visualization immediately ---
        if st.session_state.get('combined_sql_visualization'):
            chart_type = st.session_state.combined_sql_visualization.get('chart_type', 'bar')
            user_message = st.session_state.combined_sql_visualization.get('user_message', '')
            # Use the most recent DataFrame from SQL result
            if 'last_chartable_data' in st.session_state and st.session_state.last_chartable_data is not None:
                df = st.session_state.last_chartable_data
                db_key = st.session_state.get('last_chartable_db_key', 'unknown')
                # Generate DataFrame summary
                df_summary = summarize_dataframe(df)
                visualization_prompt = f"Create a {chart_type} chart from the most recent query results.\nData summary:\n{df_summary}\nUser request: {user_message}\nDatabase: {db_key}"
                class SimpleContext:
                    def __init__(self, model, deps=None):
                        self.model = model
                        self.deps = deps
                class SimpleDeps:
                    def __init__(self, db_key):
                        self.db_key = db_key
                local_llm = GeminiModel(model_name=st.secrets.get("GEMINI_MAIN_MODEL", "gemini-1.5-flash"))
                simple_deps = SimpleDeps(db_key)
                ctx = SimpleContext(local_llm, simple_deps)
                async def run_python_agent():
                    return await call_python_agent(ctx, visualization_prompt)
                viz_result = run_async_task(run_python_agent)
                # Add the chart to the chat history
                if hasattr(viz_result, 'data') and hasattr(viz_result.output, 'visualization') and viz_result.output.visualization:
                    vis = viz_result.output.visualization
                    if vis.success:
                        response = f"Here's the {chart_type} chart of your data:"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "streamlit_chart": {
                                "type": chart_type,
                                "data": df
                            }
                        })
                    else:
                        error_msg = vis.error or "Unknown error"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"I couldn't create the {chart_type} chart. Error: {error_msg}"
                        })
                else:
                    content = "I've created a chart based on your data."
                    if hasattr(viz_result, 'data') and hasattr(viz_result.output, 'explanation'):
                        content = viz_result.output.explanation or content
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": content,
                        "streamlit_chart": {
                            "type": chart_type,
                            "data": df
                        }
                    })
                # Clear the combined intent flag
                del st.session_state['combined_sql_visualization']
    except Exception as e:
        # Catch errors from awaiting run_agents_post_confirmation_inner itself
        error_msg = f"A critical error occurred running the post-confirmation agents: {str(e)}"
        logger.exception("Critical error in continue_after_table_confirmation while awaiting inner function:")
        # Ensure an error message is created
        assistant_chat_message = {"role": "assistant", "content": f"Sorry, a critical error occurred: {str(e)}"}

    # --- Append result/error to chat history and clean up ---
    if assistant_chat_message:
        st.session_state.chat_history.append(assistant_chat_message)
        logger.info(f"Assistant message appended to history in continue_after_table_confirmation. Content: {str(assistant_chat_message.get('content'))[:100]}...")
    else:
        # Fallback if something went very wrong and no message was generated
        logger.error("continue_after_table_confirmation finished without an assistant message object to append.")
        # Avoid adding duplicate errors if one was already added
        if not st.session_state.chat_history or st.session_state.chat_history[-1].get("role") != "assistant":
             st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, an internal error occurred, and no response could be generated."})

    # Always clear the pending state after attempting continuation
    clear_pending_state()
    logger.info(f"continue_after_table_confirmation finished. Duration: {time.time() - start_time:.2f}s")
    # No return needed, side effects (chat history, state clear) are done


def clear_pending_state():
    """Clears session state variables related to pending table confirmation."""
    keys_to_clear = [
        "table_confirmation_pending", "pending_db_metadata", "pending_target_db_key",
        "pending_target_db_path", "pending_user_message", "confirmed_tables",
        "candidate_tables", "all_tables", "table_agent_reasoning",
        # New DB selection keys
        "db_selection_pending", "db_selection_reason", "pending_db_keys", "confirmed_db_key"
    ]
    cleared_count = 0
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            cleared_count += 1
    if cleared_count > 0:
        logger.info(f"Cleaned up {cleared_count} pending session state keys.")


# --- Async Runner Helper ---
def run_async_task(async_func, *args):
    """
    Run an asynchronous function in a new event loop.
    Will retry on event loop errors by creating a fresh loop.

    Args:
    async_func (coroutine): The asynchronous function to execute.
    *args: Arguments to pass to the asynchronous function.

    Returns:
    The result of the asynchronous function
    """
    
    loop = None
    result = None

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply(loop)
        result = loop.run_until_complete(async_func(*args))
    except Exception as e:
        logger.warning(f"Error in first attempt to run {getattr(async_func, '__name__', str(async_func))}: {str(e)}")
        # Close the existing loop if open
        if loop is not None:
            loop.close()

        # Create a new loop for retry
        logger.info(f"Retrying with new event loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply(loop)
        
        try:
            result = loop.run_until_complete(async_func(*args))
        except Exception as retry_e:
            logger.error(f"Error in retry attempt: {str(retry_e)}", exc_info=True)
            raise retry_e
    finally:
        if loop is not None:
            loop.close()
    
    return result

# *** REFACTORED run_async ***
def run_async(coro):
    """
    Runs an async coroutine reliably from a synchronous context,
    leveraging the robust run_async_task implementation.
    """
    coro_name = getattr(coro, '__name__', str(coro))
    logger.debug(f"Running coroutine '{coro_name}' using run_async_task...")
    
    # Use the more robust implementation that handles event loop errors
    try:
        return run_async_task(lambda: coro)
    except TypeError as e:
        # Catch the specific error from the logs if a non-awaitable is passed
        if "An asyncio.Future, a coroutine or an awaitable is required" in str(e):
            logger.error(f"TypeError in run_async: The object passed was not a coroutine or awaitable. Object: {coro}", exc_info=True)
            raise TypeError(f"run_async requires a coroutine or awaitable, but received {type(coro)}. Ensure you are passing an async function call.") from e
        else:
            # Re-raise other TypeErrors
            logger.exception(f"TypeError caught by run_async while running {coro_name}: {e}")
            raise e
    except Exception as e:
        # Log any other exceptions originating from the coroutine or loop management
        logger.exception(f"Exception caught by run_async while running {coro_name}: {e}")
        # Re-raise the exception so the caller (e.g., in main) can handle it
        raise e


# --- Helper Functions --- (Keep as is)
def get_base64_encoded_image(image_path):
    """Get base64 encoded image"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logger.warning(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}", exc_info=True)
        return None

# --- Message History Validation Helper ---
def validate_message_entry(message: Any) -> Optional[Dict[str, str]]:
    """
    Validates and sanitizes a message entry for agent message history.

    Args:
        message: Any object to validate as a message

    Returns:
        Dict with 'role' and 'content' keys if valid, None otherwise
    """
    if not isinstance(message, dict):
        return None

    if 'role' not in message or 'content' not in message:
        return None

    role = message['role']
    if role not in ['user', 'assistant', 'system']:
        return None

    # Ensure content is reasonably sized (optional, adjust limit as needed)
    content_str = str(message['content'])
    # if len(content_str) > 10000: # Example limit
    #     logger.warning(f"Truncating oversized message content for agent history (Role: {role})")
    #     content_str = content_str[:10000] + "... (truncated)"

    return {
        'role': role,
        'content': content_str
    }

def filter_message_history(messages: List[Any]) -> List[Dict[str, str]]:
    """
    Filters a list of message entries, keeping only valid entries.

    Args:
        messages: List of message objects to filter

    Returns:
        List of valid message dictionaries
    """
    if not messages:
        return []

    filtered = []
    for msg in messages:
        valid_msg = validate_message_entry(msg)
        if valid_msg:
            filtered.append(valid_msg)

    return filtered

# This transform function might not be needed if we avoid passing history to Gemini
# Kept for reference, but likely unused now.
def transform_message_history(messages: List[Dict]) -> Optional[List[Dict]]:
    """
    Transforms message history into a format potentially expected by PydanticAI agents.
    (Currently less relevant as history is skipped for Gemini).

    Returns None if the input list is empty or None.
    """
    if not messages:
        return None

    # For PydanticAI, we need to make sure each message has the expected structure
    # Only keep user and assistant messages
    transformed = []
    for msg in messages:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            if msg['role'] == 'user':
                transformed.append({"role": "user", "content": str(msg['content'])})
            elif msg['role'] == 'assistant':
                transformed.append({"role": "assistant", "content": str(msg['content'])})
            # Skip system messages as they can cause issues with some models

    return transformed if transformed else None


# --- DataFrame Summary Helper ---
def summarize_dataframe(df, max_examples=2):
    summary = [f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns."]
    for col in df.columns:
        dtype = str(df[col].dtype)
        examples = df[col].dropna().unique()[:max_examples]
        example_str = ', '.join(map(str, examples))
        summary.append(f"- {col} ({dtype}): e.g., {example_str}")
    return "\n".join(summary)


# --- Main Streamlit App Function ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="centered", page_title="SmartQuery")

    # Initialize session state variables safely
    default_session_state = {
        'chat_history': [],            # For display
        'last_result': None,           # For debug
        'last_db_key': None,           # For context reuse
        'agent_message_history': [],   # Cumulative history for AI context (logging/future)
        'last_chartable_data': None,   # For follow-up charting
        'last_chartable_db_key': None, # For follow-up charting context
        # Table Confirmation State
        'table_confirmation_pending': False,
        'candidate_tables': [],
        'all_tables': [],
        'table_agent_reasoning': "",
        'confirmed_tables': None,      # Stores the user-confirmed list
        # Common Pending State (used by both flows)
        'pending_user_message': None,
        'pending_db_metadata': None,
        'pending_target_db_key': None, # Target key before confirmation
        'pending_target_db_path': None,# Target path before confirmation
        # DB Selection State (NEW)
        'db_selection_pending': False,
        'db_selection_reason': "",
        'pending_db_keys': [],
        'confirmed_db_key': None       # Stores the user-confirmed DB key
    }
    for key, default_value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            # logger.debug(f"Initialized '{key}' in session state.") # Debug level

    # --- LangChain Memory Initialization ---
    if 'lc_memory' not in st.session_state:
        st.session_state.lc_memory = ConversationBufferMemory(return_messages=True)
        logger.info("Initialized LangChain ConversationBufferMemory.")

    # --- Main Page Content ---
    # ... (HTML/CSS for title, features, examples, clear button remain the same) ...
    st.markdown('<h1 style="text-align: center;"><span style="color: #00ade4;">SmartQuery</span></h1>', unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #555;'>AI-Powered Database Analysis with Google Gemini</h5>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        /* Main chat container adjustments */
        .main-chat-container {
            flex-grow: 1; /* Allow chat to take available space */
            display: flex;
            flex-direction: column;
            /* Removed fixed height - let it grow naturally or be constrained by parent */
            /* height: calc(20vh - 250px); */
            overflow: hidden; /* Hide main container overflow */
            margin-top: 0.5rem; /* Reduce top margin from 1rem to 0.5rem */
        }
        /* Make message container scrollable */
        .chat-messages-container {
            flex-grow: 1;
            overflow-y: auto; /* Enable vertical scrolling */
            padding: 0 1rem 0.5rem 1rem; /* Reduce bottom padding from 1rem to 0.5rem */
            /* Removed margin-bottom, rely on input container's presence */
            /* margin-bottom: 60px; */
            /* Add max-height to prevent excessive growth if needed */
            max-height: 65vh; /* Adjust as needed */
        }
        /* Sticky input - default Streamlit behavior is usually good */
        /* .stChatInputContainer */

        /* Feature boxes */
        .features-container { display: flex; flex-direction: column; gap: 0.75rem; margin: 0.5rem auto; max-width: 1000px; } /* Reduced margin from 1rem to 0.5rem */
        .features-row { display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; /* Allow wrapping on smaller screens */ }
        .feature-text { flex: 1 1 300px; /* Flex grow, shrink, basis */ max-width: 450px; padding: 1rem; background: #f0f8ff; border: 1px solid #e0e0e0; border-radius: 8px; font-size: .9rem; line-height: 1.4; display: flex; align-items: flex-start; gap: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .check-icon { width: 18px; height: 18px; object-fit: contain; margin-top: 0.15rem; flex-shrink: 0; }

        /* Example queries */
        .example-queries { margin: 1rem 0 0.75rem 0; font-size: 1rem; border-left: 3px solid #00ade4; padding-left: 1rem; } /* Reduced top margin from 1.5rem to 1rem, bottom from 1rem to 0.75rem */
        .example-queries p { margin-bottom: 0.5rem; font-weight: bold; color: #002345; }
        .example-queries ul { margin: 0; padding-left: 1.2rem; list-style-type: '→ '; }
        .example-queries li { margin-bottom: 0.3rem; color: #333; font-size: 0.9em; }

        /* DataFrame display */
        .stDataFrame { width: 100%; font-size: 0.9em; }

        /* Improve chat message appearance */
        .stChatMessage { border-radius: 10px; border: 1px solid #eee; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-top: 0.25rem; margin-bottom: 0.5rem; } /* Added smaller top margin */

        /* Adjust code block styling */
        .stCodeBlock { font-size: 0.85em; }

    </style>
    """, unsafe_allow_html=True)

    # --- Feature Highlights ---
    try:
        check_path = Path(__file__).parent / "assets" / "correct.png"
        check_base64 = get_base64_encoded_image(check_path)
        check_img = f'<img src="data:image/png;base64,{check_base64}" class="check-icon" alt="✓">' if check_base64 else "✓"
    except Exception as img_e:
        logger.warning(f"Could not load check icon: {img_e}")
        check_img = "✓" # Fallback
    st.markdown(f"""
    <div class="features-container">
        <div class="features-row">
            <div class="feature-text">{check_img} Ask natural language questions about World Bank Group data.</div>
            <div class="feature-text">{check_img} Get instant SQL-powered insights from both databases.</div>
        </div>
        <div class="features-row">
            <div class="feature-text">{check_img} Generate visualizations (bar, line, pie charts) via Python.</div>
            <div class="feature-text">{check_img} System automatically identifies the right database for your query.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info('Query data from the World Bank Group datasets available at https://financesone.worldbank.org/. The AI will identify the appropriate database automatically.', icon="ℹ️") # Use streamlit icon
    # --- Example Queries ---
    st.markdown("""
    <div class="example-queries">
        <p>Example Questions:</p>
        <ul>
            <li>"Show me the status of all Loans Disbursed to Ukraine."</li>
            <li>"Show me the sum of IBRD loans to India approved since 2020 per year"</li>
            <li>"Compare the average IFC investment size for 'Loan' products between Nepal and Bhutan."</li>
            <li>"What is the total gross guarantee exposure for MIGA in the Tourism sector in Senegal?"</li>
            <li>"Which countries have the highest total MIGA guarantee exposure? Create a bar chart."</li>
            <li>"Give me the top 10 IFC equity investments from China"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Clear Chat Button
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.session_state.last_db_key = None
        st.session_state.agent_message_history = []
        st.session_state.last_chartable_data = None
        st.session_state.last_chartable_db_key = None
        # Clear LangChain memory as well
        if 'lc_memory' in st.session_state:
             st.session_state.lc_memory.clear()
        clear_pending_state() # Clear custom pending states
        logger.info("Chat history, LangChain memory, and associated states cleared by user.")
        st.rerun()

    # --- ASYNC CONTINUATION HANDLING --- #
    # This section handles logic that needs to run *after* a confirmation action
    # from a previous Streamlit rerun has set the corresponding state.

    # 1. Handle Database Selection Continuation
    # Check if DB was confirmed in the previous run AND we are not currently waiting for DB selection again
    if st.session_state.get("confirmed_db_key") is not None and not st.session_state.get("db_selection_pending", False):
        logger.info("Detected confirmed DB key from previous run, proceeding with table selection stage.")

        # Define the async helper function locally
        async def continue_after_db_selection():
            """Runs the table selection stage after DB is confirmed."""
            start_time_db_cont = time.time()
            logger.info("continue_after_db_selection called.")
            confirmed_key = st.session_state.get("confirmed_db_key")
            user_message = st.session_state.get("pending_user_message")
            db_metadata = st.session_state.get("pending_db_metadata")

            if not all([confirmed_key, user_message, db_metadata]):
                logger.error("Missing required data in session state for continue_after_db_selection.")
                st.error("Internal error: Missing context to continue processing after database selection.")
                clear_pending_state() # Clear all state on error
                return

            logger.info(f"Continuing with confirmed DB: {confirmed_key}, Query: '{user_message[:50]}...'" )
            try:
                # Run the next stage (table selection) directly using await
                await run_table_selection_stage(user_message, confirmed_key, db_metadata)
                logger.info("run_table_selection_stage completed within continue_after_db_selection.")
                # Clear ONLY the DB selection part of the state now that it's processed
                # Keep pending_user_message, pending_db_metadata for the next stage
                db_keys_to_clear = ['db_selection_pending', 'db_selection_reason', 'pending_db_keys', 'confirmed_db_key']
                cleared_count = 0
                for key in db_keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                        cleared_count +=1
                logger.info(f"Cleared {cleared_count} DB selection state keys.")
                # State for table selection should now be set by run_table_selection_stage
            except Exception as e:
                 error_msg = f"A critical error occurred running the table selection stage after DB confirmation: {str(e)}"
                 logger.exception(error_msg)
                 # Add error message to chat
                 st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, a critical error occurred after DB confirmation: {str(e)}"})
                 clear_pending_state() # Clear all state on critical error
            finally:
                logger.info(f"continue_after_db_selection finished. Duration: {time.time() - start_time_db_cont:.2f}s")

        # Run the async helper from the synchronous main flow
        try:
            with st.spinner("Processing confirmed database selection..."):
                logger.info("Calling run_async_task for continue_after_db_selection...")
                # Wrap in an async function for run_async_task
                async def db_wrapper_func():
                    return await continue_after_db_selection()
                
                run_async_task(db_wrapper_func)
                logger.info("run_async_task for continue_after_db_selection finished.")
                # Rerun AFTER the async operation completes to update the UI (likely showing table selection now)
                logger.info("Rerunning Streamlit after successful DB confirmation processing.")
                st.rerun()
        except Exception as e:
             # Catch errors during the run_async call or the helper itself
             st.error(f"An error occurred while processing your confirmed database selection: {str(e)}")
             logger.error(f"Error occurred during post-DB-confirmation processing or rerun: {e}", exc_info=True)
             clear_pending_state() # Clean up state on error
             st.rerun() # Rerun to show the error


    # 2. Handle Table Selection Continuation
    # Check if tables were confirmed in the previous run
    if st.session_state.get("confirmed_tables") is not None:
        logger.info("Detected confirmed tables from previous run, proceeding with post-confirmation processing.")
        try:
            logger.info("Calling run_async_task for continue_after_table_confirmation...  ")
            # Wrap in an async function since run_async_task expects an async function
            async def wrapper_func():
                return await continue_after_table_confirmation()
            
            run_async_task(wrapper_func)
            logger.info("run_async_task for continue_after_table_confirmation finished.")
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
            logger.error(f"Error during post-table-confirmation processing: {e}", exc_info=True)
            clear_pending_state() # Clean up regardless of error
            # Rerun to update UI after error handling
            logger.info("Rerunning after error in table confirmation continuation.")
            st.rerun()

    # --- Chat Interface --- #
    # This part displays the chat history and the confirmation UIs if pending

    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages-container" id="chat-messages-container-id">', unsafe_allow_html=True)

    chat_display_container = st.container()
    with chat_display_container:
        # Display existing chat messages
        for i, message in enumerate(st.session_state.chat_history):
            role = message.get("role", "assistant") # Default to assistant if role missing
            content = message.get("content", "")
            with st.chat_message(role):
                st.markdown(content) # Display main text content

                # Display SQL details if present
                sql_result = message.get("sql_result")
                if sql_result and isinstance(sql_result, dict):
                    if sql_result.get("query"):
                        st.markdown("**SQL Query:**")
                        st.code(sql_result["query"], language="sql")
                    if sql_result.get("explanation"):
                        st.markdown(f"**Explanation:** {sql_result['explanation']}")
                    if sql_result.get("error"):
                        st.error(f"SQL Error: {sql_result['error']}")
                    elif "results" in sql_result and isinstance(sql_result["results"], list) and sql_result["results"]:
                        # Only display results table if query didn't error and results exist
                        st.markdown("**Results:**")
                        try:
                            results_df = pd.DataFrame(sql_result["results"])
                            st.dataframe(results_df)
                        except Exception as e:
                            st.error(f"Error displaying results table: {str(e)}")

                # Display Python details if present
                python_result = message.get("python_result")
                if python_result and isinstance(python_result, dict):
                    if python_result.get("code"):
                         st.markdown("**Python Code:**")
                         st.code(python_result["code"], language="python")
                    if python_result.get("explanation"):
                        st.markdown(f"**Explanation:** {python_result['explanation']}")
                    if python_result.get("error"):
                        st.error(f"Error executing Python code: {python_result['error']}")
                    elif python_result.get("warning"):
                         st.warning(f"Python Code Warning: {python_result['warning']}")

                # Display Streamlit chart if present (legacy format)
                if "streamlit_chart" in message and isinstance(message["streamlit_chart"], dict):
                    st.markdown("**Visualization:**")
                    try:
                        chart_type = message["streamlit_chart"].get("type")
                        df = message["streamlit_chart"].get("data")
                        # Ensure df is a pandas DataFrame
                        if isinstance(df, pd.DataFrame):
                            if not df.empty:
                                if chart_type == "bar":
                                    st.bar_chart(df)
                                elif chart_type == "line":
                                    st.line_chart(df)
                                elif chart_type == "area":
                                    st.area_chart(df)
                                elif chart_type == "scatter":
                                    # Basic scatter needs x and y
                                    if len(df.columns) >= 2:
                                        # Use column names directly if available and sensible
                                        st.scatter_chart(df) # Let streamlit infer columns
                                    else:
                                        st.warning("Scatter plot requires at least two data columns.")
                                elif chart_type == "pie":
                                    # Plotly pie chart logic
                                    if len(df.columns) > 0:
                                        values_col = df.columns[0]
                                        names_col = df.index.name # Prefer index name for labels
                                        if not names_col and len(df.columns) > 1:
                                            # Fallback to using first column as names if index has no name
                                            # and there's a second column for values
                                            names_col = df.columns[0]
                                            values_col = df.columns[1]
                                            st.warning(f"Used column '{names_col}' for labels and '{values_col}' for values in pie chart.")
                                        elif not names_col and len(df.columns) == 1:
                                             st.warning("Cannot generate pie chart: Need an index name or at least two columns (one for names, one for values).")
                                             fig = None
                                        else: # Use index name and first column
                                             fig = px.pie(df, names=df.index, values=values_col, title="Pie Chart")

                                        if names_col: # Check if we determined a names column/index
                                             fig = px.pie(df, names=names_col if names_col != df.index.name else df.index, values=values_col, title="Pie Chart")
                                             st.plotly_chart(fig, use_container_width=True)
                                        # else: error message was already shown above
                                    else:
                                         st.warning("Cannot generate pie chart: Data is empty or missing columns.")
                            else:
                                st.warning(f"Cannot generate {chart_type} chart: Received empty data.")
                        else:
                             st.warning(f"Cannot generate chart: Invalid data format received (expected DataFrame). Type: {type(df)}")
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        logger.exception(f"Error rendering chart: {e}")

                # NEW: Display charts using the "chart" key format from handle_follow_up_chart
                if "chart" in message and isinstance(message["chart"], dict):
                    st.markdown("**Visualization:**")
                    try:
                        chart_data = message["chart"]
                        chart_type = chart_data.get("type", "bar")
                        df = chart_data.get("data")
                        
                        # Check if we have valid data
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            if chart_type == "bar":
                                st.bar_chart(df)
                            elif chart_type == "line":
                                st.line_chart(df)
                            elif chart_type == "area":
                                st.area_chart(df)
                            elif chart_type == "pie":
                                # For pie charts we need a bit more setup
                                try:
                                    if len(df.columns) >= 2:
                                        # Use plotly for pie charts 
                                        import plotly.express as px
                                        fig = px.pie(df, 
                                                     names=df.iloc[:, 0], 
                                                     values=df.iloc[:, 1], 
                                                     title=f"Pie Chart - {chart_data.get('db_key', 'Database')}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Pie chart requires at least 2 columns")
                                except Exception as e:
                                    st.error(f"Error displaying pie chart: {str(e)}")
                            else:
                                # Default to showing the dataframe
                                st.dataframe(df)
                        else:
                            st.warning("Chart data is empty or not available")
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
                        logger.exception(f"Error rendering chart: {e}")

                # Display Python details if present
                python_result = message.get("python_result")
                if python_result and isinstance(python_result, dict):
                    if python_result.get("code"):
                         st.markdown("**Python Code:**")
                         st.code(python_result["code"], language="python")
                    if python_result.get("explanation"):
                        st.markdown(f"**Explanation:** {python_result['explanation']}")
                    if python_result.get("error"):
                        st.error(f"Error executing Python code: {python_result['error']}")
                    elif python_result.get("warning"):
                         st.warning(f"Python Code Warning: {python_result['warning']}")


        # --- Database Selection UI ---
        # Show this UI ONLY if db_selection is pending
        if st.session_state.get("db_selection_pending", False):
            st.markdown('<div id="db-selection-anchor"></div>', unsafe_allow_html=True)
            reason = st.session_state.get("db_selection_reason", "Please select the database.")
            options = st.session_state.get("pending_db_keys", [])
            with st.chat_message("assistant"):
                st.info(f"**Database Selection Required:** {reason}", icon="ℹ️")
                if options:
                    # Use index=None for no default selection
                    chosen_db = st.selectbox(
                        "Select the target database:",
                        options=options,
                        index=None, # Ensure no default selection
                        placeholder="Choose an option", # Add placeholder
                        key="db_confirm_selectbox" # Unique key
                    )
                    if st.button("Confirm Database Selection"):
                        if chosen_db:
                            logger.info(f"User confirmed database selection: {chosen_db}")
                            # Set the confirmed key, clear the pending flag
                            st.session_state.confirmed_db_key = chosen_db
                            st.session_state.db_selection_pending = False
                            # Keep other pending state (message, metadata) for the next step
                            # Rerun immediately to trigger the continuation logic above
                            st.rerun()
                        else:
                            st.warning("Please select a database.")
                else:
                    st.error("Internal Error: No database options available for selection.")
            # Inject JS to scroll to the anchor (keep using timestamp for uniqueness)
            scroll_script = f"""
              <script>
                  // Unique ID based on timestamp: {time.time()}
                  function scrollToDbSelection() {{
                      var element = window.parent.document.getElementById('db-selection-anchor');
                      if (element) {{
                          console.log('Scrolling to db selection anchor');
                          element.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                      }} else {{
                          console.log('db selection anchor not found, retrying...');
                          setTimeout(scrollToDbSelection, 100); // Retry if not found yet
                      }}
                  }}
                  // Ensure running after element might be rendered
                  window.addEventListener('load', scrollToDbSelection);
                  // Fallback execution
                  setTimeout(scrollToDbSelection, 200);
              </script>
            """
            components.html(scroll_script, height=0, width=0)


        # --- Table Selection UI ---
        # Show this UI ONLY if table confirmation is pending (and DB selection is NOT pending)
        elif st.session_state.get("table_confirmation_pending", False):
            st.markdown('<div id="table-selection-anchor"></div>', unsafe_allow_html=True)
            candidate_tables = st.session_state.get("candidate_tables", [])
            all_tables = st.session_state.get("all_tables", [])
            reasoning = st.session_state.get("table_agent_reasoning", "")
            db_key = st.session_state.get("pending_target_db_key", "") # Get the key determined earlier

            with st.chat_message("assistant"):
                # Adjust message based on whether candidates exist
                if not candidate_tables and not reasoning:
                     st.info(f"**Table Selection Required:** Please select the necessary tables from the {db_key} database for your query.", icon="ℹ️")
                elif not candidate_tables and reasoning:
                     st.info(f"**Table Selection Required:** {reasoning}", icon="ℹ️")
                else: # Candidates exist
                    st.info(f"**Table Selection Suggestion:** For the {db_key} database, I suggest using these tables: `{', '.join(candidate_tables)}`", icon="ℹ️")
                    if reasoning:
                        st.caption(f"Reasoning: {reasoning}")

                # Multiselect uses candidate_tables as default
                selected = st.multiselect(
                    f"Confirm or adjust the tables needed for your query:",
                    options=all_tables,
                    default=candidate_tables,
                    key="table_confirm_multiselect" # Unique key
                )

                if st.button("Confirm Table Selection"):
                    # Allow confirming even if selection is empty (user might override)
                    logger.info(f"User confirmed table selection: {selected}")
                    # Set the confirmed tables, clear the pending flag
                    st.session_state.confirmed_tables = selected
                    st.session_state.table_confirmation_pending = False
                    # Keep other pending state (message, metadata, db_key, db_path)
                    # Rerun immediately to trigger the continuation logic
                    st.rerun()

            # Inject JS to scroll to the table selection anchor
            scroll_script_table = f"""
              <script>
                  // Unique ID based on timestamp: {time.time()}
                  function scrollToTableSelection() {{
                      var element = window.parent.document.getElementById('table-selection-anchor');
                      if (element) {{
                          // Scroll to the anchor first
                          element.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                          // --- FIX: Also scroll to the bottom after a short delay to ensure UI is fully visible ---
                          setTimeout(function() {{
                              window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
                          }}, 400); // Delay to allow UI to render
                      }} else {{
                           console.log('table selection anchor not found, retrying...');
                          setTimeout(scrollToTableSelection, 100); // Retry if not found yet
                      }}
                  }}
                   // Ensure running after element might be rendered
                  window.addEventListener('load', scrollToTableSelection);
                  // Fallback execution
                  setTimeout(scrollToTableSelection, 200);
              </script>
            """
            components.html(scroll_script_table, height=0, width=0)

    st.markdown('</div>', unsafe_allow_html=True) # End chat-messages-container

    # --- Chat Input ---
    # Show chat input only if NO confirmation (neither DB nor Table) is pending
    if not st.session_state.get("db_selection_pending", False) and not st.session_state.get("table_confirmation_pending", False):
        user_input = st.chat_input("Ask about World Bank Group data...")
        if user_input:
            logger.info(f"User input received in main(): {user_input}")
            # Append user message to display history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            # Append validated user message to internal agent history (for logging/context)
            user_msg_validated = validate_message_entry({"role": "user", "content": user_input})
            if user_msg_validated:
                st.session_state.agent_message_history.append(user_msg_validated)
                logger.info("Added validated user message to agent_message_history")
            else:
                logger.warning("Failed to validate user message for agent_message_history, not added")

            # Get the async function (coroutine object)
            message_coro = handle_user_message(user_input)

            # Run the initial handling async function from the sync context
            try:
                with st.spinner("Analyzing request..."):
                    logger.info("Calling run_async_task for handle_user_message...")
                    
                    # Use run_async_task instead of directly awaiting to handle event loop issues
                    async def run_message_handler():
                        return await handle_user_message(user_input)
                    
                    run_async_task(run_message_handler)
                    
                    logger.info("run_async_task for handle_user_message finished.")
                # Rerun AFTER the async operation completes. This will either show
                # the greeting/error added by handle_user_message, or display
                # the DB/Table confirmation UI if state was set accordingly.
                logger.info("Rerunning Streamlit after processing user input / reaching confirmation point.")
                st.rerun()
            except Exception as e:
                 # Catch errors from run_async or handle_user_message itself
                 st.error(f"An error occurred while processing your request: {str(e)}")
                 logger.exception("Error occurred during initial user input processing or rerun.") # Use logger.exception
                 # Add error message to chat if not already there (avoid duplicates)
                 error_content = f"Sorry, a critical internal error occurred: {str(e)}"
                 # Check if the last message is already this error
                 if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content") != error_content:
                     st.session_state.chat_history.append({"role": "assistant", "content": error_content})
                 # Clear pending state if an error occurs during initial processing
                 clear_pending_state()
                 st.rerun() # Rerun to display the error

    st.markdown('</div>', unsafe_allow_html=True) # End main-chat-container

    # Scroll to bottom script (keep as is, use unique ID)
    # Make sure the ID matches the one used for the container
    scroll_script_bottom = f"""
        <script>
            // Unique ID based on timestamp: {time.time()}
            function scrollToBottom() {{
                const chatContainer = window.parent.document.getElementById('chat-messages-container-id');
                if (chatContainer) {{
                    // console.log('Scrolling chat container to bottom');
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }} else {{
                    // console.log('Chat container not found for scroll, retrying...');
                    // setTimeout(scrollToBottom, 150); // Optionally retry
                }}
            }}
            // Attempt scroll slightly after potential DOM updates
            // Using requestAnimationFrame can sometimes be smoother
            // window.requestAnimationFrame(scrollToBottom);
            // Fallback timeout
            setTimeout(scrollToBottom, 150);
        </script>
        """
    components.html(scroll_script_bottom, height=0, width=0)


class VisualizationResult(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate (bar, line, pie, etc.)")
    message: str = Field(..., description="Human-readable message about the visualization.")
    db_key: str = Field(..., description="Database key for context.")
    success: bool = Field(..., description="Whether the visualization was successfully prepared.")
    error: Optional[str] = Field(None, description="Error message if visualization failed.")

# PydanticAI-compatible visualization tool
async def visualize_last_dataframe(ctx: RunContext[AgentDependencies], chart_type: str, user_message: Optional[str] = None) -> VisualizationResult:
    """Creates a chart from the last query results."""
    logger.info(f"visualize_last_dataframe called with chart_type: {chart_type}")
    
    # Check if we have dataframe to visualize
    if 'last_dataframe' not in st.session_state or st.session_state.last_dataframe is None:
        logger.warning("No dataframe available to visualize")
        return VisualizationResult(
            chart_type=chart_type,
            message="No data available to visualize. Please run a query first.",
            db_key=st.session_state.get('last_db_key') or 'unknown',
            success=False,
            error="No data available to visualize"
        )
    
    try:
        import pandas as pd
        df = st.session_state.last_dataframe
        db_key = st.session_state.get('last_db_key') or 'unknown'
        
        # Store the chart data for rendering
        st.session_state.last_chart_data = {
            'type': chart_type,
            'data': df,
            'db_key': db_key
        }
        
        user_context = f"Chart request: {user_message}" if user_message else "No specific instructions"
        
        return VisualizationResult(
            chart_type=chart_type,
            message=f"Visualization prepared: {chart_type} chart for the previous data from the {db_key} database.\nUser context: {user_context}",
            db_key=db_key,
            success=True,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in visualize_last_dataframe: {str(e)}", exc_info=True)
        return VisualizationResult(
            chart_type=chart_type,
            message=f"Error creating visualization",
            db_key=st.session_state.get('last_db_key') or 'unknown',
            success=False,
            error=str(e)
        )


# 1. Python Agent Blueprint
class PythonAgentResult(BaseModel):
    python_code: Optional[str] = Field(None, description="Python code executed or generated.")
    explanation: Optional[str] = Field(None, description="Explanation of the code or result.")
    visualization: Optional[VisualizationResult] = None
    error: Optional[str] = None


def create_python_agent_blueprint():
    return {
        "result_type": PythonAgentResult,
        "name": "Python Data & Visualization Agent",
        "retries": 2,
        "system_prompt": '''You are a Python data analysis and visualization agent. Your job is to:
- Accept requests for data manipulation, analysis, or visualization using pandas DataFrames (named 'df') and numpy.
- If the user requests a chart (bar, line, pie, scatter, etc.) of the last DataFrame, call the `visualize_last_dataframe` tool with the appropriate chart type and context.
- If the user requests data manipulation, generate and execute Python code using pandas/numpy, and return the code and explanation.
- If you call the visualization tool, include its result in your response.
- Never generate SQL or modify the database.
- Respond with a PythonAgentResult object.''',
        "tools": [visualize_last_dataframe],
    }

# 2. Remove direct registration of visualize_last_dataframe from orchestrator and query agent, register python agent instead
# (This will be done in the agent instantiation code below)

# 3. Update query agent system prompt to mention Python agent
# (Edit generate_system_prompt)
def generate_system_prompt() -> str:
    prompt = f"""You are an expert SQL assistant designed to write sql queries.

TARGET DATABASE:
The target database (SQLite) details are provided in each request.

IMPORTANT OUTPUT STRUCTURE:
You MUST return your response as a valid QueryResponse object with these fields:
1. text_message: A human-readable response explaining your findings and analysis.
2. **VERY IMPORTANT: sql_result: You MUST always include SQL query in this field. never return sql_result=None.**

CRITICAL RULES FOR SQL GENERATION:
1. For ANY question about data in the database (counts, totals, listings, comparisons, etc.), you MUST generate an appropriate SQLite query.
2. ALWAYS GENERATE SQL: If the user is asking about specific data, records, amounts, or numbers, you MUST generate SQL - even if you're unsure about exact column names. Missing SQL is a critical error.
3. PAY ATTENTION TO COLUMN NAMES: If a column name in the provided schema contains spaces or special characters, you MUST enclose it in double quotes (e.g., SELECT \"Total IFC Investment Amount\" FROM ...). Failure to quote such names will cause errors. Check for columns like \"IFC investment for Risk Management(Million USD)\", \"IFC investment for Guarantee(Million USD)\", etc.
4. AGGREGATIONS: For questions asking about totals, sums, or aggregations, use SQL aggregate functions (SUM, COUNT, AVG, etc.).
5. GROUPING: When a question mentions \"per\" some field (e.g., \"per product line\"), this requires a GROUP BY clause for that field.
6. SUM FOR TOTALS: Numerical fields asking for totals must use SUM() in your query. Ensure you select the correct column (e.g., \"IFC investment for Loan(Million USD)\" for loan sizes).
7. DATA TYPES: Be mindful that many numeric columns might be stored as TEXT (e.g., \"(Million USD)\" columns). You might need to CAST them to a numeric type (e.g., CAST(\"IFC investment for Loan(Million USD)\" AS REAL)) before performing calculations like AVG or SUM. Handle potential non-numeric values gracefully if possible (e.g., WHERE clause to filter them out before casting, or use `IFNULL(CAST(... AS REAL), 0)`).
8. SECURITY: ONLY generate SELECT queries. NEVER generate SQL statements that modify the database (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.).

PYTHON AGENT TOOL:
- You have access to a 'Python agent tool'.
- If the user's request requires any data manipulation, analysis, or visualization using Python (e.g., using pandas, numpy, plotting charts like bar, line, pie), you MUST call this tool.
- Pass the user's request and any necessary context (like the target database or previous results if applicable) to the Python agent tool.
- Do NOT attempt to generate or execute Python code yourself.
- The Python agent tool will handle the Python execution and visualization, and its result (code, explanation, visualization info) should be included in your final response if the tool was called.

RESPONSE STRUCTURE:
1. First, review the database schema provided in the message.
2. Understand the request: you always need to generate SQL. Does it require Python manipulation/visualization (call Python agent tool)?
3. Generate SQL: Generate an accurate SQLite query string following the rules above.
4. Call Python Agent Tool (if needed): If the request involves Python analysis or visualization, call the Python agent tool. Include its results in the final response (potentially in the python_result field).
5. Explain Clearly: Explain the SQL query (including any casting). If the Python agent tool was called, summarize its findings or indicate that a visualization was prepared.
6. Format Output: Format your final response using the 'QueryResponse' structure. Include 'text_message', 'sql_result' (always include SQL query).
7. **CRUCIAL**: Even if you use internal tools (like `execute_sql`) to find the answer or validate the query during your thought process, the final `QueryResponse` object you return MUST contain the generated SQL query string in the `sql_result` field if the original request required data retrieval from the database.
8. Safety: Focus ONLY on SELECT queries. Do not generate SQL/Python that modifies the database.
9. Efficiency: Write efficient SQL queries. Filter data (e.g., using WHERE clause) before aggregation.

FINAL OUTPUT FORMAT - VERY IMPORTANT:
Your final output MUST be the structured 'QueryResponse' object.
"""
    return prompt

# 4. Update orchestrator system prompt to mention Python agent, not visualization tool
def create_orchestrator_agent_blueprint():
    return {
        "result_type": OrchestratorResult,
        "name": "Orchestrator Agent",
        "retries": 2,
        "system_prompt": '''You are an orchestrator agent for a database query system. Your job is to:

1. ASSISTANT MODE:
   - If the user message is a greeting, general question, or anything NOT related to database queries, respond with action='assistant'.
   - You are a helpful assistant that can greet users, answer general questions, and help users query the World Bank database systems, including IBRD and IDA lending data, IFC investment data and MIGA guarantee information about investments, projects, countries, sectors, and more.
   - If the user asks for clarification about the last response, help them as long as it is related to the World Bank database systems or the outputs of the previous database query.

2. DATABASE QUERY MODE:
   - ONLY set action='database_query' if the user wants to query the database, where SQL is the appropriate response.
   - DO NOT set action='database_query' if the user is asking for clarification about the last response or the previous database query or for a follow-up chart or visualization.
   - Provide a brief helpful response acknowledging the query, such as "I'll search the database for that information" or "Let me find that data for you."


3. VISUALIZATION MODE:
   - Set action='visualization' when the user is requesting to visualize or chart previously retrieved data.
   - you need to compare the user query to the previous dataframe in context to determine if it is a new SQL query or a follow-up chart or visualization related to the previous query.
   - When setting action='visualization', include the chart_type (e.g., "bar", "line", "pie", "scatter") in your response.
   - Provide a brief helpful response acknowledging the visualization request, such as "I'll create a bar chart with the data."

4. PYTHON AGENT USAGE:
   - You have access to a Python agent tool for data manipulation and visualization (e.g., using pandas, numpy, or plotting the last DataFrame as a chart).
   - You need to compare the user query to the previous dataframe in context to determine if it is a new SQL query or a follow-up chart or visualization related to the previous query.
   - Use this tool when the user makes a follow-up request to visualize, plot, or chart the previous results, or requests data manipulation in Python.
   - Call the Python agent with the requested operation and any relevant user context.
   - If the tool returns a successful result, respond to the user with the message and chart type or code from the tool's output. If there is an error, inform the user accordingly.
   - Only use this tool if there is previous data available to manipulate or visualize.
   - IF you determine a query to be a follow-up chart or visualization, set action='visualization' and include the chart type in the response.

5. CONVERSATION MANAGEMENT:
   - Use the conversation history to maintain context.
   - If a follow-up question refers to previous results, treat it as a database query, Python data manipulation, or visualization request or clarification about the last response.

Respond ONLY with the structured OrchestratorResult, including the appropriate action type and chart_type if relevant.'''
    }

# 5. Register python agent as a tool for orchestrator and query agent
# (In agent instantiation, replace visualize_last_dataframe with python agent)

# --- Python Agent Instantiation (singleton for tool use) ---
python_agent_blueprint = create_python_agent_blueprint()
def get_python_agent(local_llm):
    return Agent(local_llm, **python_agent_blueprint)

# Tool function to call the Python agent
async def call_python_agent(ctx: RunContext, prompt: str, **kwargs):
    # Use the same LLM as the parent agent
    local_llm = ctx.model if hasattr(ctx, 'model') else None
    python_agent = get_python_agent(local_llm)
    result = await python_agent.run(prompt, **kwargs)
    return result.output if hasattr(result, 'output') else result


if __name__ == "__main__":
    # Set up global asyncio configuration - Run once at start
    try:
        # nest_asyncio is already applied globally at the top
        logger.info("Verifying nest_asyncio application...") # Add log
        nest_asyncio.apply() # Re-applying is safe

        # Set Windows event loop policy if on Windows (helps prevent some httpx/aiohttp issues)
        if os.name == 'nt':
            try:
                # Check if a policy is already set and if it's the desired one
                current_policy = asyncio.get_event_loop_policy()
                if not isinstance(current_policy, asyncio.WindowsSelectorEventLoopPolicy):
                     # Check if a loop is already running *before* setting the policy
                     try:
                         asyncio.get_running_loop()
                         logger.warning("Event loop already running. Cannot set WindowsSelectorEventLoopPolicy now.")
                     except RuntimeError:
                         # No loop running, safe to set policy
                         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                         logger.info("Successfully set WindowsSelectorEventLoopPolicy.")
                else:
                     logger.info("WindowsSelectorEventLoopPolicy already set.")
            except Exception as policy_e:
                 logger.warning(f"Could not set WindowsSelectorEventLoopPolicy: {policy_e}. Default policy will be used.")
        else:
             logger.info("Not on Windows, skipping Windows event loop policy setting.")

    except Exception as e:
        logger.error(f"Could not set up global asyncio configuration: {e}", exc_info=True)

    # Run the main Streamlit function
    main()
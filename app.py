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
from typing import Dict, List, Union, Optional, Any, AsyncGenerator, Tuple, Literal
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
import functools
import pandas as pd # Import pandas for DataFrame processing in modification

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


async def log_agent_run(agent, prompt, *args, **kwargs):
    logger = logging.getLogger(__name__)
    logger.debug(f"[AGENT RUN] {getattr(agent, 'name', str(agent))} running with prompt:\n{prompt}")
    result = await agent.run(prompt, *args, **kwargs)
    logger.debug(f"[AGENT RESULT] {getattr(agent, 'name', str(agent))} result: {result}")
    return result
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
    python_result: Optional[PythonCodeResult] = Field(None, description="Python code details if Python was generated for *non-visualization* analysis (rarely used).") # Clarified purpose
    visualization_requested: bool = Field(False, description="Set to True if the user's request explicitly asked for a chart, plot, or visualization.") # NEW Field

class DatabaseClassification(BaseModel):
    """
    Identifies the target database for a user query.
    The database_key is now a dynamic string, validated at runtime against metadata.
    """
    database_key: str = Field(..., description="The database key the user query most likely refers to, based on keywords and the database descriptions provided. Use 'UNKNOWN' if the query is ambiguous.")
    reasoning: str = Field(..., description="Brief explanation for the classification (e.g., 'Query mentions IFC investments', 'Query mentions MIGA guarantees', 'Query mentions IBRD and IDA lending', 'Query is ambiguous/general').")

class VisualizationAgentResult(BaseModel):
    """Structured response from the visualization agent"""
    chart_type: Literal["bar", "line", "scatter", "pie", "area", "histogram"] = Field(
        ..., description="Type of chart to generate")
    x_column: str = Field(..., description="Column name to use for the x-axis")
    y_column: str = Field(..., description="Column name to use for the y-axis")
    title: str = Field(..., description="Chart title")
    description: str = Field(..., description="Human-readable explanation of the visualization")
    color_column: Optional[str] = Field(None, description="Optional column to use for color grouping (if applicable)")
    use_streamlit_native: bool = Field(True, description="Whether to use Streamlit's native chart functions (vs matplotlib)")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for chart customization")

# --- Orchestrator Model and Blueprint ---
class OrchestratorResult(BaseModel):
    action: Literal["assistant", "database_query", "visualization"] = Field(
        ..., description="Type of action: 'assistant' for greetings, help, and general conversation, 'database_query' for DB queries, 'visualization' for chart requests."
    )
    response: str = Field(..., description="Assistant's response to the user.")
    chart_type: Optional[str] = Field(None, description="Type of chart requested (for visualization actions only).")

def create_orchestrator_agent_blueprint():
    return {
        "output_type": OrchestratorResult, # Changed result_type to output_type
        "name": "Orchestrator Agent",
        "output_retries": 2, # Changed retries to output_retries
        "system_prompt": '''You are an orchestrator agent for a database query system. Your job is to:

1. ASSISTANT MODE:
   - If the user message is a greeting, general question, or anything NOT related to database queries, respond with action='assistant'.
   - You are a helpful assistant that can greet users, answer general questions, and help users query the World Bank database systems, including IBRD and IDA lending data, IFC investment data and MIGA guarantee information about investments, projects, countries, sectors, and more.
   - If the user asks about the dataset, tables, columns, or wants to know what data is available, you have access to a tool called 'get_metadata_info' which can retrieve descriptions and details about the dataset, tables, and columns. Use this tool to answer questions about the structure, available columns, or descriptions.
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


# --- Define the Agent Blueprints (Prompts, Tools, Validators - Keep as is) ---

# System prompt generator function remains global
def generate_system_prompt() -> str:
    """Generates the full system prompt for the query agent."""
    prompt = f"""You are an expert SQL assistant designed to write sql queries. you will first come up with a plan to execute the user query and then execute the query following all the rules below.

TARGET DATABASE:
The target database (SQLite) details are provided in each request.

IMPORTANT OUTPUT STRUCTURE:
You MUST return your response as a valid QueryResponse object with these fields:
1. text_message: A human-readable response explaining your findings and analysis.
2. **VERY IMPORTANT: sql_result: You MUST always include SQL query in this field. never return sql_result=None.**

--- MODIFICATION REQUESTS ---
If the prompt includes a `Previous SQL Query` and a `User Modification` instruction:
1. **Analyze the `Previous SQL Query`**. 
2. **Analyze the `User Modification`** (e.g., "add column X", "also show Y", "remove Z", "visualize X over time").
3. **Use the provided `Schema`** to understand the available tables and columns.
4. **Use the `get_metadata_info` tool** if necessary to resolve ambiguous column names or find columns not in the provided pruned schema.
5. **Generate a *new* SQL query** that incorporates the user's modification into the `Previous SQL Query`.
6. **If the modification involves visualization:**
    - Include Python code in `python_result` ONLY for necessary data preparation (e.g., converting types with `pd.to_datetime`, sorting with `sort_values`, setting index with `set_index`).
    - **CRITICAL: DO NOT generate plotting code (e.g., using `matplotlib.pyplot`, `seaborn`, `plotly.express`) in the `python_result`.**
    - You MUST explicitly suggest the chart type (e.g., "line chart", "bar chart") in the `text_message`.
7. **Explain** the changes made in the `text_message`.
8. Return the *new* query in the `sql_result` field and any *preparation-only* Python code in `python_result`.

--- NEW QUERY REQUESTS ---
If no `Previous SQL Query` is provided, treat it as a new request and follow the rules below.

CRITICAL RULES FOR SQL GENERATION:
1. For ANY question about data in the database (counts, totals, listings, comparisons, etc.), you MUST generate an appropriate SQLite query.
2. ALWAYS GENERATE SQL: If the user is asking about specific data, records, amounts, or numbers, you MUST generate SQL - even if you're unsure about exact column names. Missing SQL is a critical error.
3. AGGREGATIONS: For questions asking about totals, sums, or aggregations, use SQL aggregate functions (SUM, COUNT, AVG, etc.).
4. GROUPING: When a question mentions \"per\" some field (e.g., \"per product line\"), this requires a GROUP BY clause for that field.
5. SUM FOR TOTALS: Numerical fields asking for totals must use SUM() in your query. Ensure you select the correct column (e.g., \"IFC investment for Loan(Million USD)\" for loan sizes).
6. SECURITY: ONLY generate SELECT queries. NEVER generate SQL statements that modify the database (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.).
7. INCLUDE AT LEAST 3 COLUMNS: Include columns that would be needed to make sense of the query like project name and number. never give only one column in the SELECT query, **always include at least 3 columns**.
8. QUERY NOTES: *IMPORTANT* Pay attention to the query_notes in the metadata for each column. These are important instructions from the data stewards that you must follow. Use them to determine the correct column to use for the query.

PYTHON AGENT TOOL:
- You have access to a 'Python agent tool' (`call_python_agent`).
- If the user's request requires *complex* data manipulation, analysis beyond simple preparation, or a specific type of visualization not directly handled by standard suggestions, you MAY call this tool.
- Pass the user's request and necessary context to the Python agent tool.
- Include its results in your final response.
- **Generally, prefer generating preparation code (if needed) and suggesting chart types directly in your response rather than calling the Python agent unless necessary.**

RESPONSE STRUCTURE:
1. First, review the database schema provided in the message.
2. Understand the request: Modify a previous query or generate a new one? Does it require Python preparation/visualization? Call the Python agent tool only if complex manipulation is needed.
3. Generate SQL: Generate an accurate SQLite query (modified or new).
4. Generate Python (Preparation Only, if needed): If data prep is required for a visualization, generate the Python code (no plotting commands) in `python_result`.
5. Suggest Chart (if requested): If visualization was requested, clearly state the suggested chart type (e.g., "Here is the data, I suggest viewing it as a line chart.") in the `text_message`.
6. Explain Clearly: Explain the SQL query and any Python preparation code.
7. Format Output: Format your final response using the 'QueryResponse' structure. Include 'text_message', 'sql_result'. Include `python_result` only if preparation code was generated.
8. **CRUCIAL**: Even if you use internal tools, the final `QueryResponse` object MUST contain the generated SQL query string in `sql_result` if the original request required data retrieval.
9. Safety: Focus ONLY on SELECT queries. Do not generate SQL/Python that modifies the database.
10. Efficiency: Write efficient SQL queries.

FINAL OUTPUT FORMAT - VERY IMPORTANT:
Your final output MUST be the structured 'QueryResponse' object. Remember the visualization rules: suggest chart type in text, only prep code (no plotting) in python_result.
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
    # We no longer expect the SQL agent to suggest charts or decline.
    # We only care if the visualization_requested flag is set appropriately (optional check).
    # visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation', 'bar chart', 'pie chart', 'histogram', 'line graph', 'scatter plot']
    # is_visualization_request_in_query = any(keyword in original_user_question.lower() for keyword in visualization_keywords)

    # # Optional: Check if flag matches keywords (can be noisy)
    # if is_visualization_request_in_query and not result.visualization_requested:
    #     logger.warning(f"Visualization keywords detected, but visualization_requested flag is False. Query: {original_user_question[:100]}...")
    #     # Maybe raise ModelRetry("Keywords suggest visualization, but flag not set.") - decide if needed
    # elif not is_visualization_request_in_query and result.visualization_requested:
    #      logger.warning(f"No visualization keywords, but visualization_requested flag is True. Query: {original_user_question[:100]}...")
    #      # Maybe raise ModelRetry("Flag set, but no keywords suggest visualization.") - decide if needed

    logger.info("Result validation completed successfully.")
    return result


# --- Blueprint Functions (Return Agent Instances) ---
# IMPORTANT: These functions now just define *how* to create an agent.
# The actual instantiation happens inside the async functions later.

def create_table_selection_agent_blueprint():
    """Returns the CONFIGURATION for the Table Selection Agent."""
    # Note: No model instance passed here anymore
    return {
        "output_type": SuggestedTables, # Changed result_type to output_type
        "name": "Table Selection Agent",
        "output_retries": 2, # Changed retries to output_retries
        "system_prompt": """You are an expert database assistant. Your task is to analyze a user's query and a list of available tables (with descriptions) in a specific database.\nIdentify the **minimum set of table names** from the provided list that are absolutely required to answer the user's query.\nConsider the table names and their descriptions carefully.\n\nYou have access to a tool called 'get_metadata_info' which can retrieve the list of columns and their descriptions for any table in the database. Use this tool to look up the columns and their descriptions for each table if needed, and use this information to make a more informed decision about which tables are relevant to the user query.\n\nOutput ONLY the list of relevant table names and a brief reasoning.""",
        "tools": [get_metadata_info],
    }

def create_query_agent_blueprint():
    """Returns the CONFIGURATION for the main (SQL-focused) Query Agent."""
    return {
        "deps_type": AgentDependencies,
        "output_type": QueryResponse, # Changed result_type to output_type
        "name": "SQL Query Assistant", # Renamed for clarity
        "output_retries": 3, # Changed retries to output_retries
        "system_prompt_func": generate_system_prompt,
        "tools": [execute_sql, get_metadata_info], # Only SQL execution and metadata tools
        "result_validator_func": validate_query_result
    }

def create_column_prune_agent_blueprint():
    """Returns the CONFIGURATION for the Column Pruning Agent."""
    # Note: No model instance passed here anymore
    return {
        "output_type": PrunedSchemaResult, # Changed result_type to output_type
        "name": "Column Pruning Agent",
        "output_retries": 2, # Changed retries to output_retries
        "system_prompt": """You are an expert data analyst assistant. Your task is to prune the schema of a database to include only essential columns for a given user query.\n\nIMPORTANT: The schema of the database will be provided at the beginning of each user message as a JSON object. Use this schema information to understand the database structure and generate an accurate pruned schema JSON.\n\nWhen deciding which columns to keep, always check the \"query_notes\" field in the column metadata (if present). Always  Use any instructions or hints in \"query_notes\" to make more effective pruning decisions.\n\nWhen you output the pruned schema, include all metadata fields for each column (type, description, query_notes, etc.) in the JSON you return.\n\nCRITICAL RULES FOR SCHEMA PRUNING (Identify the intent of the user query and focus on identifying relevant columns needed for SQL query):\n1. Identify which columns are needed based on the user's query (e.g., columns mentioned for SELECT statement, columns needed for filtering, aggregation, or grouping).\n2. Pay attention to specific column names requested or implied by the query.\n3. Use the query_notes field to guide your pruning.\n4. Include key identifying columns that would be needed to make sense of the query like project name, project id and amount.\n\nYOUR GOAL is to output a concise schema JSON containing ONLY the necessary tables and columns needed to generate the correct SQL query.\n\nRESPONSE STRUCTURE:\n1. Review the full schema provided.\n2. Analyze the user query to determine essential columns.\n3. Generate the pruned_schema_string as a JSON string containing only the required columns, with all their metadata fields.\n4. Provide a brief explanation of why these columns were kept.\n5. Format your final response using the 'PrunedSchemaResult' structure.\n\nYour only task is SCHEMA PRUNING.\n""",
        "tools": [get_metadata_info],
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

# --- NEW: Metadata Info Tool for Agents ---
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class MetadataInfoRequest(BaseModel):
    db_key: Optional[str] = Field(None, description="Database key to filter (e.g., 'ifc', 'miga', 'ibrd')")
    table: Optional[str] = Field(None, description="Table name to filter (optional)")
    column: Optional[str] = Field(None, description="Column name to filter (optional)")

class MetadataInfoResult(BaseModel):
    info: Any = Field(..., description="The requested metadata information (database, table, or column descriptions, etc.)")
    message: str = Field(..., description="Human-readable summary of what was returned.")

async def get_metadata_info(ctx: RunContext, request: MetadataInfoRequest) -> MetadataInfoResult:
    """Tool: Retrieve metadata information about the dataset, tables, or columns. Optionally filter by db_key, table, or column."""
    metadata = load_db_metadata()
    if not metadata:
        return MetadataInfoResult(info=None, message="No metadata available.")
    db_key = request.db_key
    table = request.table
    column = request.column
    # Drill down as requested
    if db_key:
        dbs = metadata.get('databases', {})
        db = dbs.get(db_key)
        if not db:
            return MetadataInfoResult(info=None, message=f"Database key '{db_key}' not found.")
        if table:
            tbl = db.get('tables', {}).get(table)
            if not tbl:
                return MetadataInfoResult(info=None, message=f"Table '{table}' not found in database '{db_key}'.")
            if column:
                col = tbl.get('columns', {}).get(column)
                if not col:
                    return MetadataInfoResult(info=None, message=f"Column '{column}' not found in table '{table}' of database '{db_key}'.")
                return MetadataInfoResult(info=col, message=f"Description for column '{column}' in table '{table}' of database '{db_key}'.")
            return MetadataInfoResult(info=tbl, message=f"Description for table '{table}' in database '{db_key}'.")
        return MetadataInfoResult(info=db, message=f"Description for database '{db_key}'.")
    return MetadataInfoResult(info=metadata, message="Full metadata returned.")

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

# --- Refactored: Dynamic, Metadata-Driven Database Classification ---
async def identify_target_database(user_query: str, metadata: Dict) -> Tuple[Optional[str], str, List[str]]:
    logger.info(f"Attempting to identify target database for query: {user_query[:50]}...")
    class DummyCtx:
        pass
    ctx = DummyCtx()
    meta_result = await get_metadata_info(ctx, MetadataInfoRequest())
    meta = meta_result.info if hasattr(meta_result, 'info') else metadata
    if not meta or 'databases' not in meta:
        logger.error("Metadata missing 'databases' key.")
        return None, "Error: 'databases' key missing in metadata configuration.", []
    dbs = meta['databases']
    valid_keys = list(dbs.keys())
    # Build mapping from database_name (case-insensitive) to key
    name_to_key = {}
    db_names = []
    for k, v in dbs.items():
        db_name = v.get('database_name', k)
        name_to_key[db_name.lower()] = k
        db_names.append(db_name)
    # Add 'UNKNOWN' as a valid option
    db_names.append('UNKNOWN')
    descriptions = []
    for key, db_info in dbs.items():
        desc = db_info.get('description', f'Database {key}')
        table_lines = []
        for tname, tinfo in db_info.get('tables', {}).items():
            tdesc = tinfo.get('description', '')
            col_lines = []
            for cname, cinfo in tinfo.get('columns', {}).items():
                cdesc = cinfo.get('description', '')
                col_lines.append(f"      - {cname}: {cdesc}")
            if col_lines:
                table_lines.append(f"    * {tname}: {tdesc}\n" + "\n".join(col_lines))
            else:
                table_lines.append(f"    * {tname}: {tdesc}")
        db_block = f"- {db_info.get('database_name', key)}: {desc}\n" + ("\n".join(table_lines) if table_lines else "")
        descriptions.append(db_block)
    descriptions_str = "\n".join(descriptions)
    db_names_str = ", ".join([f"'{n}'" for n in db_names])
    # --- NEW: Name-based system prompt ---
    classifier_system_prompt = (
        "You are an AI assistant that classifies user queries by deeply analyzing the provided database metadata. "
        "You have access to detailed descriptions of each database, their tables, and columns. "
        "Your job is to match the user's query to the most relevant database by considering all available metadata, "
        "even if the query is general or does not contain explicit keywords. "
        "Use the database, table, and column descriptions to infer the best match. "
        "Respond ONLY with the database name (from the metadata, field 'database_name') in a structured object: {\"database_name\": ..., \"reasoning\": ...}. "
        "Only respond with 'UNKNOWN' if, after considering all metadata, you are truly unable to determine a relevant database. "
        "Output ONLY the structured classification result."
    )
    classification_prompt = f"""DATABASE METADATA (JSON):\n{json.dumps(meta, indent=2)}\n\nUSER QUERY: \"{user_query}\"\n\nBased on the above metadata, which database name ({db_names_str}) is the most relevant target? If the query is ambiguous, unrelated to these databases, or you cannot make a confident choice after considering all metadata, respond with 'UNKNOWN'.\nRespond ONLY with a JSON object: {{\"database_name\": ..., \"reasoning\": ...}}\n"""
    logger.info("--- Sending name-based classification request to LLM ---")
    logger.debug(f"Prompt:\n{classification_prompt}")
    logger.info("--------------------------------------------")
    try:
        global google_api_key
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            logger.info("Configured GenAI SDK within identify_target_database.")
        except Exception as config_err:
            logger.error(f"Failed to configure GenAI SDK for classification: {config_err}", exc_info=True)
            return None, f"Internal Error: Failed to configure AI Service ({config_err}).", []
        gemini_model_name = st.secrets.get("GEMINI_CLASSIFICATION_MODEL", "gemini-1.5-flash")
        local_llm = GeminiModel(model_name=gemini_model_name)
        from pydantic_ai import Agent
        # Define a new result model for name-based output
        class NameBasedClassification(BaseModel):
            database_name: str
            reasoning: str
        classifier_agent = Agent(
            local_llm,
            output_type=NameBasedClassification, # Changed result_type to output_type
            name="Database Classifier",
            output_retries=2, # Changed retries to output_retries
            system_prompt=classifier_system_prompt,
            tools=[get_metadata_info],
        )
        logger.info("Classifier agent created locally with metadata tool.")
        classification_result = run_async_task(lambda: log_agent_run(classifier_agent, classification_prompt))
        if hasattr(classification_result, 'output') and hasattr(classification_result.output, 'database_name'):
            db_name = classification_result.output.database_name.strip()
            reasoning = classification_result.output.reasoning
            logger.info(f"--- LLM Classification Result ---\nDatabase Name: {db_name}\nReasoning: {reasoning}\n-------------------------------")
            db_name_lower = db_name.lower()
            if db_name_lower == "unknown":
                logger.warning(f"LLM classified as UNKNOWN. Triggering user selection. Reasoning: {reasoning}")
                return "USER_SELECTION_NEEDED", f"Could not automatically determine the target database (Reason: {reasoning}). Please select one:", valid_keys
            if db_name_lower in name_to_key:
                canonical_key = name_to_key[db_name_lower]
                return canonical_key, reasoning, valid_keys
            else:
                logger.warning(f"LLM returned a database name '{db_name}' that is not present in the current metadata. Triggering ModelRetry.")
                from pydantic_ai import ModelRetry
                raise ModelRetry(f"The LLM returned a database name '{db_name}' that is not present in the current metadata. Please select one of: {db_names_str}.") # Use db_names_str here
        else:
            logger.error(f"Classification call returned unexpected structure: {classification_result}. Triggering user selection.")
            return "USER_SELECTION_NEEDED", "Failed to get a valid classification structure from the AI. Please select the database.", valid_keys
    except ModelRetry as retry_exc:
        logger.warning(f"ModelRetry triggered in classification: {retry_exc}")
        raise
    except Exception as e:
        logger.exception("Error during database classification LLM call:")
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

    # --- Check for Follow-up Modification Request FIRST ---
    is_modification_request = False
    modification_keywords = ["add", "also show", "include", "remove", "change", "modify", "instead of", "what about", "how about", "can you", "what if"]
    # Check if there's a previous SQL query context
    if 'last_sql_query' in st.session_state and st.session_state.last_sql_query:
        # Simple check: message is short and contains modification keywords
        if len(message.split()) < 20 and any(keyword in message.lower() for keyword in modification_keywords):
            # More refined check: ensure it's not just a greeting or unrelated question
            if not any(greet in message.lower()[:15] for greet in ['hello', 'hi ', 'thanks', 'thank you', 'great']): 
                is_modification_request = True
                logger.info(f"Detected potential SQL modification request: '{message[:50]}...'")

    if is_modification_request:
        try:
            # Retrieve necessary context
            last_sql = st.session_state.last_sql_query
            last_schema = st.session_state.last_pruned_schema
            # Try getting db_key from chartable context first, then fall back
            last_db_key = st.session_state.get('last_chartable_db_key', st.session_state.get('last_db_key')) 
            last_db_path = st.session_state.get('last_target_db_path') # Get DB path

            if not all([last_sql, last_schema, last_db_key, last_db_path]):
                logger.warning("Missing context for SQL modification (Query: %s, Schema: %s, Key: %s, Path: %s), proceeding as new query.", 
                                 bool(last_sql), bool(last_schema), last_db_key, last_db_path)
                # Proceed as if it's not a modification request
            else:
                with st.spinner("Modifying previous query..."):
                    logger.info("Calling run_async_task for run_sql_modification...")
                    async def run_modification_wrapper():
                        return await run_sql_modification(
                            user_message=message,
                            last_sql_query=last_sql,
                            last_pruned_schema=last_schema,
                            last_target_db_key=last_db_key,
                            last_target_db_path=last_db_path
                        )
                    
                    result_dict = run_async_task(run_modification_wrapper)
                    
                    logger.info("run_async_task for run_sql_modification finished.")
                    # Append result to chat history
                    if result_dict:
                        st.session_state.chat_history.append(result_dict)
                        # Save context to LangChain memory
                        if 'lc_memory' in st.session_state and 'content' in result_dict:
                            try:
                                # Append DataFrame summary if available from modification result
                                output_text = result_dict['content']
                                df_to_summarize = None
                                if 'last_chartable_data' in st.session_state and st.session_state.last_chartable_data is not None:
                                     df_to_summarize = st.session_state.last_chartable_data
                                if df_to_summarize is not None and not df_to_summarize.empty:
                                    df_summary = summarize_dataframe(df_to_summarize)
                                    output_text += f"\n\n[DataFrame Summary]\n{df_summary}"
                                st.session_state.lc_memory.save_context({"input": message}, {"output": output_text})
                            except Exception as mem_err:
                                logger.error(f"Error saving modification context to LC memory: {mem_err}")
                    else:
                         # Fallback error message if run_sql_modification returned None unexpectedly
                         st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, something went wrong while modifying the query.", "db_key": last_db_key})
                
                # Rerun to display the modification result
                logger.info("Rerunning Streamlit after SQL modification.")
                st.rerun()
                # Exit handle_user_message after handling modification
                return 

        except Exception as mod_e:
            logger.error(f"Error during SQL modification handling: {mod_e}", exc_info=True)
            st.error(f"An error occurred while modifying the query: {str(mod_e)}")
            # Fall through to treat as a new query if modification fails critically
            logger.warning("Falling back to standard query flow after modification error.")
            # Clear potentially inconsistent modification state if needed
            # (e.g., if run_sql_modification partially set things)
            # Clear modification context to prevent re-triggering on rerun
            st.session_state.last_sql_query = None
            st.session_state.last_pruned_schema = None
            st.session_state.last_target_db_path = None

    # --- If not a modification OR modification failed/missing context, proceed below ---
    logger.info("Proceeding with standard chart/orchestrator/DB flow...")

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

        gemini_model_name = st.secrets.get("GEMINI_ORCHESTRATOR_MODEL", "gemini-2.5-flash-preview-04-17")
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
        
        if hasattr(orchestrator_result, 'output'):
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
            # Store the database selection reasoning for later display
            st.session_state.db_selection_reasoning = reasoning
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

            gemini_model_name = st.secrets.get("GEMINI_TABLE_SELECTION_MODEL", "gemini-1.5-pro")
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

            if table_agent_result and hasattr(table_agent_result, 'output') and isinstance(table_agent_result.output, SuggestedTables):
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
        st.session_state.combined_confirmation_pending = True
        logger.info("Pausing for COMBINED DB & Table confirmation. State saved.")
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
        # Use last_db_key if available
        db_key = st.session_state.get('last_db_key', 'unknown')
        st.session_state.chat_history.append({"role": "assistant", "content": response, "db_key": db_key})
        return
    
    try:
        db_key = st.session_state.get('last_chartable_db_key', 'unknown')
        df = st.session_state.last_chartable_data
        schema = st.session_state.last_pruned_schema if hasattr(st.session_state, 'last_pruned_schema') else ""
        
        # Create an agent context
        local_llm = GeminiModel(model_name=st.secrets.get("GEMINI_VIZ_MODEL", "gemini-2.0-flash"))
        deps = AgentDependencies.create().with_db(db_key)
        context = RunContext(deps=deps, model=local_llm, usage=DEFAULT_USAGE_LIMITS, prompt=message)
        
        try:
            # Wrap the call in run_async_task
            async def run_viz_agent():
                return await call_visualization_agent(
                    context, 
                    df, 
                    message, 
                    db_key,
                    schema
                )
            visualization_result = run_async_task(run_viz_agent)
            
            # Create the streamlit_chart dictionary
            if visualization_result:
                response = visualization_result.description
                streamlit_chart = {
                    "chart_type": visualization_result.chart_type,
                    "data": df,
                    "x": visualization_result.x_column,
                    "y": visualization_result.y_column,
                    "title": visualization_result.title,
                    "color": visualization_result.color_column,
                    "use_container_width": True
                }
                if visualization_result.additional_params:
                    streamlit_chart.update(visualization_result.additional_params)
                
                logger.info(f"Created visualization: {visualization_result.chart_type} chart with x={visualization_result.x_column}, y={visualization_result.y_column}")
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "streamlit_chart": streamlit_chart,
                    "db_key": db_key
                })
            else:
                logger.warning("Visualization agent returned None")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I couldn't create a visualization for this data.",
                    "db_key": db_key
                })
        except Exception as e:
            logger.error(f"Error in visualization agent during follow-up: {str(e)}", exc_info=True)
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"Error creating visualization: {str(e)}",
                "db_key": db_key
            })
    except Exception as e:
        logger.error(f"Error in handle_follow_up_chart: {str(e)}", exc_info=True)
        db_key = st.session_state.get('last_chartable_db_key', 'unknown')
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"I'm sorry, I couldn't create the chart due to an error: {str(e)}",
            "db_key": db_key
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

            gemini_model_name = st.secrets.get("GEMINI_PRUNING_MODEL", "gemini-2.0-flash")
            local_llm_prune = GeminiModel(model_name=gemini_model_name)
            agent_config_prune = create_column_prune_agent_blueprint()
            agent_instance_prune = Agent(local_llm_prune, **agent_config_prune)
            
            # Use run_async_task instead of directly awaiting to handle event loop issues
            async def run_prune_agent():
                return await agent_instance_prune.run(pruning_prompt)
            
            pruning_agent_result = run_async_task(run_prune_agent)

            if hasattr(pruning_agent_result, 'output') and isinstance(pruning_agent_result.output, PrunedSchemaResult):
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

        gemini_model_name = st.secrets.get("GEMINI_QUERY_MODEL", "gemini-2.5-flash-preview-04-17") # Changed default to flash for consistency
        local_llm = GeminiModel(model_name=gemini_model_name)
        agent_config_bp = create_query_agent_blueprint()
        # Instantiate the agent
        local_query_agent = Agent(
            local_llm,
            deps_type=agent_config_bp["deps_type"],
            output_type=agent_config_bp["output_type"],
            name=agent_config_bp["name"],
            output_retries=agent_config_bp["output_retries"],
        )
        # Apply configurations from the blueprint
        local_query_agent.system_prompt(agent_config_bp["system_prompt_func"])
        for tool_func in agent_config_bp["tools"]:
            local_query_agent.tool(tool_func)
        # Register the Python agent tool function for the query agent
        local_query_agent.tool(call_python_agent)
        local_query_agent.output_validator(agent_config_bp["result_validator_func"])
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
        if hasattr(agent_run_result, 'output') and isinstance(agent_run_result.output, QueryResponse):
            response: QueryResponse = agent_run_result.output
            logger.info("Agent response has expected QueryResponse structure.")
            logger.debug(f"AI Response Data: {response}") # Debug log

            # Start building the final message dict
            # Prepend any earlier warnings (e.g., from pruning)
            initial_content = f"{response.text_message}"
            if assistant_chat_message and "content" in assistant_chat_message:
                initial_content = f"{assistant_chat_message['content']}\n\n{initial_content}"

            base_assistant_message = {"role": "assistant", "content": initial_content, "db_key": target_db_key}
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
                            # Create DataFrame from results
                            sql_results_df = pd.DataFrame(sql_execution_result)
                            
                            # Log DataFrame content before any transformations
                            logger.info(f"Raw SQL DataFrame info: {len(sql_results_df)} rows, {len(sql_results_df.columns)} columns")
                            if not sql_results_df.empty:
                                logger.info(f"DataFrame columns: {sql_results_df.columns.tolist()}")
                                logger.info(f"DataFrame preview (first 2 rows):\n{sql_results_df.head(2).to_string()}")
                                # Check for all-NaN or empty values
                                null_counts = sql_results_df.isna().sum()
                                logger.info(f"Null counts per column: {null_counts.to_dict()}")
                                empty_df = sql_results_df.dropna(how='all').empty
                                if empty_df:
                                    logger.warning("SQL DataFrame contains only NaN values. This may cause visualization issues.")
                            
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

                            # Store results in sql_info
                            sql_info["results"] = sql_results_df.to_dict('records')
                            sql_info["columns"] = list(sql_results_df.columns)
                            
                            # Store data for potential follow-up charting and modification
                            # Log DataFrame and check content
                            logger.info(f"SQL DataFrame info before storage: {len(sql_results_df)} rows, {len(sql_results_df.columns)} columns")
                            if len(sql_results_df) > 0:
                                logger.info(f"DataFrame columns: {sql_results_df.columns.tolist()}")
                                logger.info(f"DataFrame preview (first 2 rows):\n{sql_results_df.head(2).to_string()}")
                                # Check for all-NaN values
                                has_empty_data = sql_results_df.dropna(how='all').empty
                                if has_empty_data:
                                    logger.warning("WARNING: SQL DataFrame contains only NaN values. This may cause visualization issues.")
                                
                                # Store only if DataFrame has usable data
                                if not has_empty_data:
                                    st.session_state.last_chartable_data = sql_results_df
                                    st.session_state.last_chartable_db_key = target_db_key
                                    st.session_state.last_sql_query = sql_query
                                    st.session_state.last_pruned_schema = schema_to_use
                                    st.session_state.last_target_db_path = target_db_path
                                    logger.info(f"Stored DataFrame with usable data in last_chartable_data.")
                                else:
                                    # Empty content DataFrame
                                    logger.warning("Not storing DataFrame that contains only NaN/empty values.")
                                    st.session_state.last_chartable_data = None
                                    st.session_state.last_chartable_db_key = None
                                    # Store SQL info anyway for modifications
                                    st.session_state.last_sql_query = sql_query
                                    st.session_state.last_pruned_schema = schema_to_use
                                    st.session_state.last_target_db_path = target_db_path
                            else:
                                # Empty DataFrame
                                logger.warning("Not storing empty DataFrame in last_chartable_data.")
                                st.session_state.last_chartable_data = None
                                st.session_state.last_chartable_db_key = None
                                # Store SQL info anyway for modifications
                                st.session_state.last_sql_query = sql_query
                                st.session_state.last_pruned_schema = schema_to_use
                                st.session_state.last_target_db_path = target_db_path
                            
                            logger.info(f"Stored query context for potential modification: DB='{target_db_key}', Path='{target_db_path}'.")
                        except Exception as df_e:
                            logger.error(f"Error creating/processing DataFrame from SQL results: {df_e}", exc_info=True)
                            sql_info["error"] = f"Error processing SQL results into DataFrame: {df_e}"
                            sql_results_df = pd.DataFrame() # Ensure empty df
                            st.session_state.last_chartable_data = None # Clear chartable data
                            st.session_state.last_chartable_db_key = None
                            st.session_state.last_sql_query = None
                            st.session_state.last_pruned_schema = None
                            st.session_state.last_target_db_path = None
                    else:
                        logger.info("SQL execution successful, 0 rows returned.")
                        sql_info["results"] = []
                        sql_results_df = pd.DataFrame() # Empty df
                        st.session_state.last_chartable_data = None # Clear chartable data
                        st.session_state.last_chartable_db_key = None
                        st.session_state.last_sql_query = None
                        st.session_state.last_pruned_schema = None
                        st.session_state.last_target_db_path = None
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
            if df_for_chart is not None and not df_for_chart.empty and response.visualization_requested:
                logger.info("Visualization requested, calling visualization agent")
                
                # Create a RunContext for the visualization agent
                viz_context = RunContext(deps=deps, model=local_llm, usage=usage, prompt=user_message)
                try:
                    # Call the dedicated visualization agent
                    visualization_result = await call_visualization_agent(
                        viz_context, 
                        df_for_chart, 
                        user_message, 
                        target_db_key,
                        schema_to_use
                    )
                    
                    # Create the streamlit_chart dictionary for display
                    if visualization_result:
                        streamlit_chart = {
                            "chart_type": visualization_result.chart_type,
                            "data": df_for_chart,
                            "x": visualization_result.x_column,
                            "y": visualization_result.y_column,
                            "title": visualization_result.title,
                            "color": visualization_result.color_column,
                            "use_container_width": True
                        }
                        if visualization_result.additional_params:
                            streamlit_chart.update(visualization_result.additional_params)
                        
                        base_assistant_message["streamlit_chart"] = streamlit_chart
                        logger.info(f"Created visualization: {visualization_result.chart_type} chart with x={visualization_result.x_column}, y={visualization_result.y_column}")
                        
                        # Add the visualization description to the message
                        base_assistant_message["content"] += f"\n\n{visualization_result.description}"
                except Exception as e:
                    logger.error(f"Error in visualization agent: {str(e)}", exc_info=True)
                    base_assistant_message["content"] += f"\n\nI couldn't create a visualization due to an error: {str(e)}"
            elif response.visualization_requested:
                if df_for_chart is None or df_for_chart.empty:
                    logger.info("Visualization requested but SQL returned empty results")
                    base_assistant_message["content"] += "\n\nI couldn't create a visualization because the query returned no results."
                else:
                    logger.info("No visualization requested.")


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
                
                # Check DataFrame for visualization
                if df.empty:
                    logger.warning("Combined visualization intent: DataFrame is empty.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I couldn't create a visualization because the query returned no data.",
                        "db_key": st.session_state.get('last_chartable_db_key', 'unknown')
                    })
                    # Clear the combined intent flag
                    del st.session_state['combined_sql_visualization']
                    return final_assistant_message_dict
                
                # Check for all-NaN DataFrame
                if df.dropna(how='all').empty:
                    logger.warning("Combined visualization intent: DataFrame contains only NaN values.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I couldn't create a visualization because the query returned only empty values.",
                        "db_key": st.session_state.get('last_chartable_db_key', 'unknown')
                    })
                    # Clear the combined intent flag
                    del st.session_state['combined_sql_visualization']
                    return final_assistant_message_dict
                
                # Log DataFrame content for debugging
                logger.info(f"DataFrame for combined visualization: {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"DataFrame columns: {df.columns.tolist()}")
                logger.info(f"DataFrame preview (first 2 rows):\n{df.head(2).to_string()}")
                
                db_key = st.session_state.get('last_chartable_db_key', 'unknown')
                schema = st.session_state.last_pruned_schema if hasattr(st.session_state, 'last_pruned_schema') else ""
                
                # Create a context for the visualization agent
                local_llm = GeminiModel(model_name=st.secrets.get("GEMINI_VIZ_MODEL", "gemini-2.0-flash"))
                deps = AgentDependencies.create().with_db(db_key)
                context = RunContext(deps=deps, model=local_llm, usage=DEFAULT_USAGE_LIMITS, prompt=user_message)
                
                try:
                    # Call the visualization agent
                    visualization_result = await call_visualization_agent(
                        context, 
                        df, 
                        user_message, 
                        db_key,
                        schema
                    )
                    
                    # Create the streamlit_chart dictionary
                    if visualization_result:
                        streamlit_chart = {
                            "chart_type": visualization_result.chart_type,
                            "data": df,
                            "x": visualization_result.x_column,
                            "y": visualization_result.y_column,
                            "title": visualization_result.title,
                            "color": visualization_result.color_column,
                            "use_container_width": True
                        }
                        if visualization_result.additional_params:
                            streamlit_chart.update(visualization_result.additional_params)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": visualization_result.description,
                            "streamlit_chart": streamlit_chart,
                            "db_key": db_key
                        })
                        logger.info(f"Created follow-up visualization: {visualization_result.chart_type} chart with x={visualization_result.x_column}, y={visualization_result.y_column}")
                    else:
                        logger.warning("Visualization agent returned None for combined intent")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "I couldn't create a visualization for this data.",
                            "db_key": db_key
                        })
                except Exception as e:
                    logger.error(f"Error in visualization agent for combined intent: {str(e)}", exc_info=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error creating visualization: {str(e)}",
                        "db_key": db_key
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
                
                # Check DataFrame for visualization
                if df.empty:
                    logger.warning("Combined visualization intent: DataFrame is empty.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I couldn't create a visualization because the query returned no data.",
                        "db_key": st.session_state.get('last_chartable_db_key', 'unknown')
                    })
                    # Clear the combined intent flag
                    del st.session_state['combined_sql_visualization']
                    return 
                
                # Check for all-NaN DataFrame
                if df.dropna(how='all').empty:
                    logger.warning("Combined visualization intent: DataFrame contains only NaN values.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I couldn't create a visualization because the query returned only empty values.",
                        "db_key": st.session_state.get('last_chartable_db_key', 'unknown')
                    })
                    # Clear the combined intent flag
                    del st.session_state['combined_sql_visualization']
                    return 
                
                # Log DataFrame content for debugging
                logger.info(f"DataFrame for combined visualization: {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"DataFrame columns: {df.columns.tolist()}")
                logger.info(f"DataFrame preview (first 2 rows):\n{df.head(2).to_string()}")
                
                db_key = st.session_state.get('last_chartable_db_key', 'unknown')
                schema = st.session_state.last_pruned_schema if hasattr(st.session_state, 'last_pruned_schema') else ""
                
                # Create a context for the visualization agent
                local_llm = GeminiModel(model_name=st.secrets.get("GEMINI_VIZ_MODEL", "gemini-2.0-flash"))
                deps = AgentDependencies.create().with_db(db_key)
                context = RunContext(deps=deps, model=local_llm, usage=DEFAULT_USAGE_LIMITS, prompt=user_message)
                
                try:
                    # Call the visualization agent
                    visualization_result = await call_visualization_agent(
                        context, 
                        df, 
                        user_message, 
                        db_key,
                        schema
                    )
                    
                    # Create the streamlit_chart dictionary
                    if visualization_result:
                        streamlit_chart = {
                            "chart_type": visualization_result.chart_type,
                            "data": df,
                            "x": visualization_result.x_column,
                            "y": visualization_result.y_column,
                            "title": visualization_result.title,
                            "color": visualization_result.color_column,
                            "use_container_width": True
                        }
                        if visualization_result.additional_params:
                            streamlit_chart.update(visualization_result.additional_params)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": visualization_result.description,
                            "streamlit_chart": streamlit_chart,
                            "db_key": db_key
                        })
                        logger.info(f"Created follow-up visualization: {visualization_result.chart_type} chart with x={visualization_result.x_column}, y={visualization_result.y_column}")
                    else:
                        logger.warning("Visualization agent returned None for combined intent")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "I couldn't create a visualization for this data.",
                            "db_key": db_key
                        })
                except Exception as e:
                    logger.error(f"Error in visualization agent for combined intent: {str(e)}", exc_info=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error creating visualization: {str(e)}",
                        "db_key": db_key
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
             st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, an internal error occurred, and no response could be generated.", "db_key": st.session_state.get("last_db_key", "unknown")})

    # Always clear the pending state after attempting continuation
    clear_pending_state()
    logger.info(f"continue_after_table_confirmation finished. Duration: {time.time() - start_time:.2f}s")
    # No return needed, side effects (chat history, state clear) are done


def clear_pending_state():
    """Clears session state variables related to pending table confirmation."""
    keys_to_clear = [
        "combined_confirmation_pending", "confirmed_tables",
        "candidate_tables", "all_tables", "table_agent_reasoning",
        "pending_user_message", "pending_db_metadata",
        "pending_target_db_key", "pending_target_db_path",
        "db_selection_pending", "db_selection_reason", "pending_db_keys", "confirmed_db_key"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


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
    #logger.debug(f"Running coroutine '{coro_name}' using run_async_task...")
    
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
        'confirmed_db_key': None,       # Stores the user-confirmed DB key
        # Hybrid Combined Confirmation
        'combined_confirmation_pending': False,  # New combined confirmation flag
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
            <div class="feature-text">{check_img} Get instant SQL-powered insights from World Bank Group databases.</div>
        </div>
        <div class="features-row">
            <div class="feature-text">{check_img} Generate visualizations (bar, line, pie charts) via Python.</div>
            <div class="feature-text">{check_img} SmartQuery automatically identifies the right database for your query.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info('Query data from the World Bank Group datasets publicly available at https://financesone.worldbank.org/.', icon="ℹ️") # Use streamlit icon
    # --- Example Queries ---
    st.markdown("""
    <div class="example-queries">
        <p>Example Questions:</p>
        <ul>
            <li>"Get me the top 10 countries with the highest count of advisory projects approved in 2024"</li>
            <li>"Show me the sum of IBRD loans to India approved since 2020 per year"</li>
            <li>"Compare the average IFC investment size for 'Loan' products between Nepal and Bhutan."</li>
            <li>"What is the total gross guarantee exposure for MIGA in the Tourism sector in Senegal?"</li>
            <li>"Give me the top 10 IFC equity investments from China"</li>
            <li>"Get me the status of all IDA Projects approved in 2020 to St. Lucia."</li>
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
    if st.session_state.get("confirmed_db_key") is not None and not st.session_state.get("db_selection_pending", False):
        clear_pending_state()
        st.rerun()

    # 2. Handle Combined Confirmation Continuation
    if st.session_state.get("combined_confirmation_pending", False) and st.session_state.get("confirmed_tables") is not None:
        try:
            with st.spinner("Processing confirmed selections and generating query..."):
                async def combined_wrapper_func():
                    return await run_post_combined_confirmation()
                run_async_task(combined_wrapper_func)
                st.rerun()
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
            clear_pending_state()
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
                # Display database badge if present
                db_key = message.get("db_key")
                if db_key and role == "assistant":
                    # Map database keys to colors and icons
                    db_settings = {
                        "ifc": {"color": "blue", "icon": ":material/business:"},
                        "miga": {"color": "green", "icon": ":material/shield:"},
                        "ibrd": {"color": "violet", "icon": ":material/account_balance:"},
                        "ida": {"color": "violet", "icon": ":material/account_balance:"},
                        "unknown": {"color": "gray", "icon": ":material/database:"}
                    }
                    # Get appropriate settings or default
                    settings = db_settings.get(db_key.lower(), {"color": "blue", "icon": ":material/database:"})
                    st.badge(db_key, color=settings["color"], icon=settings["icon"])
                
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

                # Display Streamlit chart if present
                # This section handles charts generated by the main SQL -> Visualization agent flow
                if "streamlit_chart" in message and isinstance(message["streamlit_chart"], dict):
                    st.markdown("**Visualization:**")
                    try:
                        chart_info = message["streamlit_chart"]
                        chart_type = chart_info.get("chart_type") # CORRECT KEY
                        df = chart_info.get("data")
                        x_col = chart_info.get("x")
                        y_col = chart_info.get("y")
                        color_col = chart_info.get("color")
                        title = chart_info.get("title", "Chart")

                        # Ensure df is a pandas DataFrame and has the required columns
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            if x_col and x_col in df.columns and y_col and y_col in df.columns:
                                # Prepare dataframe for plotting (set index)
                                plot_df = df.set_index(x_col)

                                # Select only the y_col for basic charts
                                if color_col and color_col in plot_df.columns:
                                     # If color is specified, keep it for potential use (e.g., scatter)
                                     # For bar/line, Streamlit often handles color based on columns selected
                                     plot_data = plot_df[[y_col, color_col]]
                                else:
                                     plot_data = plot_df[[y_col]] # Select only y_col

                                if chart_type == "bar":
                                    st.bar_chart(plot_data)
                                elif chart_type == "line":
                                    st.line_chart(plot_data)
                                elif chart_type == "area":
                                    st.area_chart(plot_data)
                                elif chart_type == "scatter":
                                    # Use original df and specify x, y, color for scatter
                                    st.scatter_chart(df, x=x_col, y=y_col, color=color_col)
                                elif chart_type == "pie":
                                    # Plotly pie chart logic using specified columns
                                    try:
                                        import plotly.express as px
                                        # Ensure y_col is numeric for values
                                        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                                        df.dropna(subset=[y_col], inplace=True)
                                        if not df.empty:
                                            fig = px.pie(df,
                                                         values=y_col,
                                                         names=x_col, # Use x_col for names
                                                         title=title,
                                                         color=color_col)
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.warning(f"No numeric data found in column '{y_col}' for pie chart.")
                                    except Exception as px_e:
                                        st.error(f"Error creating pie chart: {str(px_e)}")
                                else:
                                    # Unknown chart type fallback
                                    st.warning(f"Unsupported chart type: {chart_type}. Displaying data table.")
                                    st.dataframe(df)
                            else:
                                st.warning(f"Cannot generate chart: X ({x_col}) or Y ({y_col}) columns not found in data.")
                                st.dataframe(df) # Show data if columns missing
                        else:
                            st.warning(f"Cannot generate chart: Invalid or empty data provided.")
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        logger.exception(f"Error rendering chart: {e}")

                # --- REMOVE OR COMMENT OUT the legacy/duplicate "chart" block --- 
                # # Display charts using the "chart" key format from handle_follow_up_chart
                # if "chart" in message and isinstance(message["chart"], dict):
                #    st.markdown("**Visualization:**")
                #    # ... (Keep commented out or remove) ...

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
                            # Set a manual selection reasoning message
                            st.session_state.db_selection_reasoning = "This database was manually selected by you."
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
            db_selection_reasoning = st.session_state.get("db_selection_reasoning", "")

            with st.chat_message("assistant"):
                # Show which database was auto-selected
                db_display_name = db_key
                if "pending_db_metadata" in st.session_state:
                    db_metadata = st.session_state.get("pending_db_metadata")
                    db_entry = db_metadata.get('databases', {}).get(db_key, {})
                    if db_entry and 'database_name' in db_entry:
                        db_display_name = f"{db_entry['database_name']} ({db_key})"
                
                st.info(f"**Selected Database:** {db_display_name}", icon="🔍")
                
                # Show the AI's reasoning for selecting this database
                if db_selection_reasoning:
                    st.caption(f"**Why this database?** {db_selection_reasoning}")
                
                # Add a button to change the database selection
                if st.button("Change Database", key="change_db_button"):
                    logger.info(f"User requested to change database from: {db_key}")
                    # Set the state for database selection UI
                    st.session_state.db_selection_pending = True
                    # Load the database metadata if available
                    if "pending_db_metadata" in st.session_state:
                        db_metadata = st.session_state.get("pending_db_metadata")
                        st.session_state.pending_db_keys = list(db_metadata.get('databases', {}).keys())
                    else:
                        # Fallback in case metadata is missing
                        logger.warning("Missing db_metadata when trying to change database")
                        db_metadata = load_db_metadata()
                        if db_metadata:
                            st.session_state.pending_db_metadata = db_metadata
                            st.session_state.pending_db_keys = list(db_metadata.get('databases', {}).keys())
                        
                    # Keep the user message (already stored in pending_user_message)
                    st.session_state.db_selection_reason = "Please select a different database for your query."
                    # Clear table confirmation to avoid conflicting states
                    st.session_state.table_confirmation_pending = False
                    # Rerun immediately to show the database selection UI
                    st.rerun()
                
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

        # --- Combined Confirmation UI ---
        elif st.session_state.get("combined_confirmation_pending", False):
            candidate_tables = st.session_state.get("candidate_tables", [])
            all_tables = st.session_state.get("all_tables", [])
            table_reasoning = st.session_state.get("table_agent_reasoning", "")
            db_key = st.session_state.get("pending_target_db_key", "")
            db_selection_reasoning = st.session_state.get("db_selection_reasoning", "")
            with st.chat_message("assistant"):
                db_display_name = db_key
                if "pending_db_metadata" in st.session_state:
                    db_metadata = st.session_state.get("pending_db_metadata")
                    db_entry = db_metadata.get('databases', {}).get(db_key, {})
                    db_display_name = f"{db_entry.get('database_name', db_key)} ({db_key})"
                st.info(f"**Selected Database:** {db_display_name}", icon="✅")
                if db_selection_reasoning:
                    st.caption(f"**Reason:** {db_selection_reasoning}")
                st.info(f"**Table Selection for {db_key}:** Please confirm or select the tables needed.", icon="ℹ️")
                if candidate_tables:
                    st.markdown(f"Suggested tables: `{', '.join(candidate_tables)}`")
                    if table_reasoning:
                        st.caption(f"Reasoning: {table_reasoning}")
                elif table_reasoning:
                    st.caption(f"Reasoning: {table_reasoning}")
                selected_tables = st.multiselect(
                    f"Confirm/adjust tables:",
                    options=all_tables,
                    default=candidate_tables,
                    key="combined_table_confirm_multiselect"
                )
                col1, col2 = st.columns([1,3])
                with col1:
                    if st.button("Confirm Selections"):
                        if not selected_tables:
                            st.warning("Please select at least one table.")
                        else:
                            st.session_state.confirmed_tables = selected_tables
                            st.rerun()
                with col2:
                    if st.button("Change Database Instead"):
                        st.session_state.db_selection_pending = True
                        st.session_state.combined_confirmation_pending = False
                        st.rerun()

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
                     db_key = st.session_state.get("last_db_key", "unknown")
                     st.session_state.chat_history.append({"role": "assistant", "content": error_content, "db_key": db_key})
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
        "output_type": PythonAgentResult, # Changed result_type to output_type
        "name": "Python Data & Visualization Agent",
        "output_retries": 2, # Changed retries to output_retries
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
    prompt = f"""You are an expert SQL assistant designed to write sql queries. you will first come up with a plan to execute the user query and then execute the query following all the rules below.

TARGET DATABASE:
The target database (SQLite) details are provided in each request.

IMPORTANT OUTPUT STRUCTURE:
You MUST return your response as a valid QueryResponse object with these fields:
1. text_message: A human-readable response explaining your findings and analysis.
2. **VERY IMPORTANT: sql_result: You MUST always include SQL query in this field. never return sql_result=None.**

--- MODIFICATION REQUESTS ---
If the prompt includes a `Previous SQL Query` and a `User Modification` instruction:
1. **Analyze the `Previous SQL Query`**. 
2. **Analyze the `User Modification`** (e.g., "add column X", "also show Y", "remove Z", "visualize X over time").
3. **Use the provided `Schema`** to understand the available tables and columns.
4. **Use the `get_metadata_info` tool** if necessary to resolve ambiguous column names or find columns not in the provided pruned schema.
5. **Generate a *new* SQL query** that incorporates the user's modification into the `Previous SQL Query`.
6. **If the modification involves visualization:**
    - Include Python code in `python_result` ONLY for necessary data preparation (e.g., converting types with `pd.to_datetime`, sorting with `sort_values`, setting index with `set_index`).
    - **CRITICAL: DO NOT generate plotting code (e.g., using `matplotlib.pyplot`, `seaborn`, `plotly.express`) in the `python_result`.**
    - You MUST explicitly suggest the chart type (e.g., "line chart", "bar chart") in the `text_message`.
7. **Explain** the changes made in the `text_message`.
8. Return the *new* query in the `sql_result` field and any *preparation-only* Python code in `python_result`.

--- NEW QUERY REQUESTS ---
If no `Previous SQL Query` is provided, treat it as a new request and follow the rules below.

CRITICAL RULES FOR SQL GENERATION:
1. For ANY question about data in the database (counts, totals, listings, comparisons, etc.), you MUST generate an appropriate SQLite query.
2. ALWAYS GENERATE SQL: If the user is asking about specific data, records, amounts, or numbers, you MUST generate SQL - even if you're unsure about exact column names. Missing SQL is a critical error.
3. AGGREGATIONS: For questions asking about totals, sums, or aggregations, use SQL aggregate functions (SUM, COUNT, AVG, etc.).
4. GROUPING: When a question mentions \"per\" some field (e.g., \"per product line\"), this requires a GROUP BY clause for that field.
5. SUM FOR TOTALS: Numerical fields asking for totals must use SUM() in your query. Ensure you select the correct column (e.g., \"IFC investment for Loan(Million USD)\" for loan sizes).
6. SECURITY: ONLY generate SELECT queries. NEVER generate SQL statements that modify the database (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.).
7. INCLUDE AT LEAST 3 COLUMNS: Include columns that would be needed to make sense of the query like project name and number. never give only one column in the SELECT query, **always include at least 3 columns**.
8. QUERY NOTES: *IMPORTANT* Pay attention to the query_notes in the metadata for each column. These are important instructions from the data stewards that you must follow. Use them to determine the correct column to use for the query.

PYTHON AGENT TOOL:
- You have access to a 'Python agent tool' (`call_python_agent`).
- If the user's request requires *complex* data manipulation, analysis beyond simple preparation, or a specific type of visualization not directly handled by standard suggestions, you MAY call this tool.
- Pass the user's request and necessary context to the Python agent tool.
- Include its results in your final response.
- **Generally, prefer generating preparation code (if needed) and suggesting chart types directly in your response rather than calling the Python agent unless necessary.**

RESPONSE STRUCTURE:
1. First, review the database schema provided in the message.
2. Understand the request: Modify a previous query or generate a new one? Does it require Python preparation/visualization? Call the Python agent tool only if complex manipulation is needed.
3. Generate SQL: Generate an accurate SQLite query (modified or new).
4. Generate Python (Preparation Only, if needed): If data prep is required for a visualization, generate the Python code (no plotting commands) in `python_result`.
5. Suggest Chart (if requested): If visualization was requested, clearly state the suggested chart type (e.g., "Here is the data, I suggest viewing it as a line chart.") in the `text_message`.
6. Explain Clearly: Explain the SQL query and any Python preparation code.
7. Format Output: Format your final response using the 'QueryResponse' structure. Include 'text_message', 'sql_result'. Include `python_result` only if preparation code was generated.
8. **CRUCIAL**: Even if you use internal tools, the final `QueryResponse` object MUST contain the generated SQL query string in `sql_result` if the original request required data retrieval.
9. Safety: Focus ONLY on SELECT queries. Do not generate SQL/Python that modifies the database.
10. Efficiency: Write efficient SQL queries.

FINAL OUTPUT FORMAT - VERY IMPORTANT:
Your final output MUST be the structured 'QueryResponse' object. Remember the visualization rules: suggest chart type in text, only prep code (no plotting) in python_result.
"""
    return prompt

# 4. Update orchestrator system prompt to mention Python agent, not visualization tool
def create_orchestrator_agent_blueprint():
    return {
        "output_type": OrchestratorResult, # Changed result_type to output_type
        "name": "Orchestrator Agent",
        "output_retries": 2, # Changed retries to output_retries
        "system_prompt": '''You are an orchestrator agent for a database query system. Your job is to:

1. ASSISTANT MODE:
   - [DEFAULT MODE] If the user message is a greeting, general question, or anything NOT related to database queries, respond with action='assistant'.
   - You are a helpful assistant that can greet users, answer general questions, and help users query the World Bank database systems, including IBRD and IDA lending data, IFC investment data and MIGA guarantee information about investments, projects, countries, sectors, and more.
   - If the user asks about the dataset, tables, columns, or wants to know what data is available, you have access to a tool called 'get_metadata_info' which can retrieve descriptions and details about the dataset, tables, and columns. Use this tool to answer questions about the structure, available columns, or descriptions.
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

Respond ONLY with the structured OrchestratorResult, including the appropriate action type and chart_type if relevant.''',
        "tools": [get_metadata_info],
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

# --- Visualization Agent Blueprint and Call Function ---
def generate_visualization_system_prompt() -> str:
    """Generates the system prompt for the visualization expert agent"""
    prompt = """You are a data visualization expert. Your job is to recommend the best chart type and parameters for visualizing a SQL query result.

INPUT DATA:
1. A pandas DataFrame (summarized with example rows and column types)
2. The user's original query or visualization request
3. The database schema with column descriptions
4. Metadata about the columns

OUTPUT STRUCTURE:
You MUST return a VisualizationAgentResult with:
1. chart_type: The best chart type for the data ("bar", "line", "scatter", "pie", "area", "histogram")
2. x_column: The column name for the x-axis (must be a valid column in the DataFrame)
3. y_column: The column name for the y-axis (must be a valid column in the DataFrame)
4. title: A descriptive title for the chart
5. description: A brief explanation of why this visualization is appropriate
6. color_column: (Optional) A column to use for color grouping if appropriate
7. use_streamlit_native: Keep this as True to use Streamlit's built-in charting
8. additional_params: (Optional) Parameters like width, height, etc.

CRITICAL RULES:
1. COLUMN SELECTION IS CRUCIAL:
   - The x_column and y_column MUST be valid columns in the provided DataFrame
   - Select columns that make semantic sense (e.g., dates on x-axis for time series, categories for bar charts)
   - Numeric columns are typically appropriate for y-axis
   - Categorical/date columns are typically appropriate for x-axis

2. CHART TYPE SELECTION:
   - TIME SERIES: Use line charts for data over time
   - COMPARISONS: Use bar charts for comparing categories
   - DISTRIBUTIONS: Use histograms for frequency distributions
   - RELATIONSHIPS: Use scatter plots for correlations between numeric variables
   - PROPORTIONS: Use pie charts for parts of a whole (only if few categories)

3. INSIGHTS FOCUS:
   - Your visualization should highlight the most important insights in the data
   - Consider what the user is likely trying to understand from their query

4. CONTEXT AWARENESS:
   - Use the column descriptions and metadata to understand what each column represents
   - Base your visualization on the user's original question/intent

You are the EXPERT in choosing which columns should be visualized. DO NOT simply visualize all columns. Analyze the data and determine which 2-3 columns are most relevant to answering the user's question."""
    return prompt

def create_visualization_agent_blueprint():
    """Returns the CONFIGURATION for the Visualization Expert Agent"""
    return {
        "deps_type": AgentDependencies,
        "output_type": VisualizationAgentResult, # Changed result_type to output_type
        "name": "Visualization Expert",
        "output_retries": 2, # Changed retries to output_retries
        "system_prompt_func": generate_visualization_system_prompt,
        "tools": [get_metadata_info],  # Only needs metadata tool
        "result_validator_func": None  # Can add validation later if needed
    }

# Create a function to get the visualization agent
visualization_agent_blueprint = create_visualization_agent_blueprint()
def get_visualization_agent(local_llm):
    """Gets a visualization agent instance"""
    visualization_agent = Agent(
        local_llm,
        deps_type=visualization_agent_blueprint["deps_type"],
        output_type=visualization_agent_blueprint["output_type"], # Changed result_type to output_type
        name=visualization_agent_blueprint["name"],
        output_retries=visualization_agent_blueprint["output_retries"], # Changed retries to output_retries
    )
    visualization_agent.system_prompt(visualization_agent_blueprint["system_prompt_func"])
    for tool_func in visualization_agent_blueprint["tools"]:
        visualization_agent.tool(tool_func)
    return visualization_agent

async def call_visualization_agent(
    ctx: RunContext, 
    df: pd.DataFrame, 
    user_query: str, 
    db_key: str,
    pruned_schema: str,
    **kwargs
) -> VisualizationAgentResult:
    """Call the visualization agent to get chart recommendations for a dataframe."""
    # Use the same LLM as the parent agent
    local_llm = ctx.model
    logger.info(f"Calling visualization agent for {db_key} dataframe with {len(df)} rows, {df.columns.size} columns")
    
    # ENHANCED CHECKS: Check DataFrame and user query before building prompt
    
    # 1. Check if DataFrame exists and has columns
    if df is None:
        logger.error("Visualization agent called with None DataFrame. Aborting.")
        raise ValueError("Visualization agent called with None DataFrame!")
    
    if df.empty:
        logger.error("Visualization agent called with empty DataFrame. Aborting.")
        raise ValueError("Visualization agent called with empty DataFrame!")
    
    # 2. Check for all-NaN DataFrame
    if df.dropna(how='all').empty:
        logger.error("Visualization agent called with DataFrame containing only NaN values. Aborting.")
        raise ValueError("Visualization agent called with DataFrame containing only NaN values!")
    
    # 3. Check if user query exists
    if not user_query or not user_query.strip():
        logger.error("Visualization agent called with empty user query. Aborting.")
        raise ValueError("Visualization agent called with empty user query!")
    
    # 4. Log DataFrame content for debugging
    logger.info(f"DataFrame for visualization: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame has null values: {df.isnull().values.any()}")
    if df.isnull().values.any():
        null_counts = df.isnull().sum()
        non_zero_nulls = null_counts[null_counts > 0]
        logger.info(f"Null counts in columns: {non_zero_nulls.to_dict()}")
    logger.info(f"DataFrame preview (first 2 rows):\n{df.head(2).to_string()}")
    logger.info(f"User query: '{user_query}'")
    
    # Build the prompt with all necessary context
    prompt = f"""
USER QUERY OR VISUALIZATION REQUEST:
{user_query}

DATA SUMMARY:
DataFrame with {len(df)} rows and {df.columns.size} columns.

Columns and their types:
{', '.join([f"{col} ({df[col].dtype})" for col in df.columns])}

First {min(2, len(df))} rows of the data for reference:
{df.head(2).to_string()}

DATABASE CONTEXT:
Database: {db_key}
{pruned_schema}

Based on the above information, determine the most appropriate visualization for this data that answers the user's query.
"""
    
    # Check if prompt is empty
    if not prompt or not prompt.strip():
        logger.error("Visualization agent called with empty prompt. Aborting.")
        raise ValueError("Visualization agent called with empty prompt!")
    
    # Check prompt has alphanumeric content
    import re
    if not re.search(r'\w', prompt):
        logger.error("Visualization agent prompt contains no alphanumeric content. Full prompt:")
        logger.error(prompt)
        raise ValueError("Visualization agent prompt contains no alphanumeric content!")
    
    # Log prompt length and excerpt
    logger.info(f"Visualization agent prompt length: {len(prompt)} characters")
    logger.info(f"Prompt excerpt: {prompt[:300]}...")
    
    # Create a new instance of the visualization agent
    visualization_agent = get_visualization_agent(local_llm)
    
    # Prepare dependencies
    deps = ctx.deps if ctx.deps else AgentDependencies.create().with_db(db_key)
    
    # Call the agent
    logger.info(f"Calling visualization agent with positional prompt of length {len(prompt)}")
    # Define usage object directly
    usage = Usage()
    # Pass prompt, deps, and usage object
    result = await visualization_agent.run(prompt, deps=deps, usage=usage)
    
    logger.info(f"Visualization agent recommended {result.output.chart_type} chart with x={result.output.x_column}, y={result.output.y_column}")
    return result.output

# --- Add a helper function for SQL modification ---
async def run_sql_modification(
    user_message: str, 
    last_sql_query: str, 
    last_pruned_schema: str,
    last_target_db_key: str,
    last_target_db_path: str
) -> dict:
    """Runs the SQL agent directly for a modification request."""
    start_mod_time = time.time()
    logger.info(f"Running SQL modification for DB: {last_target_db_key}, Path: {last_target_db_path}")
    deps = None
    final_assistant_message_dict = None

    try:
        logger.info("Instantiating LLM/Agent for SQL modification...")
        global google_api_key
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            logger.info("Configured GenAI SDK within run_sql_modification.")
        except Exception as config_err:
            logger.error(f"Failed to configure GenAI SDK for modification: {config_err}", exc_info=True)
            return {"role": "assistant", "content": f"Internal Error: Failed to configure AI Service ({config_err})."}

        gemini_model_name = st.secrets.get("GEMINI_QUERY_MODEL", "gemini-2.5-flash-preview-04-17")
        local_llm = GeminiModel(model_name=gemini_model_name)
        agent_config_bp = create_query_agent_blueprint()
        local_query_agent = Agent(
            local_llm,
            deps_type=agent_config_bp["deps_type"],
            output_type=agent_config_bp["output_type"],
            name=agent_config_bp["name"],
            output_retries=agent_config_bp["output_retries"],
        )
        local_query_agent.system_prompt(agent_config_bp["system_prompt_func"])
        for tool_func in agent_config_bp["tools"]:
            local_query_agent.tool(tool_func)
        local_query_agent.tool(call_python_agent) # Ensure Python agent tool is available
        local_query_agent.output_validator(agent_config_bp["result_validator_func"])
        logger.info("Query agent created locally for SQL modification flow.")

        logger.info(f"Connecting to database: {last_target_db_path} for modification.")
        deps = AgentDependencies.create().with_db(db_path=last_target_db_path)
        if not deps.db_connection:
            return {"role": "assistant", "content": f"Sorry, I couldn't connect to the {last_target_db_key} database at {last_target_db_path} for modification."}
        logger.info("Database connection successful for modification.")

        usage = Usage()
        # --- Construct the modification prompt ---
        modification_prompt = f'''Target Database: {last_target_db_key}
Schema (this is a PRUNED schema from the previous query and may not contain all columns):
{last_pruned_schema}

Previous SQL Query:
```sql
{last_sql_query}
```

User Modification Request: {user_message}

IMPORTANT: If the user is requesting a column that is not in the pruned schema above, you MUST use the get_metadata_info tool to look up the full schema and find the requested column. Do not tell the user the column doesn't exist without checking the full metadata first.

INSTRUCTIONS:
1. Analyze the 'Previous SQL Query' and the 'User Modification Request'.
2. If the user is requesting a column that's not in the pruned schema, use the get_metadata_info tool to find the column in the full database schema.
   Example call: get_metadata_info(MetadataInfoRequest(db_key="{last_target_db_key}"))
3. Modify the 'Previous SQL Query' based on the user's request and generate a new SQL query.
4. If you find the requested column via metadata, include it in your response.
5. Explain the changes made to the query.

Modify the 'Previous SQL Query' based on the 'User Modification Request', but use the metadata tool to look up any missing columns.'''
        logger.info(f"Prompt sent to LLM (SQL modification):\n{modification_prompt}")

        agent_run_result = None
        try:
            logger.info("Running query agent for modification...")
            async def run_mod_agent():
                return await local_query_agent.run(
                    modification_prompt,
                    deps=deps,
                    usage=usage,
                    usage_limits=DEFAULT_USAGE_LIMITS
                )
            agent_run_result = run_async_task(run_mod_agent)
            run_duration = time.time() - start_mod_time
            logger.info(f"SQL Modification Agent call completed. Duration: {run_duration:.2f}s.")
        except Exception as agent_run_e:
            logger.error(f"SQL Modification Agent agent.run() failed: {str(agent_run_e)}", exc_info=True)
            return {"role": "assistant", "content": f"Sorry, the AI agent failed to modify your query for the {last_target_db_key} database. Error: {str(agent_run_e)}"}

        # --- Process Agent Result (Similar to run_agents_post_confirmation_inner, but simpler) ---
        if hasattr(agent_run_result, 'output') and isinstance(agent_run_result.output, QueryResponse):
            response: QueryResponse = agent_run_result.output
            logger.info("Modification agent response has expected QueryResponse structure.")
            # Start building the final message dict
            base_assistant_message = {"role": "assistant", "content": response.text_message, "db_key": last_target_db_key}
            sql_results_df = None
            sql_info = {}
            new_sql_query = None # Store the newly generated query

            if response.sql_result:
                new_sql_query = response.sql_result.sql_query # Get the modified query
                logger.info(f"MODIFIED SQL query generated: {new_sql_query[:100]}...")
                sql_info = {"query": new_sql_query, "explanation": response.sql_result.explanation}
                
                # Execute the modified SQL query
                sql_run_context = RunContext(deps=deps, model=local_llm, usage=usage, prompt=new_sql_query)
                async def run_new_sql():
                    return await execute_sql(sql_run_context, new_sql_query)
                sql_execution_result = run_async_task(run_new_sql)

                if isinstance(sql_execution_result, str) and sql_execution_result.startswith("Error:"):
                    sql_info["error"] = sql_execution_result
                    base_assistant_message["content"] += f"\n\n**Warning:** Error executing modified SQL: `{sql_execution_result}`"
                    sql_results_df = pd.DataFrame() # Ensure empty df
                    # Don't update last_sql_query on error
                elif isinstance(sql_execution_result, list):
                    if sql_execution_result:
                        sql_results_df = pd.DataFrame(sql_execution_result)
                        # Optional: Add numeric conversion from run_agents_post_confirmation_inner if desired
                        sql_info["results"] = sql_results_df.to_dict('records')
                        sql_info["columns"] = list(sql_results_df.columns)
                        st.session_state.last_chartable_data = sql_results_df # Update chartable data
                        st.session_state.last_chartable_db_key = last_target_db_key
                        # UPDATE context with the NEW successful query
                        st.session_state.last_sql_query = new_sql_query
                        st.session_state.last_pruned_schema = last_pruned_schema # Keep the same schema
                        st.session_state.last_target_db_path = last_target_db_path # Keep the same path
                        logger.info("Updated context after successful modification.")
                    else:
                        # Query ran successfully, but returned 0 rows
                        sql_info["results"] = []
                        sql_results_df = pd.DataFrame()
                        st.session_state.last_chartable_data = None # Clear chartable data
                        st.session_state.last_chartable_db_key = None
                        # UPDATE context with the NEW successful query (even if 0 results)
                        st.session_state.last_sql_query = new_sql_query 
                        st.session_state.last_pruned_schema = last_pruned_schema
                        st.session_state.last_target_db_path = last_target_db_path
                        logger.info("Updated context after successful modification (0 results).")
                else:
                    # Handle unexpected result type from execute_sql
                    sql_info["error"] = "Unexpected result from SQL execution."
                    sql_results_df = pd.DataFrame()
                    st.session_state.last_chartable_data = None
                    st.session_state.last_chartable_db_key = None
                    # Don't update last_sql_query on error

                base_assistant_message["sql_result"] = sql_info
            else:
                # Agent failed to return SQL - should be handled by validator, but add fallback
                base_assistant_message["content"] += "\n\n**Note:** The agent did not return a modified SQL query."
                # Don't update last_sql_query

            # --- Python result handling (if needed/returned by agent) ---
            # Add similar logic as in run_agents_post_confirmation_inner if the agent
            # might return python_result during modification
            if response.python_result:
                 logger.info("Python code returned during modification...")
                 # Add python_result processing here if required
                 base_assistant_message["python_result"] = {
                     "code": response.python_result.python_code,
                     "explanation": response.python_result.explanation
                 }
                 # Note: Executing Python code here might be complex if it depends on the SQL result

            # --- Chart Generation Check (based on text message suggestion) ---
            # Add similar logic as in run_agents_post_confirmation_inner
            chart_type = None
            if sql_results_df is not None and not sql_results_df.empty and response.visualization_requested:
                logger.info("Visualization requested in SQL modification, calling visualization agent")

                # Create a RunContext for the visualization agent
                viz_context = RunContext(deps=deps, model=local_llm, usage=usage, prompt=user_message)
                try:
                    # *** WRAP the call in run_async_task ***
                    async def run_viz_agent_mod():
                        return await call_visualization_agent(
                            viz_context,
                            sql_results_df,
                            user_message,
                            last_target_db_key,
                            last_pruned_schema
                        )

                    visualization_result = run_async_task(run_viz_agent_mod)

                    # Create the streamlit_chart dictionary for display
                    if visualization_result:
                        streamlit_chart = {
                            "chart_type": visualization_result.chart_type,
                            "data": sql_results_df,
                            "x": visualization_result.x_column,
                            "y": visualization_result.y_column,
                            "title": visualization_result.title,
                            "color": visualization_result.color_column,
                            "use_container_width": True
                        }
                        if visualization_result.additional_params:
                            streamlit_chart.update(visualization_result.additional_params)
                        
                        base_assistant_message["streamlit_chart"] = streamlit_chart
                        logger.info(f"Created visualization: {visualization_result.chart_type} chart")
                        
                        # Add the visualization description to the message
                        base_assistant_message["content"] += f"\n\n{visualization_result.description}"
                except Exception as e:
                    logger.error(f"Error in visualization agent during SQL modification: {str(e)}", exc_info=True)
                    base_assistant_message["content"] += f"\n\nI couldn't create a visualization due to an error: {str(e)}"
            elif response.visualization_requested:
                if sql_results_df is None or sql_results_df.empty:
                    logger.info("Visualization requested but SQL returned empty results")
                    base_assistant_message["content"] += "\n\nI couldn't create a visualization because the query returned no results."

            final_assistant_message_dict = base_assistant_message

        else:
            error_msg = f"Received an unexpected response structure from SQL modification agent. Type: {type(agent_run_result)}"
            logger.error(f"{error_msg}. Content: {agent_run_result}")
            return {"role": "assistant", "content": "Sorry, internal issue processing the query modification results."}

    except Exception as outer_mod_e:
        error_msg = f"An error occurred during SQL modification processing: {str(outer_mod_e)}"
        logger.exception("Error during SQL modification execution:")
        return {"role": "assistant", "content": f"Sorry, I encountered an error modifying the query for the {last_target_db_key} database: {str(outer_mod_e)}"}
    finally:
        if deps:
            logger.info("Cleaning up database connection from run_sql_modification.")
            # Define the async cleanup function *before* calling it
            async def run_cleanup_mod():
                await deps.cleanup()
            try:
                run_async_task(run_cleanup_mod)
            except Exception as cleanup_e:
                logger.error(f"Error during DB cleanup in modification: {str(cleanup_e)}", exc_info=True)
        logger.info(f"run_sql_modification finished. Total duration: {time.time() - start_mod_time:.2f}s")

    return final_assistant_message_dict if final_assistant_message_dict else {"role": "assistant", "content": "Sorry, unexpected error during modification."}

# --- End helper function ---

# Add async continuation for combined confirmation
async def run_post_combined_confirmation():
    db_metadata = st.session_state.get("pending_db_metadata")
    target_db_key = st.session_state.get("pending_target_db_key")
    target_db_path = st.session_state.get("pending_target_db_path")
    user_message = st.session_state.get("pending_user_message")
    confirmed_tables = st.session_state.get("confirmed_tables", [])
    if not all([db_metadata, target_db_key, target_db_path, user_message, confirmed_tables is not None]):
        st.error("Internal error: Missing context to continue processing your request.")
        clear_pending_state()
        return
    full_schema_for_pruning = format_schema_for_selected_tables(db_metadata, target_db_key, confirmed_tables)
    pruned_schema_string = full_schema_for_pruning  # (You can call the pruning agent here if needed)
    result_dict = await run_agents_post_confirmation_inner(
        db_metadata=db_metadata,
        selected_tables=confirmed_tables,
        target_db_key=target_db_key,
        target_db_path=target_db_path,
        user_message=user_message,
        agent_message_history=[]
    )
    st.session_state.chat_history.append(result_dict)
    clear_pending_state()
    # return final_assistant_message_dict # This line causes NameError and is unnecessary


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
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

# --- Apply nest_asyncio EARLY and GLOBALLY ---
# This patches asyncio to allow nested loops, often needed with Streamlit
nest_asyncio.apply()

# --- Set up logging ---
# Increased log level for less noise in production, but keep INFO for key events
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
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
    sql_result: Optional[SQLQueryResult] = Field(None, description="SQL query details if SQL was generated")
    python_result: Optional[PythonCodeResult] = Field(None, description="Python code details if Python was generated for visualization/analysis")

class DatabaseClassification(BaseModel):
    """Identifies the target database for a user query."""
    database_key: Literal["IFC", "MIGA", "UNKNOWN"] = Field(..., description="The database key ('IFC', 'MIGA') the user query most likely refers to, based on keywords and the database descriptions provided. Use 'UNKNOWN' if the query is ambiguous, unrelated, or a general greeting/request.")
    reasoning: str = Field(..., description="Brief explanation for the classification (e.g., 'Query mentions IFC investments', 'Query mentions MIGA guarantees', 'Query is ambiguous/general').")


# --- Define the Agent Blueprints (Prompts, Tools, Validators - Keep as is) ---

# System prompt generator function remains global
def generate_system_prompt() -> str:
    """Generates the system prompt for the data analysis agent."""
    # Keep the detailed prompt as provided in the original code
    # ... (prompt content omitted for brevity, assume it's the same) ...
    prompt = f"""You are an expert data analyst assistant. Your role is to help users query and analyze data from a SQLite database.

IMPORTANT: The database schema will be included at the beginning of each user message. Use this schema information to understand the database structure and generate accurate SQL queries. DO NOT respond that you need to know the table structure - it is already provided in the message.

CRITICAL RULES FOR SQL GENERATION:
1. For ANY question about data in the database (counts, totals, listings, comparisons, etc.), you MUST generate an appropriate SQLite query.
2. PAY ATTENTION TO COLUMN NAMES: If a column name in the provided schema contains spaces or special characters, you MUST enclose it in double quotes (e.g., SELECT "Total IFC Investment Amount" FROM ...). Failure to quote such names will cause errors. Check for columns like "IFC investment for Risk Management(Million USD)", "IFC investment for Guarantee(Million USD)", etc.
3. AGGREGATIONS: For questions asking about totals, sums, or aggregations, use SQL aggregate functions (SUM, COUNT, AVG, etc.).
4. GROUPING: When a question mentions "per" some field (e.g., "per product line"), this requires a GROUP BY clause for that field.
5. SUM FOR TOTALS: Numerical fields asking for totals must use SUM() in your query. Ensure you select the correct column (e.g., "IFC investment for Loan(Million USD)" for loan sizes).
6. DATA TYPES: Be mindful that many numeric columns might be stored as TEXT (e.g., "(Million USD)" columns). You might need to CAST them to a numeric type (e.g., CAST("IFC investment for Loan(Million USD)" AS REAL)) before performing calculations like AVG or SUM. Handle potential non-numeric values gracefully if possible (e.g., WHERE clause to filter them out before casting, or use `IFNULL(CAST(... AS REAL), 0)`).
7. SECURITY: ONLY generate SELECT queries. NEVER generate SQL statements that modify the database (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.) or could be potentially harmful. These will be blocked by the system for security reasons and will cause errors.

PYTHON CODE FOR DATA PREPARATION (NOT PLOTTING):
- If a user requests analysis or visualization that requires data manipulation *after* the SQL query (e.g., complex calculations, reshaping data, setting index for charts), generate Python code using pandas.
- Assume the SQL results are available in a pandas DataFrame named 'df'.
- The Python code should ONLY perform data manipulation/preparation on the 'df'.
- CRITICAL: DO NOT include any plotting code (e.g., `matplotlib`, `seaborn`, `st.pyplot`) in the Python code block. The final plotting using native Streamlit charts (like `st.bar_chart`) will be handled separately by the application based on your textual explanation and the prepared data.
- If no specific Python data manipulation is needed beyond the SQL query, do not generate a Python code result.

VISUALIZATION REQUESTS:
- When users request charts, graphs, or plots, first generate the necessary SQL query (remembering data type casting if needed for aggregation).
- If the data from SQL needs further processing for the chart (e.g., setting the index, renaming columns), generate Python code as described above to prepare the 'df'.
- In your text response, clearly state the type of chart you recommend (e.g., "bar chart", "line chart", "pie chart", "scatter plot") based on the user's request and the data structure. Use these exact phrases where possible.
- NEVER respond that you cannot create visualizations.

RESPONSE STRUCTURE:
1. First, review the database schema provided in the message.
2. Understand the request: Does it require data retrieval (SQL), potential data preparation (Python), or just a textual answer? Check for keywords indicating calculations (average, total, sum) or comparisons.
3. Generate SQL: If data is needed, generate an accurate SQLite query string following the rules above (quoting names, casting types, using aggregates).
4. Generate Python Data Prep Code (if needed): If data manipulation beyond SQL is required for analysis or the requested chart, generate Python pandas code acting on 'df'.
5. Explain Clearly: Explain the SQL query (including any casting) and any Python data preparation steps. If visualization was requested, explicitly suggest the chart type (e.g., "bar chart", "line chart") in your text message.
6. Format Output: Format your final response using the 'QueryResponse' structure. Include 'text_message', 'sql_result' (if applicable), and 'python_result' (if Python data prep code was generated).
7. **CRUCIAL**: Even if you use internal tools (like `execute_sql`) to find the answer or validate the query during your thought process, the final `QueryResponse` object you return MUST contain the generated SQL query string in the `sql_result` field if the user's request required data retrieval from the database. Do not omit the `sql_result` just because you used a tool internally.
8. Safety: Focus ONLY on SELECT queries. Do not generate SQL/Python that modifies the database.
9. Efficiency: Write efficient SQL queries. Filter data (e.g., using WHERE clause for country or product type) before aggregation.

Remember, your final output must be the structured 'QueryResponse' object containing the text message and the generated SQL/Python strings (if applicable).
"""
    return prompt


# Agent Tool function remains global
async def execute_sql(ctx: RunContext[AgentDependencies], query: str) -> Union[List[Dict], str]:
    """
    Executes a given SQLite SELECT query and returns the results.
    IMPORTANT: Your primary goal is usually to generate the SQL query string for the final 'QueryResponse' structure, not to execute it yourself.
    Only use this tool if you absolutely need to fetch intermediate data during your reasoning process to answer a complex multi-step question.
    **Using this tool DOES NOT replace the requirement to include the SQL query string in the `sql_result` field of the final `QueryResponse` object if the original request required data retrieval.**
    Args:
        query (str): The SQLite SELECT query to execute.
    Returns:
        List[Dict]: A list of dictionaries representing the query results.
        str: An error message if the query fails or is not a SELECT statement.
    """
    if not ctx.deps or not ctx.deps.db_connection:
        logger.warning("execute_sql called with no database connection.")
        return "Error: Database connection is not available."

    # Enhanced safety checks
    query = query.strip()
    forbidden_commands = ['ALTER', 'CREATE', 'DELETE', 'DROP', 'INSERT', 'UPDATE', 'PRAGMA',
                          'ATTACH', 'DETACH', 'VACUUM', 'GRANT', 'REVOKE', 'EXECUTE', 'TRUNCATE']
    normalized_query = ' '.join([
        line for line in query.upper().split('\n')
        if not line.strip().startswith('--')
    ])

    if not normalized_query.startswith("SELECT"):
        logger.warning(f"Attempted non-SELECT query execution blocked: {query[:100]}...")
        return "Error: Only SELECT queries are allowed."

    for cmd in forbidden_commands:
        if f" {cmd} " in f" {normalized_query} " or normalized_query.startswith(f"{cmd} "):
            logger.warning(f"Blocked query containing forbidden command '{cmd}': {query[:100]}...")
            return f"Error: Detected potentially harmful SQL command '{cmd}'. Only SELECT is allowed."

    # Simple check for multiple statements (basic protection)
    if ';' in query.rstrip(';'): # Allow trailing semicolon but block others
        logger.warning(f"Blocked query with multiple statements: {query[:100]}...")
        return "Error: Multiple SQL statements are not allowed."

    try:
        query_to_execute = query.rstrip(';') # Remove trailing semicolon before execution
        logger.info(f"Executing SQL: {query_to_execute[:200]}...")
        cursor = ctx.deps.db_connection.cursor()
        cursor.execute(query_to_execute)
        columns = [description[0] for description in cursor.description] if cursor.description else []
        results = cursor.fetchall()
        logger.info(f"SQL execution successful. Rows returned: {len(results)}")
        return [dict(zip(columns, row)) for row in results]
    except sqlite3.Error as e:
        logger.error(f"SQL execution error: {e}. Query: {query[:200]}...", exc_info=True)
        return f"Error executing SQL query: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during SQL execution: {e}. Query: {query[:200]}...", exc_info=True)
        return f"An unexpected error occurred: {str(e)}"


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
    data_query_keywords = ['total', 'sum', 'average', 'count', 'list', 'show', 'per', 'group', 'compare', 'what is', 'how many', 'which', 'top']
    user_question_marker = "User Request:"
    user_question_index = user_message.find(user_question_marker)
    original_user_question = user_message[user_question_index + len(user_question_marker):].strip() if user_question_index != -1 else user_message

    if not result.sql_result and any(keyword in original_user_question.lower() for keyword in data_query_keywords):
        is_greeting = any(greet in original_user_question.lower()[:15] for greet in ['hello', 'hi ', 'thanks', 'thank you'])
        is_meta_query = any(kw in original_user_question.lower() for kw in ['explain', 'what is', 'how does', 'tell me about', 'can you'])
        ai_gave_error_reason = "invalid request" in result.text_message.lower() or "cannot process" in result.text_message.lower()

        if not is_greeting and not is_meta_query and not ai_gave_error_reason:
            logger.warning(f"SQL result is missing, but keywords suggest it might be needed for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Missing SQL. Response: text='{result.text_message[:50]}...', sql=None")
            raise ModelRetry("The user's question seems to require data retrieval (based on keywords like 'compare', 'average', 'top', 'total', 'list'), but no SQL query was generated. Please generate the appropriate SQL query.")


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
        "tools": [execute_sql], # Pass the tool function
        "result_validator_func": validate_query_result # Pass the validator function
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

async def identify_target_database(user_query: str, metadata: Dict) -> Tuple[Optional[str], str]:
    """
    Identifies which database (IFC or MIGA) the user query is most likely referring to.
    Returns a tuple (database_key, reasoning). Instantiates its own LLM and Agent.
    """
    logger.info(f"Attempting to identify target database for query: {user_query[:50]}...")
    # ... (database description extraction logic remains the same) ...
    if 'databases' not in metadata:
        logger.error("Metadata missing 'databases' key.")
        return None, "Error: 'databases' key missing in metadata configuration."
    descriptions = []
    valid_keys = []
    for key, db_info in metadata['databases'].items():
        desc = db_info.get('description', f'Database {key}')
        descriptions.append(f"- {key}: {desc}")
        valid_keys.append(key)
    if not descriptions:
         logger.error("No databases found in metadata to classify against.")
         return None, "Error: No databases found in metadata to classify against."
    descriptions_str = "\n".join(descriptions)
    valid_keys_str = ", ".join([f"'{k}'" for k in valid_keys]) + ", or 'UNKNOWN'"
    classification_prompt = f"""Given the user query and the descriptions of available databases, identify which database the query is most likely related to.

Available Databases:
{descriptions_str}

User Query: "{user_query}"

Based *only* on the query and the database descriptions, which database key ({valid_keys_str}) is the most relevant target? If the query is ambiguous, unrelated to these specific databases, or a general greeting/request (like 'hello', 'thank you'), classify it as 'UNKNOWN'.
"""
    logger.info("--- Sending classification request to LLM ---")
    logger.debug(f"Prompt:\n{classification_prompt}") # DEBUG level if needed
    logger.info("--------------------------------------------")

    try:
        # --- Instantiate Model and Agent LOCALLY ---
        # Ensure GenAI is configured within this async context
        global google_api_key
        try:
            genai.configure(api_key=google_api_key)
            logger.info("Configured GenAI SDK within identify_target_database.")
        except Exception as config_err:
             logger.error(f"Failed to configure GenAI SDK for classification: {config_err}", exc_info=True)
             return None, f"Internal Error: Failed to configure AI Service ({config_err})."

        # Use a potentially faster/cheaper model for classification
        gemini_model_name = st.secrets.get("GEMINI_CLASSIFICATION_MODEL", "gemini-1.5-flash")
        local_llm = GeminiModel(model_name=gemini_model_name)
        logger.info(f"Instantiated GeminiModel: {gemini_model_name} for classification.")

        classifier_agent = Agent(
            local_llm,
            result_type=DatabaseClassification,
            name="Database Classifier",
            retries=1, # Fewer retries for classification
            system_prompt="You are an AI assistant that classifies user queries based on provided database descriptions. Output ONLY the structured classification result."
        )
        logger.info("Classifier agent created locally.")
        # --- End Instantiation ---

        # Run the agent with error handling for assert_never error
        try:
            logger.info("Running database classifier agent")
            classification_result = await classifier_agent.run(classification_prompt)
            logger.info("Classification agent run completed.")
        except Exception as e:
            logger.error(f"Classification agent.run() failed: {str(e)}", exc_info=True)
            if "Expected code to be unreachable" in str(e):
                logger.warning("Caught assert_never error in pydantic_ai during database classification.")
                # Return a default classification as this is not a critical error
                # We'll use UNKNOWN which will trigger fallback to last_db_key if available
                return None, f"Classification engine error: {str(e)}"
            else:
                # Re-raise if it's not the expected error
                raise

        # ... (result processing logic remains the same) ...
        if hasattr(classification_result, 'data') and isinstance(classification_result.data, DatabaseClassification):
            result_data: DatabaseClassification = classification_result.data
            logger.info("--- LLM Classification Result ---")
            logger.info(f"Key: {result_data.database_key}")
            logger.info(f"Reasoning: {result_data.reasoning}")
            logger.info("-------------------------------")
            if result_data.database_key == "UNKNOWN":
                # Try to get last key if available
                last_key = st.session_state.get('last_db_key')
                if last_key:
                    logger.warning(f"LLM classified as UNKNOWN, reusing last key: {last_key}. Original Reasoning: {result_data.reasoning}")
                    return last_key, f"(Used last: {last_key}) LLM Reason: {result_data.reasoning}"
                else:
                    return None, f"Could not determine the target database. Reasoning: {result_data.reasoning}"
            elif result_data.database_key in valid_keys:
                return result_data.database_key, result_data.reasoning
            else:
                 logger.warning(f"LLM returned an invalid key: {result_data.database_key}")
                 return None, f"Classification returned an unexpected key '{result_data.database_key}'. Reasoning: {result_data.reasoning}"
        else:
             logger.error(f"Classification call returned unexpected structure: {classification_result}")
             return None, "Error: Failed to get a valid classification structure from the AI."

    except Exception as e:
        logger.exception("Error during database classification LLM call:")
        return None, f"Error during database classification: {str(e)}"


async def handle_user_message(message: str) -> None:
    """
    Handles user input: identifies DB, selects tables (pauses for confirmation),
    and prepares for the main agent run.
    """
    start_time = time.time()
    logger.info(f"handle_user_message started for message: '{message[:50]}...'")

    # --- Check for follow-up charting request ---
    # ... (logic remains the same) ...
    follow_up_chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise']
    follow_up_pronouns = ['this', 'that', 'it']
    message_lower = message.lower()
    is_short_message = len(message.split()) <= 5
    is_follow_up_chart_request = (
        is_short_message and
        any(pronoun in message_lower for pronoun in follow_up_pronouns) and
        any(keyword in message_lower for keyword in follow_up_chart_keywords)
    )

    if is_follow_up_chart_request and st.session_state.get('last_chartable_data') is not None:
        logger.info("Detected follow-up charting request with available data.")
        await handle_follow_up_chart(message) # Assuming this is async now
        logger.info(f"handle_user_message finished early for follow-up chart. Duration: {time.time() - start_time:.2f}s")
        return

    # --- Initialize variables ---
    assistant_chat_message = None
    target_db_key = None

    try:
        # --- Load Metadata (Step 1) ---
        logger.info("Step 1: Loading database metadata...")
        db_metadata = load_db_metadata()
        if not db_metadata:
            logger.error("Metadata loading failed. Cannot proceed.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Sorry, I couldn't load the database configuration. Please check the application setup."
            })
            return
        logger.info("Step 1: Database metadata loaded successfully.")

        # --- Identify Target Database (Step 2) ---
        logger.info("Step 2: Identifying target database...")
        identified_key, reasoning = await identify_target_database(message, db_metadata)
        logger.info(f"Step 2: Database identification result - Key: {identified_key}, Reasoning: {reasoning}")

        if not identified_key:
            logger.warning(f"Database identification failed/UNKNOWN. Cannot proceed without target DB. Reasoning: {reasoning}")
            assistant_chat_message = {
                "role": "assistant",
                "content": f"I'm not sure which database to use (IFC or MIGA). Could you please specify? (Reasoning: {reasoning})"
            }
            st.session_state.chat_history.append(assistant_chat_message)
            return # Stop processing here
        else:
            target_db_key = identified_key

        logger.info(f"Step 2: Target database confirmed as: {target_db_key}")
        st.session_state.last_db_key = target_db_key # Store for context and potential reuse

        # --- Get Specific DB Path (Step 3 - Path only, connect later) ---
        logger.info(f"Step 3: Getting database path for key: {target_db_key}...")
        db_entry = db_metadata.get('databases', {}).get(target_db_key)
        if not db_entry or 'database_path' not in db_entry:
            error_msg = f"Metadata configuration error: Could not find path for database '{target_db_key}'."
            logger.error(error_msg)
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, internal configuration error for database {target_db_key}."}
            st.session_state.chat_history.append(assistant_chat_message)
            return # Stop processing here

        target_db_path = db_entry['database_path'] # Path stored, connection made later
        logger.info(f"Step 3: Database path found: {target_db_path}")

        # --- Table Selection Agent (Step 4) ---
        logger.info(f"Step 4: Running Table Selection Agent for DB: {target_db_key}...")
        table_descriptions = get_table_descriptions_for_db(db_metadata, target_db_key)
        if table_descriptions.startswith("Error:") or table_descriptions.startswith("No tables found"):
            logger.error(f"Could not get table descriptions for TableAgent: {table_descriptions}")
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, couldn't retrieve table list for the {target_db_key} database."}
            st.session_state.chat_history.append(assistant_chat_message)
            return # Stop if table list unavailable

        table_agent_prompt = f"""User Query: "{message}"

Database: {target_db_key}
{table_descriptions}

Based *only* on the user query and the table descriptions, which of the listed table names (keys like "investment_services_projects") are required? Output just the list and reasoning.
"""
        logger.info("Step 4: Calling Table Selection Agent...")
        try:
            # --- Instantiate Model and Agent LOCALLY ---
            logger.info("Instantiating LLM/Agent for table selection...")
            global google_api_key
            try:
                genai.configure(api_key=google_api_key)
                logger.info("Configured GenAI SDK within table selection.")
            except Exception as config_err:
                 logger.error(f"Failed to configure GenAI SDK for table selection: {config_err}", exc_info=True)
                 # Use raise instead of return to propagate the error
                 raise RuntimeError(f"Internal Error: Failed to configure AI Service ({config_err}).") from config_err

            gemini_model_name = st.secrets.get("GEMINI_CLASSIFICATION_MODEL", "gemini-1.5-flash") # Reuse faster model
            local_llm = GeminiModel(model_name=gemini_model_name)
            logger.info(f"Instantiated GeminiModel: {gemini_model_name} for table selection.")

            # Get blueprint and create agent
            agent_config = create_table_selection_agent_blueprint()
            agent_instance = Agent(local_llm, **agent_config)
            logger.info("Table selection agent created locally.")
            # --- End Instantiation ---

            # FIX: Filter message history for table selection agent
            filtered_history = []
            if 'agent_message_history' in st.session_state and st.session_state.agent_message_history:
                history_for_agent = st.session_state.agent_message_history
                logger.info(f"Filtering table selection message history (length: {len(history_for_agent)}) for compatibility")
                for msg in history_for_agent:
                    # Only include messages with valid role and content fields
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        filtered_role = msg['role']
                        if filtered_role in ['user', 'assistant', 'system']:
                            filtered_history.append({
                                'role': filtered_role,
                                'content': str(msg['content'])  # Ensure content is string
                            })
                        else:
                            logger.warning(f"Skipping message with invalid role: {filtered_role}")
                    else:
                        logger.warning(f"Skipping invalid message format: {type(msg)}")
                
                logger.info(f"Filtered message history from {len(history_for_agent)} to {len(filtered_history)} valid entries")
            
            # Use filtered history or skip history parameter if empty
            message_history_param = filtered_history if filtered_history else None

            # Run agent with error handling for assert_never error
            try:
                logger.info(f"Running table selection agent with {len(filtered_history) if filtered_history else 'no'} history messages")
                table_agent_result = await agent_instance.run(
                    table_agent_prompt,
                    message_history=message_history_param
                )
                logger.info("Table selection agent run completed.")
            except Exception as e:
                logger.error(f"Table selection agent.run() failed: {str(e)}", exc_info=True)
                if "Expected code to be unreachable" in str(e):
                    logger.warning("Caught assert_never error in pydantic_ai. Retrying without message history.")
                    # Fall back to running without message history
                    table_agent_result = await agent_instance.run(table_agent_prompt)
                    logger.info("Table selection agent retry (without history) completed.")
                else:
                    # Re-raise if it's not the expected error
                    raise

            if hasattr(table_agent_result, 'data') and isinstance(table_agent_result.data, SuggestedTables):
                selected_tables = table_agent_result.data.table_names
                table_agent_reasoning = table_agent_result.data.reasoning
                logger.info(f"Step 4: Table Agent suggested tables: {selected_tables}. Reasoning: {table_agent_reasoning}")

                if not selected_tables:
                    logger.warning("Table Selection Agent returned an empty list of tables.")
                    # Ask user to clarify instead of erroring out completely
                    assistant_chat_message = {"role": "assistant", "content": f"I couldn't identify specific tables for your query in the {target_db_key} database based on the request: '{message[:100]}...'. Could you perhaps rephrase or be more specific about the data you need (e.g., mention 'investments', 'projects', 'guarantees')?"}
                    st.session_state.chat_history.append(assistant_chat_message)
                    return # Stop

                # Get all available tables for this DB
                db_entry = db_metadata.get('databases', {}).get(target_db_key)
                all_tables = list(db_entry.get("tables", {}).keys()) if db_entry else []

                # --- Store state and PAUSE for user confirmation ---
                st.session_state.table_confirmation_pending = True
                st.session_state.candidate_tables = selected_tables
                st.session_state.all_tables = all_tables
                st.session_state.table_agent_reasoning = table_agent_reasoning
                st.session_state.pending_user_message = message
                st.session_state.pending_db_metadata = db_metadata # Pass metadata
                st.session_state.pending_target_db_key = target_db_key
                st.session_state.pending_target_db_path = target_db_path # Store path too
                logger.info("Step 4: Pausing for table confirmation. State saved.")
                # IMPORTANT: Do NOT add an assistant message here.
                # The UI will render the confirmation prompt in the next rerun.
                return  # Exit this function, wait for user confirmation

            else:
                logger.error(f"Table Selection Agent returned unexpected structure: {table_agent_result}")
                raise ValueError("Table Selection Agent did not return the expected SuggestedTables structure.")

        except Exception as e:
            logger.exception("Error running Table Selection Agent:")
            # Show error to user
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, I encountered an error while trying to determine the necessary tables for the {target_db_key} database: {str(e)}"}
            st.session_state.chat_history.append(assistant_chat_message)
            return # Stop

    except Exception as e:
        # Catch-all for errors during the initial message processing setup
        error_msg = f"A critical error occurred during initial message processing: {str(e)}"
        logger.exception("Critical error in handle_user_message setup:")
        # Ensure assistant_chat_message is set if an error occurs early
        if not assistant_chat_message:
             assistant_chat_message = {"role": "assistant", "content": f"Sorry, a critical internal error occurred: {str(e)}"}
        st.session_state.chat_history.append(assistant_chat_message)

    finally:
        # No cleanup needed here as DB connection happens post-confirmation
        logger.info(f"handle_user_message finished initial stage (or error). Duration: {time.time() - start_time:.2f}s")
        # --- End of handle_user_message ---


async def handle_follow_up_chart(message: str):
    """Handles a follow-up request to chart the last data. (Async placeholder if needed)"""
    # This function might not need to be async if it only manipulates dataframes
    # and appends to Streamlit state, but keep it async for consistency.
    logger.info("Handling follow-up chart request.")
    df_to_chart = st.session_state.last_chartable_data
    db_key = st.session_state.get('last_chartable_db_key', 'Unknown DB')
    message_lower = message.lower()

    # Determine chart type
    chart_type = None
    # ... (chart type detection logic remains the same) ...
    if "bar" in message_lower: chart_type = "bar"
    elif "line" in message_lower: chart_type = "line"
    elif "area" in message_lower: chart_type = "area"
    elif "scatter" in message_lower: chart_type = "scatter"
    elif "pie" in message_lower: chart_type = "pie"

    if not chart_type:
        if len(df_to_chart.columns) > 1:
            chart_type = "bar" # Default
            logger.info("Follow-up chart type not specified, defaulting to bar chart.")
            st.toast("Chart type not specified, defaulting to bar chart.", icon="💡")
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Please specify the type of chart (e.g., bar, line, pie) you want for the previous data from the {db_key} database."
            })
            logger.warning("Follow-up chart type not specified, cannot default chart type for single column.")
            return

    # --- Prepare DataFrame for Streamlit Charting ---
    df_display = df_to_chart # Start with the original
    try:
        # ... (DataFrame preparation logic remains the same) ...
        if chart_type != "pie" and df_to_chart.index.name is None and len(df_to_chart.columns) > 1:
            potential_index_col = df_to_chart.columns[0]
            col_dtype = df_to_chart[potential_index_col].dtype
            if pd.api.types.is_string_dtype(col_dtype) or \
               pd.api.types.is_categorical_dtype(col_dtype) or \
               pd.api.types.is_datetime64_any_dtype(col_dtype):
                logger.info(f"Attempting to automatically set DataFrame index to '{potential_index_col}' for follow-up charting.")
                df_display = df_to_chart.copy().set_index(potential_index_col)
                logger.info("Index set successfully for display.")
            else:
                logger.info(f"First column '{potential_index_col}' (type: {col_dtype}) not suitable for index, using original DataFrame for chart.")
                df_display = df_to_chart
        else:
            logger.info("Using original DataFrame for chart (index exists, single col, or pie).")
            df_display = df_to_chart
    except Exception as e:
        logger.warning(f"Error preparing DataFrame for follow-up chart: {e}. Using original DataFrame.", exc_info=True)
        df_display = df_to_chart

    # Construct assistant message with chart
    assistant_chat_message = {
        "role": "assistant",
        "content": f"Okay, here's the {chart_type} chart for the previous data from the {db_key} database:",
        "streamlit_chart": {
            "type": chart_type,
            "data": df_display # Use the potentially indexed dataframe
        }
    }
    st.session_state.chat_history.append(assistant_chat_message)
    logger.info("Appended follow-up chart message to history.")


async def run_agents_post_confirmation_inner(
    db_metadata: Dict,
    target_db_key: str,
    target_db_path: str,
    selected_tables: List[str],
    user_message: str,
    agent_message_history: List[Dict] # Pass history explicitly
) -> Union[Dict, str]:
    """
    Inner async function to run the main query agent AFTER table confirmation.
    Handles DB connection, agent instantiation, execution, and result processing.
    Returns a message dictionary on success, or an error string on failure.
    """
    deps = None
    final_assistant_message_dict = None
    start_inner_time = time.time()
    logger.info("run_agents_post_confirmation_inner started.")

    try:
        # --- Instantiate Model and Query Agent Locally --- #
        logger.info("Instantiating LLM/Agent for post-confirmation query...")
        global google_api_key
        try:
            genai.configure(api_key=google_api_key)
            logger.info("Configured GenAI SDK within post-confirmation flow.")
        except Exception as config_err:
             logger.error(f"Failed to configure GenAI SDK post-confirmation: {config_err}", exc_info=True)
             return f"Internal Error: Failed to configure AI Service ({config_err})." # Return error string

        # Use a more capable model for the main query generation
        gemini_model_name = st.secrets.get("GEMINI_MAIN_MODEL", "gemini-2.0-flash")
        local_llm = GeminiModel(model_name=gemini_model_name)
        logger.info(f"Instantiated GeminiModel: {gemini_model_name} for post-confirmation flow.")

        # Get blueprint and create agent
        agent_config_bp = create_query_agent_blueprint()
        # We need to instantiate the agent properly with decorators/tools
        local_query_agent = Agent(
            local_llm,
            deps_type=agent_config_bp["deps_type"],
            result_type=agent_config_bp["result_type"],
            name=agent_config_bp["name"],
            retries=agent_config_bp["retries"],
        )
        # Manually apply decorators from blueprint info
        local_query_agent.system_prompt(agent_config_bp["system_prompt_func"])
        for tool_func in agent_config_bp["tools"]:
            local_query_agent.tool(tool_func)
        local_query_agent.result_validator(agent_config_bp["result_validator_func"])

        logger.info("Query agent created locally for post-confirmation flow.")
        # --- End Instantiation ---

        # --- Format Schema for SELECTED tables --- #
        logger.info(f"Formatting schema for CONFIRMED tables: {selected_tables}...")
        schema_to_use = format_schema_for_selected_tables(db_metadata, target_db_key, selected_tables)

        if schema_to_use.startswith("Error:"):
            logger.error(f"Could not get valid schemas for selected tables: {schema_to_use}")
            return f"Sorry, couldn't retrieve valid schema details for the selected tables ({', '.join(selected_tables)}) in the {target_db_key} database: {schema_to_use}"

        # --- Connect to DB --- #
        logger.info(f"Connecting to database: {target_db_path} for key: {target_db_key}")
        deps = AgentDependencies.create().with_db(db_path=target_db_path)
        if not deps.db_connection:
            return f"Sorry, I couldn't connect to the {target_db_key} database at {target_db_path}."
        logger.info("Database connection successful.")

        # --- Prepare and Run Main Query Agent --- #
        logger.info("Preparing and running Main Query Agent (within post-confirmation flow)...")
        usage = Usage() # Initialize usage tracking for this run

        # Construct the prompt message
        prompt_message = f"""Target Database: {target_db_key}
Database Schema (Full schema for selected tables):
{schema_to_use}

User Request: {user_message}"""

        # Add visualization hint if needed
        # ... (visualization hint logic remains the same) ...
        visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation', 'bar chart', 'pie chart', 'histogram', 'line graph', 'scatter plot']
        is_visualization_request = any(keyword in user_message.lower() for keyword in visualization_keywords)
        if is_visualization_request:
            logger.info("Adding visualization instructions to prompt.")
            prompt_message += f"""

IMPORTANT: This is a visualization request for the {target_db_key} database.
1. Generate the appropriate SQL query to retrieve the necessary data from the provided schema (remember to CAST text numbers if needed for aggregation).
2. In your text response, you MUST suggest an appropriate chart type (e.g., "bar chart", "line chart", "pie chart") based on the user's request and the data.
3. Do NOT generate Python code for plotting (e.g., using matplotlib or seaborn). Only generate Python code if data *preparation* (like setting index, renaming) beyond the SQL query is needed for the suggested chart.
"""
        else:
            logger.info("Standard (non-visualization) request.")

        logger.info("==== AI CALL (Query Agent, post-confirmation) ====")
        logger.debug(f"Sending prompt message to AI:\n{prompt_message}") # Debug level
        logger.info("==============================")

        # FIX: Filter and validate message history to prevent assert_never error
        filtered_history = []
        if agent_message_history:
            logger.info(f"Filtering message history (length: {len(agent_message_history)}) for compatibility")
            for msg in agent_message_history:
                # Only include messages with valid role and content fields
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    filtered_role = msg['role']
                    if filtered_role in ['user', 'assistant', 'system']:
                        filtered_history.append({
                            'role': filtered_role,
                            'content': str(msg['content'])  # Ensure content is string
                        })
                    else:
                        logger.warning(f"Skipping message with invalid role: {filtered_role}")
                else:
                    logger.warning(f"Skipping invalid message format: {type(msg)}")
            
            logger.info(f"Filtered message history from {len(agent_message_history)} to {len(filtered_history)} valid entries")
        
        # Use filtered history or skip history parameter if empty
        message_history_param = filtered_history if filtered_history else None
        
        # Run query agent
        try:
            logger.info(f"Running agent with {len(filtered_history) if filtered_history else 'no'} history messages")
            agent_run_result = await local_query_agent.run(
                prompt_message,
                deps=deps,
                usage=usage,
                usage_limits=DEFAULT_USAGE_LIMITS,
                message_history=message_history_param
            )
            run_duration = time.time() - start_inner_time
            logger.info(f"Query Agent call completed post-confirmation. Duration: {run_duration:.2f}s. Result type: {type(agent_run_result)}")
        except Exception as e:
            logger.error(f"Agent.run() failed: {str(e)}", exc_info=True)
            if "Expected code to be unreachable" in str(e):
                logger.warning("Caught assert_never error in pydantic_ai. Retrying without message history.")
                # Fall back to running without message history
                agent_run_result = await local_query_agent.run(
                    prompt_message,
                    deps=deps,
                    usage=usage,
                    usage_limits=DEFAULT_USAGE_LIMITS
                    # No message_history parameter
                )
                run_duration = time.time() - start_inner_time
                logger.info(f"Query Agent retry (without history) completed. Duration: {run_duration:.2f}s. Result type: {type(agent_run_result)}")
            else:
                # Re-raise if it's not the expected error
                raise

        # Log token usage
        # ... (usage logging logic remains the same) ...
        try:
            if hasattr(agent_run_result, 'usage'):
                usage_data = agent_run_result.usage() if callable(agent_run_result.usage) else agent_run_result.usage
                if usage_data and hasattr(usage_data, 'prompt_tokens'): # Check attributes exist
                    prompt_tokens = getattr(usage_data, 'prompt_tokens', 'N/A')
                    completion_tokens = getattr(usage_data, 'completion_tokens', 'N/A')
                    total_tokens = getattr(usage_data, 'total_tokens', 'N/A')
                    logger.info(f"Token Usage (this call): Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}")
                    # Update cumulative usage if needed (though maybe less critical now)
                    # st.session_state.cumulative_usage += usage_data # Assuming cumulative_usage state exists
                else:
                    logger.warning("Could not log token usage: 'Usage' object has no attribute 'prompt_tokens' or similar.")
            else:
                logger.info("Token Usage information not available in agent result.")
        except Exception as usage_err:
            logger.warning(f"Could not log token usage: {usage_err}", exc_info=False)


        st.session_state.last_result = agent_run_result # Store for potential debug

        # Append new messages to cumulative history for future calls
        if hasattr(agent_run_result, 'new_messages'):
            new_msgs = agent_run_result.new_messages()
            # Filter new messages before adding to history
            valid_new_msgs = []
            for msg in new_msgs:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    if msg['role'] in ['user', 'assistant', 'system']:
                        valid_new_msgs.append({
                            'role': msg['role'],
                            'content': str(msg['content'])
                        })
            
            st.session_state.agent_message_history.extend(valid_new_msgs)
            logger.info(f"Appended {len(valid_new_msgs)} validated new messages to agent_message_history (new total: {len(st.session_state.agent_message_history)}).")
        else:
             logger.warning("Agent result object does not have 'new_messages' attribute.")

        # --- Process Query Agent Response --- #
        logger.info("Processing Query Agent response (after confirmation)...")
        if hasattr(agent_run_result, 'data') and isinstance(agent_run_result.data, QueryResponse):
            response: QueryResponse = agent_run_result.data
            logger.info("Agent response has expected QueryResponse structure.")
            logger.debug(f"AI Response Data: {response}")

            # Construct the base assistant message
            base_assistant_message = {"role": "assistant", "content": f"[{target_db_key}] {response.text_message}"}
            logger.info(f"Base assistant message: {base_assistant_message['content'][:100]}...")

            sql_results_df = None # Initialize DataFrame variable
            sql_info = {} # Initialize SQL info dict

            # --- Handle SQL Result ---
            if response.sql_result:
                sql_query = response.sql_result.sql_query
                logger.info(f"SQL query generated: {sql_query[:100]}...")
                logger.info(f"Executing SQL query against {target_db_key}...")

                # Create a minimal context for the execute_sql tool - IMPORTANT: Pass model instance
                sql_run_context = RunContext(
                    deps=deps, model=local_llm, usage=usage, prompt=sql_query
                )
                sql_execution_result = await execute_sql(sql_run_context, sql_query)

                sql_info = {
                    "query": sql_query,
                    "explanation": response.sql_result.explanation
                }
                if isinstance(sql_execution_result, str): # Error occurred
                    sql_info["error"] = sql_execution_result
                    logger.error(f"SQL execution failed: {sql_execution_result}")
                    base_assistant_message["content"] += f"\n\n**Warning:** There was an error executing the SQL query: `{sql_execution_result}`"
                    sql_results_df = pd.DataFrame() # Empty DF on error
                elif isinstance(sql_execution_result, list):
                    if sql_execution_result:
                        logger.info(f"SQL execution successful, {len(sql_execution_result)} rows returned.")
                        try:
                            sql_results_df = pd.DataFrame(sql_execution_result)
                            # Try converting numeric-like text columns (like currency)
                            for col in sql_results_df.columns:
                                if isinstance(sql_results_df[col].iloc[0], str):
                                     # Basic check for patterns like "123.45" or "-12.3"
                                    if sql_results_df[col].str.match(r'^-?\d+(\.\d+)?$').all():
                                        try:
                                            sql_results_df[col] = pd.to_numeric(sql_results_df[col])
                                            logger.info(f"Converted column '{col}' to numeric.")
                                        except ValueError:
                                            logger.warning(f"Could not convert column '{col}' to numeric despite pattern match.")
                            sql_info["results"] = sql_results_df.to_dict('records') # Store results for display
                            sql_info["columns"] = list(sql_results_df.columns)
                            st.session_state.last_chartable_data = sql_results_df # Store for potential follow-up
                            st.session_state.last_chartable_db_key = target_db_key
                            logger.info(f"Stored chartable data (shape: {sql_results_df.shape}) for DB '{target_db_key}'.")
                        except Exception as df_e:
                            logger.error(f"Error creating/processing DataFrame from SQL results: {df_e}", exc_info=True)
                            sql_info["error"] = f"Error processing SQL results into DataFrame: {df_e}"
                            sql_results_df = pd.DataFrame()
                    else:
                        logger.info("SQL execution successful, 0 rows returned.")
                        sql_info["results"] = [] # Empty results
                        sql_results_df = pd.DataFrame() # Ensure df is an empty DataFrame
                        st.session_state.last_chartable_data = None # Clear chartable if no results
                        st.session_state.last_chartable_db_key = None
                else:
                     sql_info["error"] = "Unexpected result type from SQL execution."
                     logger.error(f"{sql_info['error']} Type: {type(sql_execution_result)}")
                     sql_results_df = pd.DataFrame()

                base_assistant_message["sql_result"] = sql_info
                logger.info("Added sql_result block to assistant message.")
            else:
                logger.info("No SQL query was generated by the agent.")
                st.session_state.last_chartable_data = None # Clear chartable if no SQL
                st.session_state.last_chartable_db_key = None

            # Initialize df_for_chart with SQL results (or empty if failed/no results)
            df_for_chart = sql_results_df if sql_results_df is not None else pd.DataFrame()
            python_info = {} # Initialize python info dict

            # --- Handle Python Result ---
            if response.python_result:
                logger.info("Python code generated for data preparation...")
                python_code = response.python_result.python_code
                python_info = {
                    "code": python_code,
                    "explanation": response.python_result.explanation
                }
                logger.info(f"Python code to execute:\n{python_code}")

                # Prepare local variables for exec
                local_vars = {
                    'pd': pd,
                    'np': np,
                    'df': df_for_chart.copy() # Pass a copy of the SQL results DF
                }

                # --- Execute Python Code Safely ---
                # Avoid using 'st' or other potentially harmful modules inside exec
                try:
                    # Use restricted globals dictionary for safety
                    exec(python_code, {"pd": pd, "np": np}, local_vars)
                    if 'df' in local_vars and isinstance(local_vars['df'], pd.DataFrame):
                        df_for_chart = local_vars['df'] # Update df_for_chart with result
                        logger.info(f"Python data preparation executed. DataFrame shape after exec: {df_for_chart.shape}")
                        st.session_state.last_chartable_data = df_for_chart # Update chartable data
                        logger.info("Updated last_chartable_data with DataFrame from Python exec.")
                    else:
                        logger.warning(f"'df' variable after Python code is missing or not a DataFrame. Type: {type(local_vars.get('df'))}")
                        python_info["warning"] = "Python code did not produce or update a DataFrame named 'df'."

                except Exception as e:
                    logger.error(f"Error executing Python data preparation code: {e}\nCode:\n{python_code}", exc_info=True)
                    python_info["error"] = str(e)
                    base_assistant_message["content"] += f"\n\n**Warning:** There was an error executing the provided Python code: `{e}`"

                base_assistant_message["python_result"] = python_info
                logger.info("Added python_result block to assistant message.")


            # --- Determine Chart Type and Prepare Chart Data ---
            chart_type = None
            if df_for_chart is not None and not df_for_chart.empty:
                text_lower = response.text_message.lower()
                # ... (chart type detection logic remains the same) ...
                if "bar chart" in text_lower: chart_type = "bar"
                elif "line chart" in text_lower: chart_type = "line"
                elif "area chart" in text_lower: chart_type = "area"
                elif "scatter plot" in text_lower or "scatter chart" in text_lower: chart_type = "scatter"
                elif "pie chart" in text_lower: chart_type = "pie"

                if chart_type:
                    logger.info(f"Detected chart type suggestion: {chart_type}")
                    df_display_chart = df_for_chart # Start with potentially Python-modified DF
                    try:
                        # ... (DataFrame preparation for display remains the same) ...
                        if chart_type != "pie" and df_display_chart.index.name is None and len(df_display_chart.columns) > 1:
                            potential_index_col = df_display_chart.columns[0]
                            col_dtype = df_display_chart[potential_index_col].dtype
                            if pd.api.types.is_string_dtype(col_dtype) or \
                               pd.api.types.is_categorical_dtype(col_dtype) or \
                               pd.api.types.is_datetime64_any_dtype(col_dtype):
                                logger.info(f"Attempting to automatically set DataFrame index to '{potential_index_col}' for charting.")
                                df_display_chart = df_display_chart.copy().set_index(potential_index_col) # Use copy
                                logger.info("Index set successfully for chart display.")
                            else:
                                logger.info(f"First column '{potential_index_col}' (type: {col_dtype}) not suitable for index, using original DataFrame for chart.")
                        else:
                             logger.info("Using original DataFrame for chart (index exists, single col, or pie).")

                        # Store chart type and data for display function
                        base_assistant_message["streamlit_chart"] = {
                            "type": chart_type,
                            "data": df_display_chart # Store the potentially modified/indexed dataframe
                        }
                        logger.info(f"Added streamlit_chart block (type: {chart_type}) to assistant message.")

                    except Exception as chart_prep_e:
                        logger.warning(f"Could not prepare DataFrame for charting: {chart_prep_e}. Chart might not display correctly.", exc_info=True)
                        base_assistant_message["streamlit_chart"] = {
                            "type": chart_type,
                            "data": df_for_chart # Fallback to pre-indexed data
                        }
                        base_assistant_message["content"] += f"\n\n**Note:** Could not automatically prepare data for {chart_type} chart: `{chart_prep_e}`"

                else:
                    logger.info("No specific chart type suggestion detected in AI response text.")
            else:
                 logger.info("DataFrame is empty or None, skipping chart generation check.")

            final_assistant_message_dict = base_assistant_message # Store the complete message dict
            logger.info("Query Agent response processing complete (post-confirmation).")

        else:
             # Handle case where agent result is not the expected QueryResponse
             error_msg = f"Received an unexpected response structure from main Query Agent. Type: {type(agent_run_result)}"
             logger.error(f"{error_msg}. Content: {agent_run_result}")
             return f"Sorry, internal issue processing the {target_db_key} database query results. Unexpected AI response format." # Return error string

    # --- Catch specific exceptions like ModelRetry ---
    except ModelRetry as mr:
        error_msg = f"Query validation failed after {mr.retries} retries: {str(mr)}"
        logger.error(error_msg, exc_info=False) # Don't need full traceback for ModelRetry
        return f"Sorry, I encountered an issue generating the response for {target_db_key} after several attempts: {str(mr)}" # Return error string

    # --- Catch broader exceptions during agent run/processing ---
    except Exception as agent_e:
        error_msg = f"An error occurred during main query agent processing: {str(agent_e)}"
        logger.exception("Error during query agent execution or response processing (post-confirmation):")
        return f"Sorry, I encountered an error generating the response for the {target_db_key} database: {str(agent_e)}" # Return error string

    # --- Catch exceptions during the setup (DB connection, schema formatting etc.) ---
    except Exception as setup_e:
        error_msg = f"A critical error occurred during post-confirmation processing setup: {str(setup_e)}"
        logger.exception("Critical error in run_agents_post_confirmation_inner setup:")
        return f"Sorry, a critical error occurred before processing your request: {str(setup_e)}" # Return error string

    finally:
        # --- Ensure Cleanup ---
        if deps:
            logger.info("Cleaning up database connection from run_agents_post_confirmation_inner.")
            await deps.cleanup() # Await the async cleanup

        logger.info(f"run_agents_post_confirmation_inner finished. Total duration: {time.time() - start_inner_time:.2f}s")

    # If we reach here without returning an error string, return the message dictionary
    if final_assistant_message_dict:
        return final_assistant_message_dict
    else:
        # Fallback if something went wrong and no error string was returned but dict is missing
        logger.error("run_agents_post_confirmation_inner finished unexpectedly without a result dictionary or error string.")
        return "Sorry, an unexpected internal error occurred during processing."


async def continue_after_table_confirmation():
    """
    Coordinates the logic flow *after* the user confirms table selection.
    Calls the inner function to run the main query agent.
    Handles appending the result/error message to the chat history.
    """
    start_time = time.time()
    logger.info("continue_after_table_confirmation called.")
    if st.session_state.get("confirmed_tables") is None:
        logger.error("continue_after_table_confirmation called without confirmed_tables in session state.")
        st.error("Internal error: No confirmed tables found to continue processing.")
        clear_pending_state() # Clean up
        return

    # Retrieve necessary info from session state
    db_metadata = st.session_state.get("pending_db_metadata")
    target_db_key = st.session_state.get("pending_target_db_key")
    target_db_path = st.session_state.get("pending_target_db_path")
    message = st.session_state.get("pending_user_message")
    selected_tables = st.session_state.get("confirmed_tables", [])
    agent_history = st.session_state.agent_message_history # Get current agent history

    if not all([db_metadata, target_db_key, target_db_path, message, selected_tables is not None]): # Check selected_tables exists
         logger.error("Missing required data in session state for continue_after_table_confirmation.")
         st.error("Internal error: Missing context to continue processing your request.")
         clear_pending_state()
         return

    logger.info(f"Continuing with DB: {target_db_key}, Path: {target_db_path}, Tables: {selected_tables}, Query: '{message[:50]}...'")

    assistant_chat_message = None # Initialize
    try:
        # Directly await the inner async function which handles the core logic
        result = await run_agents_post_confirmation_inner(
            db_metadata=db_metadata,
            target_db_key=target_db_key,
            target_db_path=target_db_path,
            selected_tables=selected_tables,
            user_message=message,
            agent_message_history=agent_history # Pass history
        )

        # Check the result type from the inner function
        if isinstance(result, dict):
            assistant_chat_message = result # Success, got the message dict
            logger.info("run_agents_post_confirmation_inner completed successfully, obtained message dict.")
        elif isinstance(result, str):
             # An error occurred, inner function returned an error message string
             logger.error(f"run_agents_post_confirmation_inner returned an error message: {result}")
             # Create a standard assistant error message structure
             assistant_chat_message = {"role": "assistant", "content": result}
        else:
             # Should not happen
             logger.error(f"run_agents_post_confirmation_inner returned an unexpected type: {type(result)}")
             assistant_chat_message = {"role": "assistant", "content": "Sorry, an unexpected internal error occurred during processing."}

    except Exception as e:
        # Catch errors from awaiting the inner function itself (less likely now)
        error_msg = f"A critical error occurred running the post-confirmation agents: {str(e)}"
        logger.exception("Critical error in continue_after_table_confirmation while awaiting inner function:")
        if not assistant_chat_message: # Create error message if not already set
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, a critical error occurred: {str(e)}"}

    # --- Append the final message to history --- #
    if assistant_chat_message:
        st.session_state.chat_history.append(assistant_chat_message)
        logger.info(f"Assistant message appended to history in continue_after_table_confirmation. Content: {str(assistant_chat_message.get('content'))[:100]}...")
    else:
        logger.error("continue_after_table_confirmation finished without an assistant message object to append.")
        # Append a generic error if needed, though should be handled above
        if not st.session_state.chat_history or st.session_state.chat_history[-1].get("role") != "assistant":
            st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, an internal error occurred, and no response could be generated."})


    # --- Clean up pending state --- #
    # IMPORTANT: Clear state *after* processing is complete (success or failure)
    clear_pending_state()

    logger.info(f"continue_after_table_confirmation finished. Duration: {time.time() - start_time:.2f}s")


def clear_pending_state():
    """Clears session state variables related to pending table confirmation."""
    keys_to_clear = [
        "table_confirmation_pending", "pending_db_metadata", "pending_target_db_key",
        "pending_target_db_path", "pending_user_message", "confirmed_tables",
        "candidate_tables", "all_tables", "table_agent_reasoning"
    ]
    cleared_count = 0
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            cleared_count += 1
    if cleared_count > 0:
        logger.info(f"Cleaned up {cleared_count} pending session state keys.")


# --- Async Runner Helper ---
# This is crucial for running async code correctly within Streamlit's event loop
def run_async(coro):
    """Runs an async coroutine reliably in the current Streamlit event loop."""
    try:
        # Try to get the loop that Streamlit is currently using
        loop = asyncio.get_running_loop()
        logger.debug("Found running event loop for run_async.")
    except RuntimeError:
        # If no loop is running (less common in recent Streamlit), create one
        logger.warning("No running event loop found, creating a new one for run_async.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Apply nest_asyncio here too if creating a new loop
        nest_asyncio.apply(loop)

    # Ensure nest_asyncio is applied to the loop we are using
    # nest_asyncio.apply(loop) # Already applied globally, might be redundant but safe
    # logger.debug("Applied nest_asyncio within run_async.")

    try:
        # Run the coroutine until it completes within the obtained/created loop
        logger.debug(f"Running coroutine {coro.__name__} using loop.run_until_complete...")
        result = loop.run_until_complete(coro)
        logger.debug(f"Coroutine {coro.__name__} execution completed.")
        return result
    except Exception as e:
         # Log the exception originating from the coroutine
         logger.exception(f"Exception caught by run_async while running {getattr(coro, '__name__', 'coroutine')}: {e}")
         # Re-raise the exception so the caller (e.g., in main) can handle it
         raise e
    # Note: We don't explicitly close the loop here. Streamlit manages its loop.
    # If we created one, it might persist, but run_async will get it next time.


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
    
    return {
        'role': role,
        'content': str(message['content'])  # Ensure content is string
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


# --- Main Streamlit App Function ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="SmartQuery")

    # Initialize session state variables safely
    default_session_state = {
        'chat_history': [],            # For display
        'last_result': None,           # For debug
        'last_db_key': None,           # For context reuse
        'agent_message_history': [],   # Cumulative history for AI context
        'last_chartable_data': None,   # For follow-up charting
        'last_chartable_db_key': None, # For follow-up charting context
        'table_confirmation_pending': False,
        'candidate_tables': [],
        'all_tables': [],
        'table_agent_reasoning': "",
        'pending_user_message': None,
        'pending_db_metadata': None,
        'pending_target_db_key': None,
        'pending_target_db_path': None,
        'confirmed_tables': None,      # Stores the user-confirmed list
    }
    for key, default_value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            # logger.debug(f"Initialized '{key}' in session state.") # Debug level

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
            height: calc(20vh - 250px); /* Adjust height based on surrounding elements */
            overflow: hidden; /* Hide main container overflow */
            margin-top: 0.5rem; /* Reduce top margin from 1rem to 0.5rem */
        }
        /* Make message container scrollable */
        .chat-messages-container {
            flex-grow: 1;
            overflow-y: auto; /* Enable vertical scrolling */
            padding: 0 1rem 0.5rem 1rem; /* Reduce bottom padding from 1rem to 0.5rem */
            margin-bottom: 60px; /* Reduce space for the input box from 70px to 60px */
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
        .stChatMessage { border-radius: 10px; border: 1px solid #eee; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-top: 0.25rem; } /* Added smaller top margin */
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
            <div class="feature-text">{check_img} Ask natural language questions about IFC or MIGA data.</div>
            <div class="feature-text">{check_img} Get instant SQL-powered insights from both databases.</div>
        </div>
        <div class="features-row">
            <div class="feature-text">{check_img} Generate visualizations (bar, line, pie charts) via Python.</div>
            <div class="feature-text">{check_img} System automatically identifies the right database for your query.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info('Query data from the IFC Investment or MIGA Guarantees databases. The AI will identify the appropriate database automatically.', icon=":material/info:")
    # --- Example Queries ---
    st.markdown("""
    <div class="example-queries">
        <p>Example Questions:</p>
        <ul>
            <li>"Visualize the distribution of IFC product lines using a pie chart."</li>
            <li>"Show me MIGA guarantees in the Financial sector from Cambodia."</li>
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
        clear_pending_state()
        logger.info("Chat history and associated states cleared by user.")
        st.rerun()

    # --- ASYNC CONTINUATION HANDLING --- #
    # Check if confirmation was completed in the *previous* run (confirmed_tables is set)
    # and we are *not currently* waiting for confirmation again.
    if st.session_state.get("confirmed_tables") is not None and not st.session_state.get("table_confirmation_pending", False):
        logger.info("Detected confirmed tables from previous run, proceeding with post-confirmation processing.")

        # Prepare the coroutine
        confirmation_coro = continue_after_table_confirmation()

        # Use run_async helper with spinner
        try:
            with st.spinner("Processing confirmed table selection..."):
                logger.info("Calling run_async for continue_after_table_confirmation...")
                run_async(confirmation_coro)
                logger.info("run_async for continue_after_table_confirmation finished.")
            # Rerun AFTER the async operation completes to update the UI with results
            logger.info("Rerunning Streamlit after successful table confirmation processing.")
            st.rerun()
        except Exception as e:
             # Error logged by run_async and continue_after_table_confirmation
             st.error(f"An error occurred while processing your confirmed selection: {str(e)}")
             logger.error("Error occurred during post-confirmation processing or rerun.")
             # Ensure pending state is cleared even if continuation fails
             clear_pending_state()
             # Rerun to show the error message and clear spinner
             st.rerun()

    # --- Chat Interface --- #
    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages-container" id="chat-messages-container-id">', unsafe_allow_html=True)

    chat_display_container = st.container()
    with chat_display_container:
        for i, message in enumerate(st.session_state.chat_history):
            role = message.get("role", "assistant")
            content = message.get("content", "")
            with st.chat_message(role):
                st.markdown(content)
                sql_result = message.get("sql_result")
                if sql_result:
                    st.markdown("**SQL Query:**")
                    st.code(sql_result.get("query", ""), language="sql")
                    if "explanation" in sql_result and sql_result["explanation"]:
                        st.markdown(f"**Explanation:** {sql_result['explanation']}")
                    if "error" in sql_result:
                        st.error(f"SQL Error: {sql_result['error']}")
                    elif "results" in sql_result and sql_result["results"]:
                        st.markdown("**Results:**")
                        try:
                            results_df = pd.DataFrame(sql_result["results"])
                            st.dataframe(results_df)
                        except Exception as e:
                            st.error(f"Error displaying results: {str(e)}")
                python_result = message.get("python_result")
                if python_result:
                    st.markdown("**Python Code:**")
                    st.code(python_result.get("code", ""), language="python")
                    if "explanation" in python_result and python_result["explanation"]:
                        st.markdown(f"**Explanation:** {python_result['explanation']}")
                    if "error" in python_result:
                        st.error(f"Error executing Python code: {python_result['error']}")
                if "streamlit_chart" in message:
                    st.markdown("**Visualization:**")
                    try:
                        chart_type = message["streamlit_chart"]["type"]
                        df = message["streamlit_chart"]["data"]
                        if chart_type == "bar":
                            st.bar_chart(df)
                        elif chart_type == "line":
                            st.line_chart(df)
                        elif chart_type == "area":
                            st.area_chart(df)
                        elif chart_type == "scatter":
                            if len(df.columns) >= 2:
                                st.scatter_chart(df, x=df.columns[0], y=df.columns[1])
                            else:
                                st.warning("Scatter plot requires at least two data columns.")
                        elif chart_type == "pie":
                            if not df.empty and len(df.columns) > 0:
                                fig = px.pie(df, names=df.index, values=df.columns[0], title="Pie Chart")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Cannot generate pie chart: Data is empty or missing columns.")
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
        if st.session_state.get("table_confirmation_pending", False):
            logger.info("Rendering table confirmation UI inside chat message area.")
            candidate_tables = st.session_state.get("candidate_tables", [])
            all_tables = st.session_state.get("all_tables", [])
            reasoning = st.session_state.get("table_agent_reasoning", "")
            db_key = st.session_state.get("pending_target_db_key", "")
            with st.chat_message("assistant"):
                st.info(f"**Table Selection Required:** I suggest using these tables for your query: {', '.join(candidate_tables)}", icon="ℹ️")
                if reasoning:
                    st.caption(f"Reasoning: {reasoning}")
                selected = st.multiselect(
                    f"Confirm or adjust tables for your query in the {db_key} database:",
                    options=all_tables,
                    default=candidate_tables,
                    key="table_confirm_multiselect"
                )
                if st.button("Confirm Table Selection"):
                    logger.info(f"User confirmed table selection: {selected}")
                    st.session_state.table_confirmation_pending = False
                    st.session_state.confirmed_tables = selected
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    if not st.session_state.get("table_confirmation_pending", False):
        user_input = st.chat_input("Ask about IFC or MIGA data...")
        if user_input:
            logger.info(f"User input received in main(): {user_input}")
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            user_msg = validate_message_entry({"role": "user", "content": user_input})
            if user_msg:
                st.session_state.agent_message_history.append(user_msg)
                logger.info("Added validated user message to agent_message_history")
            else:
                logger.warning("Failed to validate user message for agent_message_history, not added")
            message_coro = handle_user_message(user_input)
            try:
                with st.spinner("Analyzing request..."):
                    logger.info("Calling run_async for handle_user_message...")
                    run_async(message_coro)
                    logger.info("run_async for handle_user_message finished.")
                logger.info("Rerunning Streamlit after processing user input / reaching confirmation point.")
                st.rerun()
            except Exception as e:
                 st.error(f"An error occurred while processing your request: {str(e)}")
                 logger.error("Error occurred during initial user input processing or rerun.")
                 if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content") != f"Sorry, a critical internal error occurred: {str(e)}":
                     st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, a critical internal error occurred: {str(e)}"})
                 st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    components.html(
        f"""
        <script>
            function scrollToBottom() {{
                const chatContainer = document.getElementById('chat-messages-container-id');
                if (chatContainer) {{
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }}
            }}
            setTimeout(scrollToBottom, 150);
        </script>
        """, height=0, width=0
    )


if __name__ == "__main__":
    # Set up global asyncio configuration - Run once at start
    try:
        # nest_asyncio is already applied globally at the top
        # Set Windows event loop policy if on Windows (helps prevent some httpx/aiohttp issues)
        if os.name == 'nt':
            try:
                # Only set policy if no loop is running yet, avoid conflicts
                if not asyncio.get_event_loop_policy().__class__.__name__.startswith('Windows'):
                     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                     logger.info("Successfully set WindowsSelectorEventLoopPolicy.")
                else:
                     logger.info("WindowsSelectorEventLoopPolicy already set or another policy in place.")
            except Exception as policy_e:
                 logger.warning(f"Could not set WindowsSelectorEventLoopPolicy: {policy_e}. Default policy will be used.")

    except Exception as e:
        logger.warning(f"Could not set up global asyncio configuration: {e}")

    # Run the main Streamlit function
    main()
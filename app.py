# app.py

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
from pydantic_ai.format_as_xml import format_as_xml
import plotly.express as px

# --- REMOVED nest_asyncio --- #
# import nest_asyncio
# nest_asyncio.apply()
# logger = logging.getLogger(__name__) # Initialize logger after applying nest_asyncio if needed
# logger.info("nest_asyncio applied.")

# --- Google Generative AI Import ---
try:
    import google.generativeai as genai
except ImportError:
    st.error("Google Generative AI SDK not installed. Please run `pip install google-generativeai`.")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO) # Adjusted level for production
logger = logging.getLogger(__name__) # Use __name__ for module-level logger
# logger.setLevel(logging.DEBUG) # Uncomment for detailed debugging


# --- Configuration and Dependencies (Moved from config.py) ---

# Default SQLite database path (ensure this file exists or is created)
DEFAULT_DB_PATH = st.secrets.get("DATABASE_PATH", "assets/data1.sqlite")

class AgentDependencies:
    """Manages dependencies like database connections."""
    def __init__(self):
        self.db_connection: Optional[sqlite3.Connection] = None

    @classmethod
    def create(cls) -> 'AgentDependencies':
        return cls()

    def with_db(self, db_path: str = DEFAULT_DB_PATH) -> 'AgentDependencies':
        """Establishes SQLite connection."""
        try:
            self.db_connection = sqlite3.connect(db_path)
            logger.info(f"Successfully connected to database: {db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {db_path}: {e}")
            st.error(f"Failed to connect to the database: {e}")
            self.db_connection = None # Ensure connection is None if failed
        return self

    async def cleanup(self):
        """Closes database connection."""
        if self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed.")

# Define Token Usage Limits for Gemini
# Instead of trying to initialize UsageLimits with parameters, create a simple object
# Initialize the class directly - assumes it takes no arguments or has defaults
DEFAULT_USAGE_LIMITS = UsageLimits()
# Remove setattr calls - limits are often handled during run
# setattr(DEFAULT_USAGE_LIMITS, 'request_limit', 500) # Removed
# setattr(DEFAULT_USAGE_LIMITS, 'total_tokens_limit', 1000000) # Removed


# --- Pydantic Models for API Structure ---

class SQLQueryResult(BaseModel):
    """Response when SQL could be successfully generated"""
    sql_query: str = Field(..., description="The SQL query to execute")
    explanation: str = Field("", description="Explanation of what the SQL query does")

class PythonCodeResult(BaseModel):
    """Response when Python code needs to be executed for analysis or visualization"""
    python_code: str = Field(..., description="The Python code to execute using pandas (df) and matplotlib (plt)")
    explanation: str = Field("", description="Explanation of what the Python code does")

class InvalidRequest(BaseModel):
    """Response when the request cannot be processed"""
    error_message: str = Field(..., description="Error message explaining why the request is invalid")

class QueryResponse(BaseModel):
    """Complete response from the agent, potentially including text, SQL, and Python code"""
    text_message: str = Field(..., description="Human-readable response explaining the action or findings")
    sql_result: Optional[SQLQueryResult] = Field(None, description="SQL query details if SQL was generated")
    python_result: Optional[PythonCodeResult] = Field(None, description="Python code details if Python was generated for visualization/analysis")

# --- NEW: Pydantic Model for Database Classification ---
class DatabaseClassification(BaseModel):
    """Identifies the target database for a user query."""
    database_key: Literal["IFC", "MIGA", "UNKNOWN"] = Field(..., description="The database key ('IFC', 'MIGA') the user query most likely refers to, based on keywords and the database descriptions provided. Use 'UNKNOWN' if the query is ambiguous, unrelated, or a general greeting/request.")
    reasoning: str = Field(..., description="Brief explanation for the classification (e.g., 'Query mentions IFC investments', 'Query mentions MIGA guarantees', 'Query is ambiguous/general').")

# --- Configure Google Gemini ---
try:
    # Load API key from Streamlit secrets instead of environment variable
    google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not google_api_key:
        st.error("🔴 GOOGLE_API_KEY not found in secrets.toml or environment variables!")
        st.stop()

    # Configure the Google Generative AI SDK
    genai.configure(api_key=google_api_key)
    logger.info("Google Generative AI SDK configured.")

    # Initialize Gemini Model using pydantic_ai
    gemini_model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash")
    
    # Initialize the model according to pydantic_ai documentation
    # Remove api_key argument - it should use env vars/secrets automatically
    llm = GeminiModel(
        model_name=gemini_model_name
    )
    logger.info(f"Using Gemini Model via pydantic_ai: {gemini_model_name}")

except Exception as e:
    logger.error(f"Error configuring Google Gemini: {e}")
    st.error(f"Error configuring Google Gemini: {e}")
    st.stop()

# --- Define the Agent ---
# This agent handles both SQL generation and Python code generation for visualization.
# It uses the configured Gemini model.
query_agent = Agent(
    llm, # Use the configured Gemini model instance
    deps_type=AgentDependencies,
    result_type=QueryResponse,
    name="SQL and Visualization Assistant",
    retries=3, # Correct parameter for validation retries
    # System prompt is now defined using the decorator below
)

# --- NEW: System Prompt using Decorator --- #
@query_agent.system_prompt
def generate_system_prompt() -> str:
    """Generates the system prompt for the data analysis agent."""
    prompt = f"""You are an expert data analyst assistant. Your role is to help users query and analyze data from a SQLite database.

IMPORTANT: The database schema will be included at the beginning of each user message. Use this schema information to understand the database structure and generate accurate SQL queries. DO NOT respond that you need to know the table structure - it is already provided in the message.

CRITICAL RULES FOR SQL GENERATION:
1. For ANY question about data in the database (counts, totals, listings, comparisons, etc.), you MUST generate an appropriate SQLite query.
2. PAY ATTENTION TO COLUMN NAMES: If a column name in the provided schema contains spaces or special characters, you MUST enclose it in double quotes (e.g., SELECT "Total IFC Investment Amount" FROM ...). Failure to quote such names will cause errors.
3. AGGREGATIONS: For questions asking about totals, sums, or aggregations, use SQL aggregate functions (SUM, COUNT, AVG, etc.).
4. GROUPING: When a question mentions "per" some field (e.g., "per product line"), this requires a GROUP BY clause for that field.
5. SUM FOR TOTALS: Numerical fields asking for totals must use SUM() in your query.
6. SECURITY: ONLY generate SELECT queries. NEVER generate SQL statements that modify the database (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.) or could be potentially harmful. These will be blocked by the system for security reasons and will cause errors.

PYTHON CODE FOR DATA PREPARATION (NOT PLOTTING):
- If a user requests analysis or visualization that requires data manipulation *after* the SQL query (e.g., complex calculations, reshaping data, setting index for charts), generate Python code using pandas.
- Assume the SQL results are available in a pandas DataFrame named 'df'.
- The Python code should ONLY perform data manipulation/preparation on the 'df'.
- CRITICAL: DO NOT include any plotting code (e.g., `matplotlib`, `seaborn`, `st.pyplot`) in the Python code block. The final plotting using native Streamlit charts (like `st.bar_chart`) will be handled separately by the application based on your textual explanation and the prepared data.
- If no specific Python data manipulation is needed beyond the SQL query, do not generate a Python code result.

VISUALIZATION REQUESTS:
- When users request charts, graphs, or plots, first generate the necessary SQL query.
- If the data from SQL needs further processing for the chart (e.g., setting the index, renaming columns), generate Python code as described above to prepare the 'df'.
- In your text response, clearly state the type of chart you recommend (e.g., "bar chart", "line chart", "pie chart", "scatter plot") based on the user's request and the data structure. Use these exact phrases where possible.
- NEVER respond that you cannot create visualizations.

RESPONSE STRUCTURE:
1. First, review the database schema provided in the message.
2. Understand the request: Does it require data retrieval (SQL), potential data preparation (Python), or just a textual answer?
3. Generate SQL: If data is needed, generate an accurate SQLite query string following the rules above, suitable for the `sql_result` field in the response.
4. Generate Python Data Prep Code (if needed): If data manipulation beyond SQL is required for analysis or the requested chart, generate Python pandas code acting on 'df'.
5. Explain Clearly: Explain the SQL query and any Python data preparation steps. If visualization was requested, explicitly suggest the chart type (e.g., "bar chart", "line chart") in your text message.
6. Format Output: Format your final response using the 'QueryResponse' structure. Include 'text_message', 'sql_result' (if applicable), and 'python_result' (if Python data prep code was generated).
7. Safety: Focus ONLY on SELECT queries. Do not generate SQL/Python that modifies the database.
8. Efficiency: Write efficient SQL queries.

Remember, your final output must be the structured 'QueryResponse' object containing the text message and the generated SQL/Python strings (if applicable).
"""
    return prompt

# --- Define Agent Tools ---
@query_agent.tool
async def execute_sql(ctx: RunContext[AgentDependencies], query: str) -> Union[List[Dict], str]:
    """
    Executes a given SQLite SELECT query and returns the results. 
    IMPORTANT: Your primary goal is usually to generate the SQL query string for the final 'QueryResponse' structure, not to execute it yourself. 
    Only use this tool if you absolutely need to fetch intermediate data during your reasoning process to answer a complex multi-step question.
    Otherwise, just generate the SQL query string as part of the QueryResponse.
    Args:
        query (str): The SQLite SELECT query to execute.
    Returns:
        List[Dict]: A list of dictionaries representing the query results.
        str: An error message if the query fails or is not a SELECT statement.
    """
    if not ctx.deps.db_connection:
        return "Error: Database connection is not available."
    
    # Enhanced safety checks: whitelist approach - only allow SELECT statements
    query = query.strip()
    
    # Check for SQL commands that could modify the database or compromise security
    forbidden_commands = ['ALTER', 'CREATE', 'DELETE', 'DROP', 'INSERT', 'UPDATE', 'PRAGMA', 
                          'ATTACH', 'DETACH', 'VACUUM', 'GRANT', 'REVOKE', 'EXECUTE', 'TRUNCATE']
    
    # Normalized query for checking (uppercase without comments)
    normalized_query = ' '.join([
        line for line in query.upper().split('\n') 
        if not line.strip().startswith('--')
    ])
    
    # Basic safety check: only allow SELECT statements
    if not normalized_query.startswith("SELECT"):
        logger.warning(f"Attempted non-SELECT query execution: {query}")
        return "Error: Only SELECT queries are allowed."
    
    # Check for forbidden commands that might be hidden in subqueries or clauses
    for cmd in forbidden_commands:
        if f" {cmd} " in f" {normalized_query} ":
            logger.warning(f"Blocked query containing forbidden command '{cmd}': {query}")
            return f"Error: Detected potentially harmful SQL command '{cmd}'. For security reasons, this operation is not allowed."
    
    # Check for multiple statements with semicolons (except those in quotes)
    # Simple check - this isn't perfect but adds another layer of protection
    statement_count = 0
    in_quotes = False
    for char in query:
        if char in ["'", '"']:
            in_quotes = not in_quotes
        if char == ';' and not in_quotes:
            statement_count += 1
    
    if statement_count > 0:
        logger.warning(f"Blocked query with multiple statements: {query}")
        return "Error: Multiple SQL statements are not allowed for security reasons."
    
    try:
        cursor = ctx.deps.db_connection.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description] if cursor.description else []
        results = cursor.fetchall()
        logger.info(f"Executed SQL query successfully. Rows returned: {len(results)}")
        return [dict(zip(columns, row)) for row in results]
    except sqlite3.Error as e:
        logger.error(f"SQL execution error: {e}. Query: {query}")
        return f"Error executing SQL query: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during SQL execution: {e}. Query: {query}")
        return f"An unexpected error occurred: {str(e)}"


# --- New functions to handle metadata JSON ---
METADATA_PATH = Path(__file__).parent / "assets" / "database_metadata.json"

@st.cache_data
def load_db_metadata(path: Path = METADATA_PATH) -> Optional[Dict]:
    """Loads the database metadata from the specified JSON file."""
    if not path.exists():
        st.error(f"Metadata file not found: {path}")
        logger.error(f"Metadata file not found: {path}")
        return None
    try:
        with open(path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Successfully loaded database metadata from {path}")
        return metadata
    except json.JSONDecodeError as e:
        st.error(f"Error decoding metadata JSON from {path}: {e}")
        logger.error(f"Error decoding metadata JSON from {path}: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading metadata file {path}: {e}")
        logger.error(f"Error loading metadata file {path}: {e}")
        return None

def format_schema_from_metadata(metadata: Optional[Dict]) -> str:
    """Formats the schema string from loaded metadata for the AI prompt."""
    if not metadata or 'tables' not in metadata:
        return "Error: Could not load or parse database metadata."

    schema_parts = []
    db_desc = metadata.get("description")
    if db_desc:
        schema_parts.append(f"Database Description: {db_desc}")

    for table_name, table_info in metadata.get("tables", {}).items():
        schema_parts.append(f"\nTable: {table_name}")
        table_desc = table_info.get("description")
        if table_desc:
            schema_parts.append(f"  (Description: {table_desc})")

        columns = table_info.get("columns", {})
        if columns:
            for col_name, col_info in columns.items():
                col_type = col_info.get("type", "UNKNOWN")
                col_desc = col_info.get("description", "")
                schema_parts.append(f"  - {col_name} ({col_type}) - {col_desc}")
        else:
            schema_parts.append("  - (No columns found in metadata)")

    if not schema_parts or len(schema_parts) <= 1: # Check if only DB desc was added
        return "No tables found in the database metadata."

    return "\n".join(schema_parts)

# --- MODIFIED: Function to format schema for a SPECIFIC database ---
def format_schema_for_db(metadata: Dict, db_key: str) -> str:
    """Formats the schema string for a specific database key from loaded metadata."""
    if 'databases' not in metadata or db_key not in metadata['databases']:
        return f"Error: Database key '{db_key}' not found in metadata."

    db_entry = metadata['databases'][db_key]
    schema_parts = []
    db_name = db_entry.get("database_name", db_key)
    db_desc = db_entry.get("description", "")

    schema_parts.append(f"Database: {db_name} ({db_key})")
    if db_desc:
        schema_parts.append(f"Description: {db_desc}")

    tables = db_entry.get("tables", {})
    if not tables:
         schema_parts.append("\nNo tables found in metadata for this database.")
         return "\n".join(schema_parts)

    for table_name, table_info in tables.items():
        schema_parts.append(f"\nTable: {table_name}")
        table_desc = table_info.get("description")
        if table_desc:
            schema_parts.append(f"  (Description: {table_desc})")

        columns = table_info.get("columns", {})
        if columns:
            for col_name, col_info in columns.items():
                col_type = col_info.get("type", "TEXT") # Default to TEXT if missing
                col_desc = col_info.get("description", "")
                # Ensure type is not empty, default again if it somehow is
                col_type_display = col_type if col_type else "TEXT"
                schema_parts.append(f"  - {col_name} ({col_type_display}) - {col_desc}")
        else:
            schema_parts.append("  - (No columns found in metadata for this table)")

    return "\n".join(schema_parts)

# --- NEW: Function to identify target database using LLM ---
async def identify_target_database(
    user_query: str,
    metadata: Dict,
    model: GeminiModel # Reuse the existing model
) -> Tuple[Optional[str], str]:
    """
    Uses the LLM to classify the user query against database descriptions.

    Args:
        user_query: The user's input message.
        metadata: The loaded database metadata dictionary.
        model: The GeminiModel instance to use for classification.

    Returns:
        A tuple: (identified_db_key or None, reasoning_message)
        - identified_db_key: "IFC", "MIGA", or None if classification is "UNKNOWN" or fails.
        - reasoning_message: Explanation from the classification model or error message.
    """
    logger.info(f"Attempting to identify target database for query: {user_query[:50]}...")
    if 'databases' not in metadata:
        return None, "Error: 'databases' key missing in metadata."

    descriptions = []
    valid_keys = []
    for key, db_info in metadata['databases'].items():
        desc = db_info.get('description', f'Database {key}')
        descriptions.append(f"- {key}: {desc}")
        valid_keys.append(key)

    if not descriptions:
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
    logger.info(f"Prompt:\n{classification_prompt}")
    logger.info("--------------------------------------------")

    try:
        # Use a separate Agent instance or direct model call if Agent interferes
        # For simplicity, let's try a direct structured output call if available,
        # otherwise use a temporary agent setup.
        # Using a temporary Agent for structured output:
        classifier_agent = Agent(
            model,
            result_type=DatabaseClassification,
            name="Database Classifier",
            system_prompt="You are an AI assistant that classifies user queries based on provided database descriptions. Output ONLY the structured classification result."
        )
        # Note: We don't need dependencies (deps) for this classification call
        classification_result = await classifier_agent.run(classification_prompt)

        if hasattr(classification_result, 'data') and isinstance(classification_result.data, DatabaseClassification):
            result_data: DatabaseClassification = classification_result.data
            logger.info(f"--- LLM Classification Result ---")
            logger.info(f"Key: {result_data.database_key}")
            logger.info(f"Reasoning: {result_data.reasoning}")
            logger.info("-------------------------------")
            if result_data.database_key == "UNKNOWN":
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

# --- Optional: Result Validation ---
@query_agent.result_validator
async def validate_query_result(ctx: RunContext[AgentDependencies], result: QueryResponse) -> QueryResponse:
    """
    Validate the generated response.
    - Checks SQL syntax using EXPLAIN QUERY PLAN.
    - Cleans potential extraneous characters from SQL.
    - Checks if SQL is missing when likely needed.
    - Checks if Python code is missing or declined for visualization requests.
    - Enforces SQL security by blocking potentially harmful statements.
    Raises ModelRetry if validation fails, prompting the LLM to correct the response.
    """
    user_message = ctx.prompt # The message sent to the agent (includes schema and user query)
    logger.info(f"Running result validation for prompt: {user_message[:100]}...")
    logger.debug(f"Validator received result object: {result}") # DEBUG log for full object if needed

    # --- SQL Validation --- #
    if result.sql_result and ctx.deps.db_connection:
        # Clean SQL query - remove potential extraneous backslashes
        original_sql = result.sql_result.sql_query
        cleaned_sql = original_sql.replace('\\', '') # Replace backslashes
        if cleaned_sql != original_sql:
            logger.info(f"Cleaned SQL query. Original: '{original_sql}', Cleaned: '{cleaned_sql}'")
            result.sql_result.sql_query = cleaned_sql # Update the result object
        else:
            cleaned_sql = original_sql # Ensure cleaned_sql is set

        # --- Enhanced Security Validation --- #
        # Check for potentially harmful SQL statements - similar to execute_sql but earlier in the pipeline
        forbidden_commands = ['ALTER', 'CREATE', 'DELETE', 'DROP', 'INSERT', 'UPDATE', 'PRAGMA', 
                           'ATTACH', 'DETACH', 'VACUUM', 'GRANT', 'REVOKE', 'EXECUTE', 'TRUNCATE']
        
        # Normalized query for checking (uppercase without comments)
        normalized_query = ' '.join([
            line for line in cleaned_sql.upper().split('\n') 
            if not line.strip().startswith('--')
        ])
        
        # Check if query is a SELECT statement
        if not normalized_query.strip().startswith("SELECT"):
            logger.warning(f"Non-SELECT query generated: {cleaned_sql}")
            raise ModelRetry("Only SELECT queries are allowed for security reasons. Please regenerate a proper SELECT query.")
        
        # Check for forbidden commands that might be hidden in subqueries or clauses
        for cmd in forbidden_commands:
            if f" {cmd} " in f" {normalized_query} ":
                logger.warning(f"Detected forbidden SQL command '{cmd}' in: {cleaned_sql}")
                raise ModelRetry(f"The SQL query contains a potentially harmful command '{cmd}'. For security reasons, only pure SELECT statements are allowed. Please regenerate the query without this command.")
        
        # Check for multiple statements with semicolons (except those in quotes)
        statement_count = 0
        in_quotes = False
        for char in cleaned_sql:
            if char in ["'", '"']:
                in_quotes = not in_quotes
            if char == ';' and not in_quotes:
                statement_count += 1
        
        if statement_count > 0:
            logger.warning(f"Multiple SQL statements detected: {cleaned_sql}")
            raise ModelRetry("Multiple SQL statements are not allowed for security reasons. Please provide a single SELECT query without semicolons.")

        # Validate SQL Syntax using EXPLAIN QUERY PLAN (suitable for SQLite)
        try:
            cursor = ctx.deps.db_connection.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {cleaned_sql}")
            cursor.fetchall()
            logger.info("Generated SQL query syntax validated successfully.")
        except sqlite3.Error as e:
            error_detail = f"SQL Syntax Validation Error: {e}. Query: {cleaned_sql}"
            logger.error(error_detail)
            logger.warning(f"Raising ModelRetry due to SQL Syntax Error. Response details: text='{result.text_message}', sql='{cleaned_sql}'")
            raise ModelRetry(f"The generated SQL query has invalid syntax: {str(e)}. Please correct the SQL query.") from e
        except Exception as e:
             error_detail = f"Unexpected SQL Validation Error: {e}. Query: {cleaned_sql}"
             logger.error(error_detail)
             logger.warning(f"Raising ModelRetry due to Unexpected SQL Error. Response details: text='{result.text_message}', sql='{cleaned_sql}'")
             raise ModelRetry(f"An unexpected error occurred during SQL validation: {str(e)}. Please try generating the SQL query again.") from e

    # --- Check for Missing SQL when Expected --- #
    # Simple keyword check on the original user query part of the prompt
    data_query_keywords = ['total', 'sum', 'average', 'count', 'list', 'show', 'per', 'group', 'compare', 'what is', 'how many']
    # Extract the user's actual question if possible (assuming structure "User Question: ...")
    user_question_marker = "User Question:"
    original_user_question = user_message[user_message.find(user_question_marker):] if user_question_marker in user_message else user_message

    if not result.sql_result and any(keyword in original_user_question.lower() for keyword in data_query_keywords):
        # Check if it's just a clarification or greeting
        is_greeting = any(greet in original_user_question.lower() for greet in ['hello', 'hi', 'thanks', 'thank you'])
        # More robust clarification check - look for question marks or specific keywords
        is_clarification = '?' not in original_user_question and not any(kw in original_user_question.lower() for kw in ['explain', 'what is', 'how does'])

        if not is_greeting and not is_clarification:
            logger.warning(f"SQL result is missing, but keywords {data_query_keywords} suggest it might be needed for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Missing SQL. Response details: text='{result.text_message}', sql=None")
            raise ModelRetry("The user's question appears to require data retrieval, but no SQL query was generated. Please generate the appropriate SQL query.")

    # --- Visualization Validation ---
    visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation', 'bar chart', 'pie chart', 'histogram', 'line graph']
    decline_phrases = ['cannot plot', 'unable to plot', 'cannot visualize', 'unable to visualize', 'cannot create chart', 'unable to create chart', 'do not have the ability to create plots']
    chart_suggestion_phrases = ["bar chart", "line chart", "pie chart", "scatter plot", "area chart"] # Expected suggestions
    is_visualization_request = any(keyword in original_user_question.lower() for keyword in visualization_keywords)
    has_declined_visualization = any(phrase in result.text_message.lower() for phrase in decline_phrases)
    has_suggested_chart = any(phrase in result.text_message.lower() for phrase in chart_suggestion_phrases)

    if is_visualization_request:
        # Check 1: Did the AI decline?
        if has_declined_visualization:
            logger.warning(f"Visualization requested, but the text response seems to decline the capability for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Visualization Declined. Response details: text='{result.text_message}', sql='{result.sql_result.sql_query if result.sql_result else None}'")
            raise ModelRetry("The response incorrectly stated an inability to create visualizations. You MUST suggest an appropriate chart type (e.g., 'bar chart', 'line chart') in your text message and generate the necessary SQL query.")

        # Check 2: Was SQL generated? (Essential for any viz)
        if not result.sql_result:
            logger.warning(f"Visualization requested, but SQL query is missing for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Missing SQL for Viz. Response details: text='{result.text_message}', sql=None")
            raise ModelRetry("The user requested a visualization, but the SQL query needed to fetch the data was missing. Please generate the appropriate SQL query and suggest a chart type in your text message.")

        # Check 3: Was a chart type suggested in the text?
        if not has_suggested_chart:
            logger.warning(f"Visualization requested, SQL provided, but no chart type suggested in text for query: {original_user_question[:100]}...")
            logger.warning(f"Raising ModelRetry due to Missing Chart Suggestion. Response details: text='{result.text_message}', sql='{result.sql_result.sql_query if result.sql_result else None}', python='{'present' if result.python_result else 'absent'}'")
            # If Python prep code exists, the AI might think that's enough. Clarify text suggestion is needed.
            if result.python_result:
                 raise ModelRetry("The user requested a visualization, and you provided SQL and Python data preparation code. However, you MUST also explicitly suggest the chart type (e.g., 'bar chart', 'line chart') in your text message.")
            else:
                 raise ModelRetry("The user requested a visualization, and you provided the SQL query. However, you MUST also explicitly suggest the chart type (e.g., 'bar chart', 'line chart') in your text message.")

        # Check 4: If Python code exists, is SQL missing? (Should be caught by Check 2, but good redundancy)
        # This scenario implies Python code was generated perhaps for non-viz reasons, but viz was also requested.
        if result.python_result and not result.sql_result:
             logger.warning(f"Visualization requested and Python code generated, but the necessary SQL query is missing in this response for query: {original_user_question[:100]}...")
             logger.warning(f"Raising ModelRetry due to Missing SQL with Python for Viz. Response details: text='{result.text_message}', sql=None, python='present'")
             raise ModelRetry("The user requested a visualization and Python code was generated, but the SQL query needed to fetch the data was missing from the response. Please provide BOTH the SQL query and suggest a chart type in your text message.")

    # --- Check if Python code depends on SQL results that weren't generated ---
    # (Keeping the original warning logic, but could also be a ModelRetry case)
    if result.python_result and not result.sql_result and not is_visualization_request:
         logger.warning("Python code generated without corresponding SQL query.")
         # Modify explanation instead of retry for now, unless it's clearly broken
         result.text_message += "\nNote: Python code was generated, but no SQL query was provided for this step. The Python code might expect data that isn't available."
         result.python_result.explanation += " Warning: This code might assume data from a previous step or may not run correctly without prior data loading."
         # Alternatively, could raise retry:
         # logger.warning(f"Raising ModelRetry due to Python without SQL. Response details: text='{result.text_message}', sql=None, python='present'")
         # raise ModelRetry("Python code was generated, but it seems to depend on data from a SQL query which was not generated. Please generate the SQL query first, then the Python code.")

    logger.info("Result validation completed successfully.")
    return result # Return the validated (or modified) result

# --- Helper Functions ---
def get_base64_encoded_image(image_path):
    """Get base64 encoded image"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logger.warning(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None

# --- Main Application Logic ---
async def handle_user_message(message: str) -> None:
    """Handles user input, identifies DB, runs the agent, and updates the chat history state."""
    logger.info(f"handle_user_message started for message: {message[:50]}...")
    
    # Log the user's full message
    logger.info("==== USER QUERY ====")
    logger.info(message)
    logger.info("===================")

    # Ensure we're running in a thread with an event loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop detected in handle_user_message, creating one")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    deps = None
    assistant_chat_message = None
    agent_run_result = None
    target_db_key = None # Track the identified database

    # --- Load Metadata (Step 1) ---
    db_metadata = load_db_metadata() # Load metadata using cached function
    if not db_metadata:
        # Error handled within load_db_metadata, just stop processing
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": "Sorry, I couldn't load the database configuration. Please check the logs."
        })
        return

    try:
        # --- Identify Target Database (Step 2) ---
        target_db_key, reasoning = await identify_target_database(message, db_metadata, llm)

        if not target_db_key:
            # Check if there's a previous context
            last_key = st.session_state.get('last_db_key')
            if last_key:
                logger.warning(f"Database identification failed, but reusing last key: {last_key}. Original Reasoning: {reasoning}")
                target_db_key = last_key # Reuse the last key
            else:
                # No previous context, ask the user
                logger.warning(f"Database identification failed or returned UNKNOWN. Reasoning: {reasoning}")
                assistant_chat_message = {
                    "role": "assistant",
                    "content": f"I'm not sure which database to use for your query (IFC or MIGA). Could you please specify? (Reasoning: {reasoning})"
                }
                st.session_state.chat_history.append(assistant_chat_message)
                return # Stop processing if DB not identified

        logger.info(f"Target database identified as: {target_db_key}")
        st.session_state.last_db_key = target_db_key # Store the successfully identified key

        # --- Get Specific DB Path and Connect (Step 3) ---
        db_entry = db_metadata.get('databases', {}).get(target_db_key)
        if not db_entry or 'database_path' not in db_entry:
            error_msg = f"Metadata configuration error: Could not find path for database '{target_db_key}'."
            logger.error(error_msg)
            st.error(error_msg)
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, internal configuration error for database {target_db_key}."}
            st.session_state.chat_history.append(assistant_chat_message)
            return

        target_db_path = db_entry['database_path']
        logger.info(f"Connecting to database: {target_db_path} for key: {target_db_key}")
        deps = AgentDependencies.create().with_db(db_path=target_db_path)

        if not deps.db_connection:
             st.error(f"Database connection failed for {target_db_path}. Cannot process request.")
             assistant_chat_message = {
                 "role": "assistant",
                 "content": f"Sorry, I couldn't connect to the {target_db_key} database..."
             }
             st.session_state.chat_history.append(assistant_chat_message)
             # No need for cleanup here as connection failed
             return

        # --- Format Schema for Identified Database (Step 4) ---
        schema_info = format_schema_for_db(db_metadata, target_db_key)
        logger.info(f"Formatted schema for {target_db_key}:\n{schema_info}")
        if schema_info.startswith("Error:"):
            st.error(schema_info) # Display the specific error
            assistant_chat_message = {
                "role": "assistant",
                "content": f"Sorry, I encountered an issue loading the schema for the {target_db_key} database: {schema_info}"
            }
            st.session_state.chat_history.append(assistant_chat_message)
            if deps: await deps.cleanup()
            return

        # --- Prepare and Run Main Query Agent (Step 5) ---
        usage = Usage() # Initialize usage tracking

        # --- Token Limit Check --- #
        current_total_tokens = getattr(usage, 'total_tokens', None) # Get current tokens, default to None
        # Ensure current_total_tokens is treated as 0 if it's None
        if current_total_tokens is None:
            current_total_tokens = 0

        # Get the configured limit, default to None
        limit_value = getattr(DEFAULT_USAGE_LIMITS, 'total_tokens_limit', None)
        # Set a default integer limit if the configured value is None or not an integer
        total_tokens_limit = limit_value if isinstance(limit_value, int) else 1000000

        # Perform the comparison using integers
        if current_total_tokens > total_tokens_limit:
            error_msg = "Token limit exceeded..."
            st.error(error_msg)
            assistant_chat_message = {"role": "assistant", "content": error_msg}
            st.session_state.chat_history.append(assistant_chat_message)
            return

        try:
            logger.info(f"Analyzing request for database '{target_db_key}' and contacting Gemini...")

            # Get the current cumulative history for the agent
            history_for_agent = st.session_state.agent_message_history
            logger.info(f"Passing cumulative history (length {len(history_for_agent)}) to agent.")

            # Construct the prompt message
            if not history_for_agent:
                # No history, this is the first turn. Include schema.
                logger.info("No history found, adding schema to current prompt.")
                # Check if this is a visualization request (slightly different framing)
                visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation', 'bar chart', 'pie chart', 'histogram']
                is_visualization_request = any(keyword in message.lower() for keyword in visualization_keywords)
                if is_visualization_request:
                    logger.info("Detected a visualization request - adjusting initial prompt.")
                    prompt_message = f"""Target Database: {target_db_key}
Database Schema:
{schema_info}

User Request: {message}

IMPORTANT: This is a visualization request for the {target_db_key} database.
1. Generate the appropriate SQL query to retrieve the necessary data from the provided schema.
2. In your text response, you MUST suggest an appropriate chart type (e.g., "bar chart", "line chart", "pie chart") based on the user's request and the data.
Do NOT generate Python code for plotting (e.g., using matplotlib or seaborn).
"""
                else:
                    # Regular non-visualization request for the first turn
                    prompt_message = f"""Target Database: {target_db_key}
Database Schema:
{schema_info}

User Question: {message}"""
            else:
                # History exists, just pass the user's message directly.
                # Schema and context are in the history_for_agent list.
                prompt_message = message
                logger.info("History found, using raw message for current prompt.")

            # Log the message we're sending to the AI *for this turn*
            logger.info("==== AI CALL (This Turn's Prompt) ====")
            logger.info(f"Sending prompt message to AI:\n{prompt_message}")
            logger.info("=====================================")

            agent_run_result = await query_agent.run(
                prompt_message,  # Use the correctly constructed message
                deps=deps,
                usage=usage,
                usage_limits=DEFAULT_USAGE_LIMITS,
                message_history=history_for_agent # Pass the cumulative history
            )
            st.session_state.last_result = agent_run_result # Still store last result for display logic if needed
            logger.info("Stored agent run result in session state.")

            # Append new messages to the cumulative history
            st.session_state.agent_message_history.extend(agent_run_result.new_messages())
            logger.info(f"Appended {len(agent_run_result.new_messages())} new messages to agent_message_history (new total: {len(st.session_state.agent_message_history)}). ")
            
            # Log the raw agent result for debugging
            logger.info("==== AGENT RUN RESULT ====")
            logger.info(f"Agent result type: {type(agent_run_result)}")
            logger.info(f"Agent result attributes: {dir(agent_run_result)}")
            if hasattr(agent_run_result, 'raw_response'):
                logger.info(f"Raw response: {agent_run_result.raw_response}")
            if hasattr(agent_run_result, 'data'):
                logger.info(f"Data type: {type(agent_run_result.data)}")
                if agent_run_result.data:
                    logger.info(f"Data attributes: {dir(agent_run_result.data)}")
            logger.info("=========================")

            if hasattr(agent_run_result, 'data') and isinstance(agent_run_result.data, QueryResponse):
                response: QueryResponse = agent_run_result.data
                logger.info("Processing agent response...")
                
                # Log the response from the AI
                logger.info("==== AI RESPONSE ====")
                logger.info(f"Text message: {response.text_message}")
                if response.sql_result:
                    logger.info(f"SQL query: {response.sql_result.sql_query}")
                    logger.info(f"SQL explanation: {response.sql_result.explanation}")
                
                if response.python_result:
                    logger.info(f"Python code explanation: {response.python_result.explanation}")
                    logger.info("Python code snippet:")
                    for line in response.python_result.python_code.split('\n'):
                        logger.info(f"  {line}")
                logger.info("=====================")

                assistant_chat_message = {"role": "assistant", "content": f"[{target_db_key} database] {response.text_message}"}

                sql_results_df = None
                if response.sql_result:
                    # --- ADDED LOGGING FOR THE SQL QUERY --- #
                    logger.info(f"LLM generated SQL query: {response.sql_result.sql_query}")
                    # --- END ADDED LOGGING ---
                    logger.info(f"Executing SQL query against {target_db_key} database...")
                    sql_run_context = RunContext(
                        deps=deps,
                        model=llm,
                        usage=usage,
                        prompt=response.sql_result.sql_query
                    )
                    sql_execution_result = await execute_sql(sql_run_context, response.sql_result.sql_query)
                    sql_info = {
                        "query": response.sql_result.sql_query,
                        "explanation": response.sql_result.explanation
                    }
                    if isinstance(sql_execution_result, str): # Error
                        sql_info["error"] = sql_execution_result
                        logger.error(f"SQL execution failed: {sql_execution_result}")
                    elif isinstance(sql_execution_result, list):
                        if sql_execution_result:
                            sql_results_df = pd.DataFrame(sql_execution_result)
                            sql_info["results"] = sql_results_df.to_dict('records')
                            sql_info["columns"] = list(sql_results_df.columns)
                        else:
                            sql_info["results"] = [] # Empty results
                            sql_results_df = pd.DataFrame() # Ensure df is an empty DataFrame
                    else:
                         sql_info["error"] = "Unexpected result type from SQL execution."
                         logger.error(sql_info["error"])
                         sql_results_df = pd.DataFrame() # Ensure df is an empty DataFrame on unexpected error

                    assistant_chat_message["sql_result"] = sql_info

                # Initialize df_for_chart with the SQL results (or empty if failed/no results)
                df_for_chart = sql_results_df if sql_results_df is not None else pd.DataFrame()
                chart_type = None # Initialize chart type

                if response.python_result:
                    logger.info("Executing Python data preparation code...")
                    python_info = {
                        "code": response.python_result.python_code,
                        "explanation": response.python_result.explanation
                    }
                    python_code = response.python_result.python_code

                    # Prepare local variables for exec, including the DataFrame from SQL
                    local_vars = {
                        'pd': pd,
                        'np': np,
                        'st': st, # Keep st in case it's used for non-plotting things
                        'df': df_for_chart.copy() # Pass a copy to avoid modifying the original outside exec scope
                    }

                    try:
                        exec(python_code, globals(), local_vars)
                        # Retrieve the potentially modified DataFrame from local_vars
                        df_for_chart = local_vars['df']
                        logger.info("Python data preparation code executed successfully.")
                    except Exception as e:
                        logger.error(f"Error executing Python data preparation code: {e}\nCode:\n{python_code}")
                        python_info["error"] = str(e)

                    assistant_chat_message["python_result"] = python_info

                # --- Determine Chart Type from AI Response --- #
                if df_for_chart is not None and not df_for_chart.empty:
                    text_lower = response.text_message.lower()
                    if "bar chart" in text_lower:
                        chart_type = "bar"
                    elif "line chart" in text_lower:
                        chart_type = "line"
                    elif "area chart" in text_lower:
                        chart_type = "area"
                    elif "scatter plot" in text_lower or "scatter chart" in text_lower:
                        chart_type = "scatter"
                    elif "pie chart" in text_lower:
                        chart_type = "pie" # Note: st.pie_chart is deprecated, requires alternatives

                    if chart_type:
                        logger.info(f"Detected chart type: {chart_type}")
                        # --- Prepare DataFrame for Streamlit Charting (Example: Set Index) --- #
                        # Streamlit charts often use the index for category labels.
                        # Attempt to automatically set index if not already done by AI's python code.
                        if df_for_chart.index.name is None and len(df_for_chart.columns) > 1:
                            potential_index_col = df_for_chart.columns[0]
                            # Check if the first column is suitable as an index (e.g., string type)
                            if pd.api.types.is_string_dtype(df_for_chart[potential_index_col]):
                                try:
                                     df_for_chart = df_for_chart.set_index(potential_index_col)
                                     logger.info(f"Automatically set DataFrame index to '{potential_index_col}' for charting.")
                                except Exception as e:
                                     logger.warning(f"Could not automatically set index for charting: {e}")

                        # Store chart type and data for display function
                        assistant_chat_message["streamlit_chart"] = {
                            "type": chart_type,
                            "data": df_for_chart
                        }
                    else:
                        logger.info("No specific chart type detected in AI response, or data is empty.")
                else:
                     logger.info("DataFrame is empty or None, skipping chart generation.")

                logger.info("Agent processing complete.")

            else:
                 error_msg = "Received an unexpected response structure..."
                 logger.error(f"{error_msg} Raw RunResult: {agent_run_result}")
                 st.error(error_msg)
                 assistant_chat_message = {"role": "assistant", "content": f"Sorry, internal issue with the {target_db_key} database query... ({error_msg})"}

        except Exception as e:
            error_msg = f"An error occurred during agent processing: {str(e)}"
            logger.exception("Error during agent execution or response processing:")
            st.error(error_msg)
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, I encountered an error querying the {target_db_key} database: {str(e)}"}

    except Exception as e:
        error_msg = f"A critical error occurred: {str(e)}"
        logger.exception("Critical error in handle_user_message setup:")
        st.error(error_msg)
        assistant_chat_message = {"role": "assistant", "content": error_msg}

    finally:
        if assistant_chat_message:
            st.session_state.chat_history.append(assistant_chat_message)
            logger.info("Assistant message appended to history.")
        else:
             logger.error("handle_user_message finished without generating an assistant message object.")
             # Append a generic error message if nothing else was generated
             if not any(m['role'] == 'assistant' and m['content'].startswith("Sorry, I encountered an error") for m in st.session_state.chat_history[-2:]):
                st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, an internal error occurred, and no response could be generated."})

        if deps:
            await deps.cleanup()
        logger.info(f"handle_user_message finished for message: {message[:50]}...")


def main():
    """Main function to run the Streamlit application."""

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [] # For display
    # if 'conversation_context' not in st.session_state: # This seems unused, commenting out
    #     st.session_state.conversation_context = []
    #     logger.info("Initialized conversation context.")
    if 'last_result' not in st.session_state: # Retained for now, might be removable later
        st.session_state.last_result = None
        logger.info("Initialized last_result in session state.")
    if 'last_db_key' not in st.session_state:
        st.session_state.last_db_key = None
        logger.info("Initialized last_db_key in session state.")
    if 'agent_message_history' not in st.session_state: # ADDED: Initialize cumulative history
        st.session_state.agent_message_history = []
        logger.info("Initialized agent_message_history in session state.")

    # --- Main Page Content ---
    st.markdown('<h1 style="text-align: center;"><span style="color: #00ade4;">SmartQuery</span></h1>', unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #555;'>AI-Powered Database Analysis with Google Gemini</h5>", unsafe_allow_html=True)


    # --- CSS Styles ---
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
    # Adjusted path for single file structure
    check_path = Path(__file__).parent / "assets" / "correct.png"
    check_base64 = get_base64_encoded_image(check_path)
    check_img = f'<img src="data:image/png;base64,{check_base64}" class="check-icon" alt="✓">' if check_base64 else "✓"

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

        # Clear Chat Button (moved from sidebar)
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        # st.session_state.conversation_context = [] # Unused
        st.session_state.last_result = None
        st.session_state.last_db_key = None
        st.session_state.agent_message_history = [] # ADDED: Reset cumulative history
        logger.info("Chat history, last_result, last_db_key, and agent_message_history cleared.")
        st.rerun()

    # --- Chat Interface --- #
    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages-container">', unsafe_allow_html=True)

    # Chat display container - Renders based on current chat_history
    chat_display_container = st.container()
    with chat_display_container:
        # Display chat history loop
        for i, message in enumerate(st.session_state.chat_history):
            role = message.get("role", "assistant")
            content = message.get("content", "")
            with st.chat_message(role):
                st.markdown(content)
                
                # Display SQL results if present
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
                
                # Display Python results if present
                python_result = message.get("python_result")
                if python_result:
                    st.markdown("**Python Code:**")
                    st.code(python_result.get("code", ""), language="python")
                    
                    if "explanation" in python_result and python_result["explanation"]:
                        st.markdown(f"**Explanation:** {python_result['explanation']}")
                    
                    if "error" in python_result:
                        st.error(f"Error executing Python code: {python_result['error']}")

                # Display visualization if available (Moved outside python_result check)
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
                            # Make sure columns are specified for scatter
                            if len(df.columns) >= 2:
                                st.scatter_chart(df, x=df.columns[0], y=df.columns[1])
                            else:
                                st.warning("Scatter plot requires at least two data columns.")
                        # --- ADDED PIE CHART HANDLING WITH PLOTLY --- #
                        elif chart_type == "pie":
                            if not df.empty and len(df.columns) > 0:
                                # Assuming index = names, first column = values
                                fig = px.pie(df, names=df.index, values=df.columns[0], title="Pie Chart")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Cannot generate pie chart: Data is empty or missing columns.")
                        # --- END PIE CHART HANDLING ---
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")

    # Close chat messages container
    st.markdown('</div>', unsafe_allow_html=True)

    # --- User Input --- #
    user_input = st.chat_input("Ask about IFC or MIGA data...")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Handle Input --- #
    # Use explicit loop management with get_event_loop()
    if user_input:
        logger.info(f"User input received: {user_input}")
        
        # Add user message to chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Use a spinner while processing the user's query
        with st.spinner("Thinking..."):
            # Process the user's message
            try:
                logger.info("Setting up asyncio for handle_user_message.")
                # Create a new event loop for this thread if one doesn't exist
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop exists in this thread, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    logger.info("Created new event loop for ScriptRunner thread.")

                # Run the coroutine in the event loop
                loop.run_until_complete(handle_user_message(user_input))
                logger.info("handle_user_message completed successfully.")

            except Exception as e:
                 logger.exception(f"Error processing user input in main: {e}")
                 st.error(f"An error occurred while processing your request: {e}")
                 # Append error to history state if handle_user_message failed
                 if not st.session_state.chat_history or not st.session_state.chat_history[-1]['content'].endswith(str(e)):
                      st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

        # Force refresh the UI to show new messages
        st.rerun()

        logger.info("Input processed, script run finishing.")


if __name__ == "__main__":
    # Set event loop policy if needed (might still be relevant without nest_asyncio)
    try:
        if os.name == 'nt': # Check if OS is Windows
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
             logger.info("Set WindowsSelectorEventLoopPolicy.")
        
        # Ensure there's a default event loop for the main thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info("Created new event loop for main thread.")
    except Exception as e:
        logger.warning(f"Could not set event loop policy: {e}")
    main()
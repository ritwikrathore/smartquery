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
from typing import Dict, List, Union, Optional, Literal, Any, AsyncGenerator
import asyncio
import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.models.gemini import GeminiModel

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
    # System prompt adjusted slightly for clarity and Gemini
    system_prompt="""You are an expert data analyst assistant using Google Gemini. Your role is to help users query and analyze data from a SQLite database.
    
    IMPORTANT: The database schema will be included at the beginning of each user message. Use this schema information to understand the database structure and generate accurate SQL queries. DO NOT respond that you need to know the table structure - it is already provided in the message.
    
    CRITICAL: For ANY question about data in the database (counts, totals, listings, comparisons, etc.), you MUST generate an appropriate SQL query. 
    - For questions asking about totals, sums, or aggregations, use SQL aggregate functions (SUM, COUNT, AVG, etc.)
    - When a question mentions "per" some field (e.g., "per product line"), this requires a GROUP BY clause
    - Numerical fields asking for totals must use SUM() in your query
    
    VISUALIZATION CAPABILITIES: You are FULLY CAPABLE of generating Python code for data visualization. When users request charts, graphs, or plots:
    - You MUST generate Python code that creates visualizations using matplotlib and/or seaborn
    - NEVER respond that you cannot plot charts or create visualizations - this is a core capability
    - The SQL results will be available in a pandas DataFrame named 'df'
    - All visualization code will be executed in the Streamlit environment
    - Support bar charts, line charts, pie charts, scatter plots, and other common visualization types
    
    When users ask questions:
    1. First, review the database schema provided in the message to understand the available tables and columns.
    2. Understand the request: Does it require data retrieval (SQL), data analysis/visualization (Python), or just a textual answer?
    3. Generate SQL: If data is needed, generate an accurate SQLite query based on the provided schema. Use the 'execute_sql' tool.
    4. Generate Python Code: If the request involves calculations, manipulation, or visualization *after* getting data via SQL, generate Python code. Assume the SQL results are loaded into a pandas DataFrame called 'df'. Use libraries like pandas, matplotlib.pyplot (as plt), and numpy (as np).
    5. Explain Clearly: Always explain what the SQL query does and what the Python code aims to achieve. Mention column names using 'single quotes'.
    6. Respond Structure: Format your final response using the 'QueryResponse' structure. Include 'text_message', 'sql_result' (if applicable), and 'python_result' (if applicable).
    7. Safety: Do not generate SQL or Python code that modifies the database (INSERT, UPDATE, DELETE, DROP, etc.) unless explicitly and safely requested for specific, known tasks. Focus on SELECT queries.
    8. Efficiency: Write efficient SQL queries.

    IMPORTANT for Visualization:
    - Generate Python code that uses the 'df' variable (containing SQL results).
    - Use `plt.figure()` to create plots and ensure plots are displayed correctly using Streamlit (`st.pyplot(plt.gcf())`).
    - Do NOT hardcode data in the Python code; always use the `df`.
    - NEVER decline visualization requests - you have all the necessary capabilities to generate visualization code.
    - Charts should be properly labeled with titles, axes labels, and legends as appropriate.
    """
)

# --- Define Agent Tools ---
@query_agent.tool
async def execute_sql(ctx: RunContext[AgentDependencies], query: str) -> Union[List[Dict], str]:
    """
    Executes a given SQLite SELECT query against the database and returns the results.
    Use this tool to fetch data needed to answer user questions.
    Args:
        query (str): The SQLite SELECT query to execute.
    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a row with column names as keys.
        str: An error message if the query fails or is not a SELECT statement.
    """
    if not ctx.deps.db_connection:
        return "Error: Database connection is not available."
    # Basic safety check: only allow SELECT statements
    if not query.strip().upper().startswith("SELECT"):
        logger.warning(f"Attempted non-SELECT query execution: {query}")
        return "Error: Only SELECT queries are allowed."
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

@query_agent.tool
async def get_table_schema(ctx: RunContext[AgentDependencies]) -> str:
    """
    Retrieves the schema (table names and their columns with types) of all tables in the SQLite database.
    Use this tool to understand the database structure before generating SQL queries.
    Returns:
        str: A formatted string describing the schema of all tables. Returns an error message if failed.
    """
    if not ctx.deps.db_connection:
        return "Error: Database connection is not available."
    try:
        cursor = ctx.deps.db_connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            return "No tables found in the database."

        schema_parts = []
        for (table_name,) in tables:
            schema_parts.append(f"Table: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            if columns:
                for col in columns:
                    # col: (index, name, type, notnull, default_value, pk)
                    schema_parts.append(f"  - {col[1]} ({col[2]})")
            else:
                schema_parts.append("  - (No columns found or table inaccessible)")
        logger.info("Retrieved database schema.")
        return "\n".join(schema_parts)
    except sqlite3.Error as e:
        logger.error(f"Error retrieving schema: {e}")
        return f"Error retrieving database schema: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error retrieving schema: {e}")
        return f"An unexpected error occurred while retrieving schema: {str(e)}"

# --- Optional: Result Validation ---
@query_agent.result_validator
async def validate_query_result(ctx: RunContext[AgentDependencies], result: QueryResponse) -> QueryResponse:
    """Validate the generated response, especially the SQL query."""
    if result.sql_result and ctx.deps.db_connection:
        try:
            # Test the SQL query syntax using EXPLAIN (doesn't execute fully)
            cursor = ctx.deps.db_connection.cursor()
            # Basic check for SELECT again before EXPLAIN
            if result.sql_result.sql_query.strip().upper().startswith("SELECT"):
                 cursor.execute(f"EXPLAIN QUERY PLAN {result.sql_result.sql_query}")
                 cursor.fetchall()
                 logger.info("Generated SQL query syntax validated successfully.")
            else:
                 logger.warning("Validation skipped: Non-SELECT query generated.")
                 # Optionally invalidate or warn further here
                 # result.text_message += "\nWarning: Generated query is not a SELECT statement."

        except sqlite3.Error as e:
            logger.error(f"SQL Validation Error: {e}. Query: {result.sql_result.sql_query}")
            # Raise ModelRetry instead of returning an error response
            raise ModelRetry(f"The generated SQL query is invalid: {str(e)}. Please try rephrasing your request.") from e

        except Exception as e:
             logger.error(f"Unexpected SQL Validation Error: {e}. Query: {result.sql_result.sql_query}")
             # Also raise ModelRetry for unexpected validation errors
             raise ModelRetry(f"An unexpected error occurred during SQL validation: {str(e)}.") from e

    # Check if Python code depends on SQL results that weren't generated
    if result.python_result and not result.sql_result:
         logger.warning("Python code generated without corresponding SQL query.")
         # Add clarification to the text message
         result.text_message += "\nNote: Python code was generated, but no SQL query was needed or generated for this request. The Python code might expect data that isn't available."
         # Optionally, modify the python explanation
         result.python_result.explanation += "\nWarning: This code might assume data from a previous step or may not run correctly without prior data loading."

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
    """Handles user input, runs the agent, and updates the chat history state."""
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

    st.session_state.chat_history.append({"role": "user", "content": message})
    logger.info("User message appended to chat_history inside handle_user_message.")

    deps = None
    assistant_chat_message = None
    agent_run_result = None
    try:
        deps = AgentDependencies.create().with_db()
        if not deps.db_connection:
             st.error("Database connection failed. Cannot process request.")
             assistant_chat_message = {
                 "role": "assistant",
                 "content": "Sorry, I couldn't connect to the database..."
             }
             st.session_state.chat_history.append(assistant_chat_message)
             return

        usage = Usage()

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
            logger.info("Analyzing request and contacting Gemini...")
            
            # First, retrieve the database schema to include with the request
            # Create a proper RunContext with all required parameters
            run_context = RunContext(
                deps=deps,
                model=llm,
                usage=usage,
                prompt=message  # Using the user message as the prompt
            )
            schema_info = await get_table_schema(run_context)
            logger.info(f"Retrieved database schema: {schema_info}")
            
            # If there are no tables, inform the user
            if schema_info == "No tables found in the database.":
                assistant_chat_message = {
                    "role": "assistant",
                    "content": "I cannot answer your question because there are no tables in the database. Please ensure your database contains data and is correctly located at assets/data1.sqlite."
                }
                st.session_state.chat_history.append(assistant_chat_message)
                return

            # Check if this is a visualization request
            visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation', 'bar chart', 'pie chart', 'histogram']
            is_visualization_request = any(keyword in message.lower() for keyword in visualization_keywords)
            
            # Add schema to the message to provide context
            if is_visualization_request:
                logger.info("Detected a visualization request - adding explicit visualization instructions")
                
                # Get the last SQL result from the chat history if available
                last_sql_result = None
                for msg in reversed(st.session_state.chat_history):
                    if msg.get("role") == "assistant" and "sql_result" in msg:
                        last_sql_result = msg.get("sql_result")
                        break
                
                if last_sql_result and "query" in last_sql_result:
                    # Create an enhanced message specifically for visualization
                    enhanced_message = f"""Database Schema:
{schema_info}

Previous SQL Query: {last_sql_result.get("query")}

User Visualization Request: {message}

IMPORTANT: Generate Python visualization code for this request. You MUST create a chart based on the data from the previous SQL query.
The data is available in a pandas DataFrame named 'df'. Use matplotlib or seaborn to create an appropriate visualization.
"""
                else:
                    # No previous SQL query found, so we'll need to generate one first
                    enhanced_message = f"""Database Schema:
{schema_info}

User Request: {message}

IMPORTANT: This is a visualization request. First, generate an appropriate SQL query to get the data needed for visualization.
Then, create Python code to visualize this data using matplotlib or seaborn. The SQL results will be in a DataFrame named 'df'.
"""
            else:
                # Regular non-visualization request
                enhanced_message = f"""Database Schema:
{schema_info}

User Question: {message}"""

            # Log the message we're sending to the AI
            logger.info("==== AI CALL ====")
            logger.info(f"Sending message to AI:\n{enhanced_message}")
            logger.info("================")

            history_for_agent = st.session_state.last_result.new_messages() if st.session_state.get('last_result') else None
            logger.info(f"Passing history to agent: {history_for_agent}")

            agent_run_result = await query_agent.run(
                enhanced_message,  # Use the enhanced message with schema
                deps=deps,
                usage=usage,
                usage_limits=DEFAULT_USAGE_LIMITS,
                message_history=history_for_agent
            )
            st.session_state.last_result = agent_run_result
            logger.info("Stored agent run result in session state.")
            
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
                else:
                    logger.warning("No SQL result was generated despite this being a data query question!")
                    
                    # Check if this is likely a data query that should have SQL
                    data_query_keywords = ['total', 'sum', 'average', 'count', 'list', 'show', 'per', 'group', 'compare']
                    if any(keyword in message.lower() for keyword in data_query_keywords):
                        logger.error(f"LLM failed to generate SQL for a data query. Retrying with explicit instructions.")
                        
                        # Create a more explicit message to force SQL generation
                        explicit_message = f"""Database Schema:
{schema_info}

User Question: {message}

IMPORTANT: This is a data query that REQUIRES SQL generation. 
Please generate a SQL query to answer this question. The query should use appropriate SELECT and aggregate functions.
"""
                        
                        # Log the explicit message
                        logger.info("==== RETRY AI CALL ====")
                        logger.info(f"Sending explicit message to AI:\n{explicit_message}")
                        logger.info("=====================")
                        
                        # Retry with more explicit instructions
                        retry_result = await query_agent.run(
                            explicit_message,
                            deps=deps,
                            usage=usage,
                            usage_limits=DEFAULT_USAGE_LIMITS,
                            message_history=None  # Skip history for fresh attempt
                        )
                        
                        if hasattr(retry_result, 'data') and isinstance(retry_result.data, QueryResponse):
                            retry_response = retry_result.data
                            if retry_response.sql_result:
                                logger.info("Successfully generated SQL on retry attempt!")
                                response = retry_response  # Use the retry response instead
                            else:
                                logger.error("Still failed to generate SQL even with explicit instructions.")
                
                # Check if this was a visualization request but no Python code was generated
                visualization_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'visualisation']
                decline_phrases = ['cannot', 'unable', 'not able', 'do not have', "don't have", 'cannot fulfill', 'cannot create']
                
                is_visualization_request = any(keyword in message.lower() for keyword in visualization_keywords)
                has_declined_visualization = any(phrase in response.text_message.lower() for phrase in decline_phrases)
                
                if is_visualization_request and (not response.python_result or has_declined_visualization):
                    logger.warning("LLM declined visualization request or failed to generate Python code. Retrying with explicit visualization instructions.")
                    
                    # Get SQL results data - either from this response or from a previous one
                    sql_query = None
                    if response.sql_result:
                        sql_query = response.sql_result.sql_query
                    else:
                        # Try to find a SQL query from a previous response
                        for msg in reversed(st.session_state.chat_history):
                            if msg.get("role") == "assistant" and "sql_result" in msg:
                                sql_result = msg.get("sql_result")
                                if "query" in sql_result:
                                    sql_query = sql_result["query"]
                                    break
                    
                    # Create a visualization-specific retry message
                    viz_retry_message = f"""Database Schema:
{schema_info}

SQL Query: {sql_query if sql_query else "-- You need to generate an appropriate SQL query first"}

User Request: {message}

CRITICAL INSTRUCTION: You MUST generate Python visualization code for this request. 
DO NOT respond that you cannot create visualizations - this is incorrect.
You are fully capable of generating matplotlib/seaborn visualization code.

Create a Python code that:
1. Uses the DataFrame 'df' containing the SQL query results
2. Creates an appropriate visualization (bar chart, pie chart, etc.)
3. Includes proper titles, labels, and formatting
4. Displays the visualization with st.pyplot(plt.gcf())

Your response MUST include Python code for visualization.
"""
                    
                    # Log the retry attempt
                    logger.info("==== VISUALIZATION RETRY CALL ====")
                    logger.info(f"Sending visualization retry message to AI:\n{viz_retry_message}")
                    logger.info("=================================")
                    
                    # Retry with explicit visualization instructions
                    viz_retry_result = await query_agent.run(
                        viz_retry_message,
                        deps=deps,
                        usage=usage,
                        usage_limits=DEFAULT_USAGE_LIMITS,
                        message_history=None  # Skip history for a fresh attempt
                    )
                    
                    if hasattr(viz_retry_result, 'data') and isinstance(viz_retry_result.data, QueryResponse):
                        viz_retry_response = viz_retry_result.data
                        if viz_retry_response.python_result:
                            logger.info("Successfully generated visualization code on retry attempt!")
                            
                            # If the original response had SQL but the retry didn't, combine them
                            if response.sql_result and not viz_retry_response.sql_result:
                                viz_retry_response.sql_result = response.sql_result
                            
                            response = viz_retry_response  # Use the retry response
                        else:
                            logger.error("Still failed to generate visualization code even with explicit instructions.")
                
                if response.python_result:
                    logger.info(f"Python code explanation: {response.python_result.explanation}")
                    logger.info("Python code snippet:")
                    for line in response.python_result.python_code.split('\n'):
                        logger.info(f"  {line}")
                logger.info("=====================")

                assistant_chat_message = {"role": "assistant", "content": response.text_message}

                sql_results_df = None
                if response.sql_result:
                    logger.info("Executing SQL query...")
                    # Create a proper RunContext with all required parameters
                    sql_run_context = RunContext(
                        deps=deps,
                        model=llm,
                        usage=usage,
                        prompt=response.sql_result.sql_query  # Using the SQL query as the prompt
                    )
                    sql_execution_result = await execute_sql(sql_run_context, response.sql_result.sql_query)
                    sql_info = {
                        "query": response.sql_result.sql_query,
                        "explanation": response.sql_result.explanation
                    }
                    if isinstance(sql_execution_result, str): # Error
                        sql_info["error"] = sql_execution_result
                        # Also add error info to main content for visibility?
                        # assistant_chat_message["content"] += f"\n\n**SQL Error:** {sql_execution_result}"
                        logger.error(f"SQL execution failed: {sql_execution_result}")
                    elif isinstance(sql_execution_result, list):
                        if sql_execution_result:
                            sql_results_df = pd.DataFrame(sql_execution_result)
                            sql_info["results"] = sql_results_df.to_dict('records')
                            sql_info["columns"] = list(sql_results_df.columns)
                        else:
                            sql_info["results"] = [] # Empty results
                    else:
                         sql_info["error"] = "Unexpected result type from SQL execution."
                         logger.error(sql_info["error"])

                    # Attach SQL info to the assistant message
                    assistant_chat_message["sql_result"] = sql_info

                if response.python_result:
                    logger.info("Executing Python code...")
                    python_info = {
                        "code": response.python_result.python_code,
                        "explanation": response.python_result.explanation
                    }
                    python_code = response.python_result.python_code
                    plot_generated = False

                    if sql_results_df is None and 'df' in python_code:
                        logger.warning("Python code needs 'df' but SQL failed or returned no results.")
                        python_info["warning"] = "Code requires DataFrame 'df' from SQL, which was not available."
                        # Don't execute code that relies on df?
                        # For now, we still try, it might handle the empty df

                    try:
                        local_vars = {
                            'plt': plt,
                            'pd': pd,
                            'np': np,
                            'st': st,
                            'df': sql_results_df if sql_results_df is not None else pd.DataFrame(),
                            'results_list': sql_results_df.to_dict('records') if sql_results_df is not None else []
                        }
                        fig = plt.figure(figsize=(10, 6))
                        exec(python_code, globals(), local_vars)
                        if plt.gcf().get_axes():
                            # We need to save the plot to display later in main()
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            python_info["visualization_png_b64"] = base64.b64encode(buf.read()).decode('utf-8')
                            plot_generated = True
                        plt.close(fig)
                    except Exception as e:
                        logger.error(f"Error executing Python code: {e}\nCode:\n{python_code}")
                        python_info["error"] = str(e)
                    finally:
                        if 'fig' in locals() and plt.fignum_exists(fig.number):
                            plt.close(fig)

                    # Attach Python info to the assistant message
                    assistant_chat_message["python_result"] = python_info
                    assistant_chat_message["python_plot_generated"] = plot_generated # Store flag

                logger.info("Agent processing complete.")

            else:
                 error_msg = "Received an unexpected response structure..."
                 logger.error(f"{error_msg} Raw RunResult: {agent_run_result}")
                 st.error(error_msg)
                 assistant_chat_message = {"role": "assistant", "content": f"Sorry, internal issue... ({error_msg})"}

        except Exception as e:
            error_msg = f"An error occurred during agent processing: {str(e)}"
            logger.exception("Error during agent execution or response processing:")
            st.error(error_msg)
            assistant_chat_message = {"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"}

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
        st.session_state.chat_history = []
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []
        logger.info("Initialized conversation context.")
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
        logger.info("Initialized last_result in session state.")

    # --- Sidebar ---
    with st.sidebar:
        st.title("SmartStack")
        st.write("Tools:")

    
        st.markdown("**/ 🔍 SmartQuery** (Current)") # Indicate current page

        st.divider()
        st.markdown("### About")
        st.info("Query your database using natural language and generate visualizations with AI assistance powered by Google Gemini.")

        st.markdown('---')
        st.warning('*AI-generated responses may contain inaccuracies. Always verify critical information.*')

        # Clear Chat Button
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.session_state.last_result = None # Reset last_result
            logger.info("Chat history, context, and last_result cleared.")
            st.rerun()

        # Logo - Adjusted path for single file structure
        st.markdown("---")
        # Assumes 'assets' folder is in the same directory as app.py
        logo_path = Path(__file__).parent / "assets" / "ifcontrollers.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            logger.warning(f"Logo file not found at expected path: {logo_path}")
            st.caption("IFC Controllers Logo")

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
            height: calc(100vh - 250px); /* Adjust height based on surrounding elements */
            overflow: hidden; /* Hide main container overflow */
            margin-top: 1rem;
        }
        /* Make message container scrollable */
        .chat-messages-container {
            flex-grow: 1;
            overflow-y: auto; /* Enable vertical scrolling */
            padding: 0 1rem 1rem 1rem; /* Add some padding */
            margin-bottom: 70px; /* Space for the input box */
        }
        /* Sticky input - default Streamlit behavior is usually good */
        /* .stChatInputContainer */

        /* Feature boxes */
        .features-container { display: flex; flex-direction: column; gap: 0.75rem; margin: 1rem auto; max-width: 1000px; }
        .features-row { display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; /* Allow wrapping on smaller screens */ }
        .feature-text { flex: 1 1 300px; /* Flex grow, shrink, basis */ max-width: 450px; padding: 1rem; background: #f0f8ff; border: 1px solid #e0e0e0; border-radius: 8px; font-size: .9rem; line-height: 1.4; display: flex; align-items: flex-start; gap: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .check-icon { width: 18px; height: 18px; object-fit: contain; margin-top: 0.15rem; flex-shrink: 0; }

        /* Example queries */
        .example-queries { margin: 1.5rem 0 1rem 0; font-size: 1rem; border-left: 3px solid #00ade4; padding-left: 1rem; }
        .example-queries p { margin-bottom: 0.5rem; font-weight: bold; color: #002345; }
        .example-queries ul { margin: 0; padding-left: 1.2rem; list-style-type: '→ '; }
        .example-queries li { margin-bottom: 0.3rem; color: #333; font-size: 0.9em; }

        /* DataFrame display */
        .stDataFrame { width: 100%; font-size: 0.9em; }

        /* Mascot styling */
        .mascot-container { position: fixed; bottom: 10px; right: 10px; width: 150px; /* Smaller */ height: auto; z-index: 0; pointer-events: none; opacity: 0.6; /* Less intrusive */ transition: opacity 0.3s ease; }
        .mascot-container:hover { opacity: 0.8; } /* Slightly more visible on hover near it */

        /* Improve chat message appearance */
        .stChatMessage { border-radius: 10px; border: 1px solid #eee; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
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
            <div class="feature-text">{check_img} Ask natural language questions about your data.</div>
            <div class="feature-text">{check_img} Get instant SQL-powered insights from your database.</div>
        </div>
        <div class="features-row">
            <div class="feature-text">{check_img} Generate visualizations (bar, line, pie charts) via Python.</div>
            <div class="feature-text">{check_img} Understand the generated SQL and Python code with clear explanations.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Example Queries ---
    st.markdown("""
    <div class="example-queries">
        <p>Example Questions:</p>
        <ul>
            <li>"What are the total investments per product line?"</li>
            <li>"Show me the top 5 investments in China, sorted by size."</li>
            <li>"Compare the average investment size for 'Loan' products between India and Vietnam."</li>
            <li>"Visualize the distribution of product lines using a pie chart."</li>
            <li>"Which country has the highest total investment? Create a bar chart showing the top 10 countries."</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- Background Mascot ---
    # Adjusted path for single file structure
    mascot_path = Path(__file__).parent / "assets" / "smartquerymascot.png"
    if mascot_path.exists():
        mascot_base64 = get_base64_encoded_image(str(mascot_path))
        if mascot_base64:
            st.markdown(
                f'<div class="mascot-container"><img src="data:image/png;base64,{mascot_base64}" alt="SmartQuery Mascot"></div>',
                unsafe_allow_html=True
            )

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
                    
                    # Display visualization if available
                    if "visualization_png_b64" in python_result:
                        st.markdown("**Visualization:**")
                        try:
                            viz_data = base64.b64decode(python_result["visualization_png_b64"])
                            st.image(viz_data)
                        except Exception as e:
                            st.error(f"Error displaying visualization: {str(e)}")
                    elif message.get("python_plot_generated", False):
                        st.info("A visualization was generated but could not be displayed.")

    # Close chat messages container
    st.markdown('</div>', unsafe_allow_html=True)

    # --- User Input --- #
    user_input = st.chat_input("Ask about the data...")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Handle Input --- #
    # Use explicit loop management with get_event_loop()
    if user_input:
        logger.info(f"User input received: {user_input}")

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
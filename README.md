# SmartQuery

SmartQuery is an AI-powered database analysis tool built with Streamlit and Google Gemini. It allows users to query databases using natural language and generate visualizations without writing SQL or Python code.

Try it out at: https://smartquery.streamlit.app/


## Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **SQL Generation**: Automatically converts natural language to SQL queries
- **Data Visualization**: Creates charts and graphs based on query results
- **Explanation**: Provides clear explanations of generated SQL and Python code
- **Interactive Interface**: User-friendly chat interface for data exploration

## Installation

### Prerequisites

- Python 3.8+
- SQLite database (default path: `assets/data1.sqlite`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smartquery.git
   cd smartquery
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `secrets.toml` file in the `.streamlit` directory:
   ```toml
   GOOGLE_API_KEY = "your_google_api_key_here"
   GEMINI_MODEL = "gemini-2.0-flash"  # or another Gemini model
   DATABASE_PATH = "assets/data1.sqlite"
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Begin asking questions about your data:
   - "What are the total investments per product line?"
   - "Show me the top 5 investments in China, sorted by size."
   - "Compare the average investment size for 'Loan' products between India and Vietnam."
   - "Visualize the distribution of product lines using a pie chart."
   - "Which country has the highest total investment? Create a bar chart showing the top 10 countries."

## How It Works

1. **User Input**: The user enters a natural language question about their data
2. **Database Schema Analysis**: The system retrieves the database schema to understand available tables and columns
3. **AI Processing**: Google Gemini processes the query and generates appropriate SQL and/or Python code
4. **SQL Execution**: The generated SQL query is executed against the database
5. **Visualization**: If requested, Python code is executed to create visualizations using matplotlib/seaborn
6. **Response**: Results are displayed in the chat interface, including data tables and visualizations

## Dependencies

- `streamlit`: Web application framework
- `google-generativeai`: Google Gemini AI SDK
- `pydantic-ai`: Agent framework for structured AI interactions
- `pandas`: Data manipulation and analysis
- `matplotlib`: Data visualization
- `numpy`: Numerical computing
- `sqlite3`: Database interface

## Project Structure

- `app.py`: Main application file
- `assets/`: Contains static assets and database
  - `data1.sqlite`: Default SQLite database
  - `smartquerymascot.png`: Application mascot/logo
  - `ifcontrollers.png`: Company logo
  - `correct.png`: UI element for feature highlights

## Troubleshooting

- **Database Connection Issues**: Ensure your SQLite database exists at the specified path
- **API Key Errors**: Verify your Google API key is correctly set in the secrets.toml file
- **Visualization Errors**: Check that matplotlib and its dependencies are properly installed

## License

[Specify your license here]

## Acknowledgements

- Google Gemini for AI capabilities
- Streamlit for the web application framework
- [Any other acknowledgements] 
# SmartQuery

SmartQuery is an AI-powered database analysis tool built with Streamlit and Google Gemini. It allows users to query databases using natural language and generate visualizations without writing SQL or Python code.

Try it out at: https://smartquery.streamlit.app/


## Features

- **Natural Language Queries**: Ask questions about your data in plain English.
- **Smart Request Routing**: An Orchestrator agent routes requests for general chat, database queries, or visualization.
- **Automatic Database Identification**: Identifies the most relevant database based on your query and metadata, with user confirmation if needed.
- **Interactive Table Selection**: Suggests relevant tables for your query and allows you to confirm or modify the selection.
- **Intelligent SQL Generation**: Converts natural language to SQL, leveraging pruned schemas and metadata.
- **Query Modification**: Easily modify previous queries with follow-up instructions.
- **Data Visualization**: An expert agent recommends and generates appropriate charts (bar, line, pie, etc.) based on the query results and user request.
- **Python Data Preparation**: Handles simple data preparation steps if needed before visualization.
- **Explanation**: Provides clear explanations of generated SQL and suggested visualizations.
- **Contextual Conversation**: Maintains conversation history for follow-up questions and context.
- **Interactive Interface**: User-friendly chat interface for data exploration.

## Installation

### Prerequisites

- Python 3.8+
- SQLite database (default path: `assets/dbs/db.sqlite`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ritwikathore/smartquery.git
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
   DATABASE_PATH = "assets/dbs/db.sqlite"
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Begin asking questions about your data:
   - "Get me the top 10 countries with the highest count of advisory projects approved in 2024"
   - "Show me the sum of IBRD loans to India approved since 2020 per year"
   - "Compare the average IFC investment size for 'Loan' products between Nepal and Bhutan."
   - "What is the total gross guarantee exposure for MIGA in the Tourism sector in Senegal?"
   - "Give me the top 10 IFC equity investments from China"
   - "Get me the status of all IDA Projects approved in 2020 to St. Lucia."
   - (After getting results) "Can you show that as a bar chart?"
   - (After getting results) "Now add the project sector to the results"

## How It Works

SmartQuery employs a multi-agent system powered by Google Gemini to process user requests:

1.  **User Input**: The user enters a natural language question via the Streamlit chat interface.
2.  **Modification Check**: The system first checks if the query is a modification of the *last* SQL query. If so, it directly calls the SQL Query Assistant with the previous context.
3.  **Follow-up Visualization Check**: If it's not a modification, it checks if it's a simple request to visualize the *last* results. If so, it calls the Visualization Expert Agent directly.
4.  **Orchestration**: For new queries or general chat, the **Orchestrator Agent** classifies the user's intent (`assistant`, `database_query`, or `visualization`).
    *   If `assistant`, it responds directly.
    *   If `visualization` (and the query implies data retrieval is also needed), it sets a flag and proceeds to the database query flow.
    *   If `database_query`, it proceeds to the database query flow.
5.  **Database Classification**: The **Database Classifier Agent** analyzes the query and database metadata (descriptions, tables, columns) to determine the target database. It prompts the user for confirmation if unsure.
6.  **Table Selection**: The **Table Selection Agent** analyzes the query and table descriptions for the chosen database, suggesting the most relevant tables. It prompts the user to confirm or modify the selection.
7.  **Schema Pruning**: The **Column Pruning Agent** receives the full schema for the confirmed tables and prunes it down to only the columns likely needed for the specific query, improving efficiency and accuracy.
8.  **SQL Generation & Execution**: The **SQL Query Assistant** takes the user query and the pruned schema to generate the final SQL query. It utilizes tools to execute the SQL (`execute_sql`) against the database and can fetch metadata (`get_metadata_info`) if needed. It can also call a **Python Agent** (`call_python_agent`) for complex data manipulations (though simple preparation might be handled directly).
9.  **Visualization Recommendation & Generation**: If visualization was requested (either initially or as a follow-up), the **Visualization Expert Agent** analyzes the resulting data (DataFrame), the user query, and schema to determine the best chart type and parameters (`x_column`, `y_column`, `title`, etc.). The recommended chart is then generated and displayed.
10. **Response**: The final response, including text explanations, the SQL query, query results (as a table), and any generated visualizations, is displayed in the chat interface. Context (like the last SQL query, schema, and data) is stored for potential follow-up modifications or visualizations.

## Dependencies

- `streamlit`: Web application framework
- `google-generativeai`: Google Gemini AI SDK
- `pydantic-ai`: Agent framework for structured AI interactions
- `pydantic`: Data validation
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Data visualization (used indirectly by Streamlit charts)
- `plotly`: Data visualization (especially for pie charts)
- `sqlite3`: Database interface
- `nest_asyncio`: Allows nested asyncio event loops (for Streamlit compatibility with Gemini)
- `langchain`: For conversational memory (`ConversationBufferMemory`)

## Project Structure

- `app.py`: Main application file
- `assets/`: Contains static assets and database
  - `db.sqlite`: Default SQLite database
  - `correct.png`: UI element for feature highlights

## Troubleshooting

- **Database Connection Issues**: Ensure your SQLite database exists at the specified path
- **API Key Errors**: Verify your Google API key is correctly set in the secrets.toml file
- **Visualization Errors**: Check that matplotlib and its dependencies are properly installed

## License

# Creative Commons Attribution-NonCommercial 1.0

CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE LEGAL SERVICES. DISTRIBUTION OF THIS DRAFT LICENSE DOES NOT CREATE AN ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES REGARDING THE INFORMATION PROVIDED, AND DISCLAIMS LIABILITY FOR DAMAGES RESULTING FROM ITS USE.

License

THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS AUTHORIZED UNDER THIS LICENSE IS PROHIBITED.

BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

1. Definitions

     a. "Collective Work" means a work, such as a periodical issue, anthology or encyclopedia, in which the Work in its entirety in unmodified form, along with a number of other contributions, constituting separate and independent works in themselves, are assembled into a collective whole. A work that constitutes a Collective Work will not be considered a Derivative Work (as defined below) for the purposes of this License.

      b. "Derivative Work" means a work based upon the Work or upon the Work and other pre-existing works, such as a translation, musical arrangement, dramatization, fictionalization, motion picture version, sound recording, art reproduction, abridgment, condensation, or any other form in which the Work may be recast, transformed, or adapted, except that a work that constitutes a Collective Work will not be considered a Derivative Work for the purpose of this License.

     c. "Licensor" means the individual or entity that offers the Work under the terms of this License.

     d. "Original Author" means the individual or entity who created the Work.

     e. "Work" means the copyrightable work of authorship offered under the terms of this License.

     f. "You" means an individual or entity exercising rights under this License who has not previously violated the terms of this License with respect to the Work, or who has received express permission from the Licensor to exercise rights under this License despite a previous violation.

2. Fair Use Rights. Nothing in this license is intended to reduce, limit, or restrict any rights arising from fair use, first sale or other limitations on the exclusive rights of the copyright owner under copyright law or other applicable laws.

3. License Grant. Subject to the terms and conditions of this License, Licensor hereby grants You a worldwide, royalty-free, non-exclusive, perpetual (for the duration of the applicable copyright) license to exercise the rights in the Work as stated below:

     a. to reproduce the Work, to incorporate the Work into one or more Collective Works, and to reproduce the Work as incorporated in the Collective Works;

     b. to create and reproduce Derivative Works;

     c. to distribute copies or phonorecords of, display publicly, perform publicly, and perform publicly by means of a digital audio transmission the Work including as incorporated in Collective Works;

     d. to distribute copies or phonorecords of, display publicly, perform publicly, and perform publicly by means of a digital audio transmission Derivative Works;

The above rights may be exercised in all media and formats whether now known or hereafter devised. The above rights include the right to make such modifications as are technically necessary to exercise the rights in other media and formats. All rights not expressly granted by Licensor are hereby reserved.

4. Restrictions. The license granted in Section 3 above is expressly made subject to and limited by the following restrictions:

     a. You may distribute, publicly display, publicly perform, or publicly digitally perform the Work only under the terms of this License, and You must include a copy of, or the Uniform Resource Identifier for, this License with every copy or phonorecord of the Work You distribute, publicly display, publicly perform, or publicly digitally perform. You may not offer or impose any terms on the Work that alter or restrict the terms of this License or the recipients' exercise of the rights granted hereunder. You may not sublicense the Work. You must keep intact all notices that refer to this License and to the disclaimer of warranties. You may not distribute, publicly display, publicly perform, or publicly digitally perform the Work with any technological measures that control access or use of the Work in a manner inconsistent with the terms of this License Agreement. The above applies to the Work as incorporated in a Collective Work, but this does not require the Collective Work apart from the Work itself to be made subject to the terms of this License. If You create a Collective Work, upon notice from any Licensor You must, to the extent practicable, remove from the Collective Work any reference to such Licensor or the Original Author, as requested. If You create a Derivative Work, upon notice from any Licensor You must, to the extent practicable, remove from the Derivative Work any reference to such Licensor or the Original Author, as requested.

     b. You may not exercise any of the rights granted to You in Section 3 above in any manner that is primarily intended for or directed toward commercial advantage or private monetary compensation. The exchange of the Work for other copyrighted works by means of digital file-sharing or otherwise shall not be considered to be intended for or directed toward commercial advantage or private monetary compensation, provided there is no payment of any monetary compensation in connection with the exchange of copyrighted works.

     c. If you distribute, publicly display, publicly perform, or publicly digitally perform the Work or any Derivative Works or Collective Works, You must keep intact all copyright notices for the Work and give the Original Author credit reasonable to the medium or means You are utilizing by conveying the name (or pseudonym if applicable) of the Original Author if supplied; the title of the Work if supplied; in the case of a Derivative Work, a credit identifying the use of the Work in the Derivative Work (e.g., "French translation of the Work by Original Author," or "Screenplay based on original Work by Original Author"). Such credit may be implemented in any reasonable manner; provided, however, that in the case of a Derivative Work or Collective Work, at a minimum such credit will appear where any other comparable authorship credit appears and in a manner at least as prominent as such other comparable authorship credit.

5. Representations, Warranties and Disclaimer

By offering the Work for public release under this License, Licensor represents and warrants that, to the best of Licensor's knowledge after reasonable inquiry: Licensor has secured all rights in the Work necessary to grant the license rights hereunder and to permit the lawful exercise of the rights granted hereunder without You having any obligation to pay any royalties, compulsory license fees, residuals or any other payments; The Work does not infringe the copyright, trademark, publicity rights, common law rights or any other right of any third party or constitute defamation, invasion of privacy or other tortious injury to any third party. EXCEPT AS EXPRESSLY STATED IN THIS LICENSE OR OTHERWISE AGREED IN WRITING OR REQUIRED BY APPLICABLE LAW, THE WORK IS LICENSED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES REGARDING THE CONTENTS OR ACCURACY OF THE WORK.

6. Limitation on Liability. EXCEPT TO THE EXTENT REQUIRED BY APPLICABLE LAW, AND EXCEPT FOR DAMAGES ARISING FROM LIABILITY TO A THIRD PARTY RESULTING FROM BREACH OF THE WARRANTIES IN SECTION 5, IN NO EVENT WILL LICENSOR BE LIABLE TO YOU ON ANY LEGAL THEORY FOR ANY SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR EXEMPLARY DAMAGES ARISING OUT OF THIS LICENSE OR THE USE OF THE WORK, EVEN IF LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

7. Termination

     a. This License and the rights granted hereunder will terminate automatically upon any breach by You of the terms of this License. Individuals or entities who have received Derivative Works or Collective Works from You under this License, however, will not have their licenses terminated provided such individuals or entities remain in full compliance with those licenses. Sections 1, 2, 5, 6, 7, and 8 will survive any termination of this License.

     b. Subject to the above terms and conditions, the license granted here is perpetual (for the duration of the applicable copyright in the Work). Notwithstanding the above, Licensor reserves the right to release the Work under different license terms or to stop distributing the Work at any time; provided, however that any such election will not serve to withdraw this License (or any other license that has been, or is required to be, granted under the terms of this License), and this License will continue in full force and effect unless terminated as stated above.

8. Miscellaneous

     a. Each time You distribute or publicly digitally perform the Work or a Collective Work, the Licensor offers to the recipient a license to the Work on the same terms and conditions as the license granted to You under this License.

     b. Each time You distribute or publicly digitally perform a Derivative Work, Licensor offers to the recipient a license to the original Work on the same terms and conditions as the license granted to You under this License.

     c. If any provision of this License is invalid or unenforceable under applicable law, it shall not affect the validity or enforceability of the remainder of the terms of this License, and without further action by the parties to this agreement, such provision shall be reformed to the minimum extent necessary to make such provision valid and enforceable.

     d. No term or provision of this License shall be deemed waived and no breach consented to unless such waiver or consent shall be in writing and signed by the party to be charged with such waiver or consent.

     e. This License constitutes the entire agreement between the parties with respect to the Work licensed here. There are no understandings, agreements or representations with respect to the Work not specified here. Licensor shall not be bound by any additional provisions that may appear in any communication from You. This License may not be modified without the mutual written agreement of the Licensor and You.

Creative Commons is not a party to this License, and makes no warranty whatsoever in connection with the Work. Creative Commons will not be liable to You or any party on any legal theory for any damages whatsoever, including without limitation any general, special, incidental or consequential damages arising in connection to this license. Notwithstanding the foregoing two (2) sentences, if Creative Commons has expressly identified itself as the Licensor hereunder, it shall have all rights and obligations of Licensor.

Except for the limited purpose of indicating to the public that the Work is licensed under the CCPL, neither party will use the trademark "Creative Commons" or any related trademark or logo of Creative Commons without the prior written consent of Creative Commons. Any permitted use will be in compliance with Creative Commons' then-current trademark usage guidelines, as may be published on its website or otherwise made available upon request from time to time.

Creative Commons may be contacted at http://creativecommons.org/.

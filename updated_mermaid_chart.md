flowchart TD
    %% Main User Interface Flow
    subgraph User_Interaction_Streamlit_UI["User Interaction (Streamlit UI)"]
        A[User Enters Query] --> B[st.chat_input]
        B --> C[Append User Msg to History]
        C --> D[Call run_async handle_user_message]
    end

    %% Orchestration & Routing
    subgraph Handle_User_Message_Routing["handle_user_message: Orchestrator Routing"]
        D_In(From UI: handle_user_message) --> G[Run Orchestrator Agent]
        G --> H{Orchestrator Action?}
        H -- assistant --> Assist_Path["To Final Output - Assistant Response"]
        H -- visualization --> FollowUpViz_Path["To Follow-up Viz Flow"]
        H -- database_query --> I{Is Modification?}
        I -- Yes --> Mod_Path["To Modification Flow"]
        I -- No --> DB_Setup_Path["To DB/Table Setup Flow"]
    end

    %% Modification Flow for Existing Queries
    subgraph Modification_Flow["Modification Flow (run_sql_modification)"]
        Mod_In("From Routing: Is Modification=True")
        Mod1["Retrieve Context<br>SQL, Schema, DB Key/Path"] --> Mod2[Connect DB]
        Mod2 --> Mod3[Instantiate SQL Query Agent]
        Mod3 --> Mod5["Run Agent w/ Modification Prompt<br>using get_metadata_info if needed"]
        Mod5 --> Mod6{Process QueryResponse}
        Mod6 --> Mod7[Execute New SQL]
        Mod7 --> Mod8["Store New Context / Results<br>Update last_sql_query, last_chartable_data"]
        Mod8 --> Mod9{"Visualization<br>Requested?"}
        Mod9 -- Yes --> Mod10[Call Visualization Agent]
        Mod10 --> Mod11[Package Result]
        Mod9 -- No --> Mod11
        Mod11 --> Mod12[Disconnect DB]
    end

    %% Follow-up Visualization Flow
    subgraph FollowUp_Visualization_Flow["Follow-up Visualization Flow"]
        FollowUpViz_In("From Routing: Action=visualization")
        FollowUpViz1["Retrieve Context<br>last_chartable_data"]
        FollowUpViz1 --> FollowUpViz2[Call Visualization Agent]
        FollowUpViz2 --> FollowUpViz3["Package Result<br>Append Chart"]
    end

    %% New Query Setup Flow
    subgraph New_Query_Setup_Flow["New Query Setup Flow"]
        %% Database Classification & Selection
        subgraph DB_Selection["Database Classification & Selection"]
            DB_Setup_In("From Routing: Is Modification=False")
            DBClass1[Load Metadata]
            DBClass2["Run DB Classifier Agent<br>identify_target_database"]
            DBClass3{"DB Identified?"}
            DBClass4[User DB Selection UI]
            DBClass5[Store DB Key & Path]

            DB_Setup_In --> DBClass1 --> DBClass2 --> DBClass3
            DBClass3 -- No --> DBClass4 --> DBClass5
            DBClass3 -- Yes --> DBClass5
        end

        %% Table Selection
        subgraph Table_Selection["Table Selection Process"]
            TableSel_In("From DB Confirmation")
            TableSel1["Run Table Selection Agent<br>run_table_selection_stage"]
            TableSel2[User Table Selection UI]
            TableSel3[Store Selected Tables]
            TableSel4[Continue to Main Processing]

            TableSel_In --> TableSel1 --> TableSel2 --> TableSel3 --> TableSel4
        end

        DBClass5 --> TableSel_In
    end

    %% Main Processing Flow
    subgraph Main_Processing["Main Processing"]
        MainProc_In("From Table Selection Confirmation")
        T["Retrieve State<br>DB Key/Path, Tables"]
        U[Format Schema for Selected Tables]
        U2[Run Column Pruning Agent]
        U3[Get Pruned Schema]
        V[Connect to DB]
        W1["Create SQL Query Agent<br>with Python Agent Tool"]
        W3[Run Agent w/ Pruned Schema]
        X{QueryResponse}
        X_Val["Validate QueryResult<br>Agent Config"]
        Y{Process QueryResponse}
        Z{"SQL Present?"}
        AA[Execute SQL]
        AB["Store Results<br>last_chartable_data, last_sql_query"]
        AC{"Python Code<br>Present?"}
        AD[Execute Python Code]
        AE[Update DataFrame]
        AF{"Visualization<br>Requested?"}
        AG[Call Visualization Agent]
        AH["Package Final Message<br>Text, SQL, Python, Chart"]
        AI[Disconnect DB]

        MainProc_In --> T --> U --> U2 --> U3 --> V --> W1 --> W3 --> X --> X_Val --> Y
        Y --> Z
        Z -- Yes --> AA --> AB
        Z -- No --> AB
        AB --> AC
        AC -- Yes --> AD --> AE
        AC -- No --> AE
        AE --> AF
        AF -- Yes --> AG --> AH
        AF -- No --> AH
        AH --> AI
    end

    %% Agent Subsystems
    subgraph Agent_Subsystems["Agent Subsystems"]
        subgraph PythonAgent["Python Agent"]
            PY1["Receives Request<br>complex manipulation"]
            PY2{"Visualization<br>Request?"}
            PY3[Call visualize_last_dataframe Tool]
            PY4["Execute Code<br>pandas/numpy"]
            PY5["Return Result<br>PythonAgentResult"]

            PY1 --> PY2
            PY2 -- Yes --> PY3 --> PY5
            PY2 -- No --> PY4 --> PY5
        end

        subgraph VisualizationAgent["Visualization Agent"]
            Viz1["Receives DataFrame,<br>User Query, Schema"]
            Viz2[Instantiate Agent]
            Viz4["Run Agent<br>selects cols, chart type"]
            Viz5["Return Result<br>VisualizationAgentResult"]

            Viz1 --> Viz2 --> Viz4 --> Viz5
        end
    end

    %% Final Output in Streamlit UI
    subgraph Final_Output_Streamlit_UI["Final Output (Streamlit UI)"]
        AJ_In("From Processing Flows")
        AJ[Append Result to History]
        AK[Clear Pending State]
        End[Update Chat Display]

        AJ_In --> AJ --> AK --> End
    end

    %% Cross-flow connections
    D --> D_In
    Mod12 --> AJ_In
    FollowUpViz3 --> AJ_In
    AI --> AJ_In
    Assist_Path --> AJ_In
    FollowUpViz_Path --> FollowUpViz_In
    Mod_Path --> Mod_In
    DB_Setup_Path --> DB_Setup_In
    TableSel4 --> MainProc_In

    %% Styling enhancements
    classDef agent fill:#D6EAF8,stroke:#2874A6,stroke-width:1px
    classDef db fill:#E8F8F5,stroke:#138D75,stroke-width:1px
    classDef ui fill:#EAECEE,stroke:#566573,stroke-width:1px
    classDef flow fill:#FDEDEC,stroke:#C0392B,stroke-width:1px
    classDef query fill:#D5F5E3,stroke:#1E8449,stroke-width:1px
    classDef decision fill:#F9E79F,stroke:#D4AC0D,stroke-width:1px

    class G,DBClass2,TableSel1,U2,W1,W3,Mod3,Mod5,PY1,Viz2,Viz4 agent
    class Mod2,V,AI,Mod12 db
    class User_Interaction_Streamlit_UI,Final_Output_Streamlit_UI,DBClass4,TableSel2 ui
    class Handle_User_Message_Routing,Modification_Flow,FollowUp_Visualization_Flow,New_Query_Setup_Flow,Main_Processing,Agent_Subsystems,DB_Selection,Table_Selection flow
    class H,I,DBClass3,Z,AC,AF,Mod9,PY2,Mod6,X,X_Val,Y decision
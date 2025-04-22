flowchart TD
    subgraph User_Interaction_Streamlit_UI["User Interaction (Streamlit UI)"]
        A[User Enters Query] --> B[st.chat_input]
        B --> C[Append User Msg to History]
        C --> D[Call run_async handle_user_message]
    end

    subgraph Handle_User_Message_Routing["handle_user_message: Initial Routing"]
        D_In(From UI)
        E{"Modification Request?"}
        F{"Follow-up Chart Request?"}
        G[Instantiate/Run Orchestrator Agent]
        H{"Orchestrator Action?"}

        D_In --> E
        E -- Yes --> Mod_Path[To Modification Flow]
        E -- No --> F
        F -- Yes --> FollowUpViz_Path[To Follow-up Viz Flow]
        F -- No --> G
        G --> H
        H -- assistant --> Assist_Path[Respond Directly]
        H -- visualization --> VizIntent_Path[Set Viz Flag / Go to DB Flow]
        H -- database_query --> DB_Path[Go to DB Flow]
    end

    subgraph Modification_Flow["Modification Flow (run_sql_modification)"]
        Mod_In(From Routing)
        Mod1[Retrieve Context] --> Mod2[Connect DB]
        Mod2 --> Mod3[Instantiate SQL Query Agent]
        Mod3 --> Mod4[Register Tools]
        Mod4 --> Mod5[Run Agent w/ Modification Prompt]
        Mod5 --> Mod6{Process QueryResponse}
        Mod6 --> Mod7[Execute New SQL]
        Mod7 --> Mod8[Store New Context / Results]
        Mod8 --> Mod9{"Visualization Requested?"}
        Mod9 -- Yes --> Mod10[Call Visualization Agent]
        Mod10 --> Mod11[Package Result]
        Mod9 -- No --> Mod11
        Mod11 --> Mod12[Disconnect DB]
    end

    subgraph FollowUp_Visualization_Flow["Follow-up Visualization Flow"]
        FollowUpViz_In(From Routing)
        FollowUpViz1[Retrieve Context]
        FollowUpViz2[Call Visualization Agent]
        FollowUpViz3[Package Result]
        
        FollowUpViz_In --> FollowUpViz1
        FollowUpViz1 --> FollowUpViz2
        FollowUpViz2 --> FollowUpViz3
    end

    subgraph Database_Query_Flow["Database Query Flow"]
        DB_In(From Routing or VizIntent)
        DB1[Load Metadata]
        DB2[Run DB Classifier Agent]
        DB3{"DB Identified?"}
        DB4[User DB Selection UI]
        DB5[Store DB Key]
        DB6[Run Table Selection]
        DB7[Run Table Selection Agent]
        DB8[User Table Selection UI]
        DB9[Store Tables]
        DB10[Continue After Confirmation]
        
        DB_In --> DB1
        DB1 --> DB2
        DB2 --> DB3
        DB3 -- No --> DB4
        DB4 --> DB5
        DB3 -- Yes --> DB5
        DB5 --> DB6
        DB6 --> DB7
        DB7 --> DB8
        DB8 --> DB9
        DB9 --> DB10
    end

    subgraph Main_Processing["Main Processing"]
        MainProc_In(From DB Query Flow)
        T[Retrieve State]
        U[Format Schema]
        U2[Run Column Pruning Agent]
        U3[Get Pruned Schema]
        V[Connect to DB]
        W1[Create SQL Query Agent]
        W2[Register Tools]
        W3[Run Agent]
        X{QueryResponse}
        X_Val[Validate QueryResult]
        Y{Process QueryResponse}
        Z{"SQL Present?"}
        AA[Execute SQL]
        AB[Store Results]
        AC{"Python Code?"}
        AD[Execute Python Code]
        AE[Update DataFrame]
        AF{"Visualization Requested?"}
        AG[Call Visualization Agent]
        AH[Package Final Message]
        AI[Disconnect DB]
        
        MainProc_In --> T --> U --> U2 --> U3 --> V --> W1 --> W2 --> W3 --> X --> X_Val --> Y
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

    subgraph Agent_Subsystems["Agent Subsystems"]
        subgraph PythonAgent["Python Agent"]
            PY1[Receives Request]
            PY2{"Visualization Request?"}
            PY3[Call visualize Tool]
            PY4[Execute Code]
            PY5[Return Result]
            
            PY1 --> PY2
            PY2 -- Yes --> PY3 --> PY5
            PY2 -- No --> PY4 --> PY5
        end
        
        subgraph VisualizationAgent["Visualization Agent"]
            Viz1[Receives DataFrame]
            Viz2[Instantiate Agent]
            Viz3[Register Tools]
            Viz4[Run Agent]
            Viz5[Return Result]
            
            Viz1 --> Viz2 --> Viz3 --> Viz4 --> Viz5
        end
    end

    subgraph Final_Output_Streamlit_UI["Final Output (Streamlit UI)"]
        AJ_In(From Processing)
        AJ[Append Result to History]
        AK[Clear Pending State]
        End[Update Chat Display]
        
        AJ_In --> AJ --> AK --> End
    end

    %% Connections between subgraphs
    D --> D_In
    Mod12 --> AJ_In
    FollowUpViz3 --> AJ_In
    AI --> AJ_In
    Assist_Path --> AJ
    VizIntent_Path --> DB_In
    DB_Path --> DB_In
    DB10 --> MainProc_In
    Mod_Path --> Mod_In
    FollowUpViz_Path --> FollowUpViz_In

    %% Simplified styling
    classDef agent fill:#D6EAF8
    classDef db fill:#E8F8F5
    classDef ui fill:#EAECEE
    classDef flow fill:#FDEDEC
    classDef query fill:#D5F5E3
    
    class G,DB2,DB7,U2,W1,W3,Mod3,Mod5 agent
    class Mod2,V,AI,Mod12 db
    class User_Interaction_Streamlit_UI,Final_Output_Streamlit_UI ui
    class E,F,H,DB3,Z,AC,AF,Mod9,PY2 query
    class Handle_User_Message_Routing,Modification_Flow,FollowUp_Visualization_Flow,Database_Query_Flow,Main_Processing,Agent_Subsystems flow
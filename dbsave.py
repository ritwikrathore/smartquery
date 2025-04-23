from itertools import count
from locale import currency
import pandas as pd
import sqlite3

# Load the excel     file
df_ibrd_lending = pd.read_csv('assets/csv_files/ibrd_statement_of_loans_and_guarantees_latest_available_snapshot_04-18-2025.csv')
df_ida_lending = pd.read_csv('assets/csv_files/ida_statement_of_credits_grants_and_guarantees_latest_available_snapshot_04-18-2025.csv')
df_ifc_investments = pd.read_csv('assets/csv_files/ifc_investment_services_projects_04-18-2025.csv')
df_ifc_advisory = pd.read_csv('assets/csv_files/ifc_advisory_services_projects_04-18-2025.csv')
df_miga_guarantees = pd.read_csv('assets/csv_files/miga_issued_projects_04-18-2025.csv')


df_ibrd_lending['End of Period'] = pd.to_datetime(df_ibrd_lending['End of Period'])
df_ibrd_lending['Agreement Signing Date'] = pd.to_datetime(df_ibrd_lending['Agreement Signing Date'])
df_ibrd_lending['Board Approval Date'] = pd.to_datetime(df_ibrd_lending['Board Approval Date'])
df_ibrd_lending['First Repayment Date'] = pd.to_datetime(df_ibrd_lending['First Repayment Date'])
df_ibrd_lending['Last Repayment Date'] = pd.to_datetime(df_ibrd_lending['Last Repayment Date'])
df_ibrd_lending['Last Disbursement Date'] = pd.to_datetime(df_ibrd_lending['Last Disbursement Date'])
df_ibrd_lending['Effective Date (Most Recent)'] = pd.to_datetime(df_ibrd_lending['Effective Date (Most Recent)'])
df_ibrd_lending['Closed Date (Most Recent)'] = pd.to_datetime(df_ibrd_lending['Closed Date (Most Recent)'])

# Original Principal Amount (US$)     float64
# Cancelled Amount (US$)              float64
# Undisbursed Amount (US$)            float64
# Disbursed Amount (US$)              float64
# Repaid to IBRD (US$)                float64
# Due to IBRD (US$)                   float64
# Exchange Adjustment (US$)           float64
# Borrower's Obligation (US$)         float64
# Sold 3rd Party (US$)                float64
# Repaid 3rd Party (US$)              float64
# Due 3rd Party (US$)                 float64
# Loans Held (US$)                    float64


#replace $ in column names with 'd'
df_ibrd_lending.columns = df_ibrd_lending.columns.str.replace('$', 'd', regex=False)

#convert to snake case lower and replace spaces with underscores and remove ()
df_ibrd_lending.columns = df_ibrd_lending.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

#drop currency_of_commitment column
df_ibrd_lending = df_ibrd_lending.drop(columns=['currency_of_commitment'])

# save to sqlite, create table name ibrd_lending if not exists, database name ibrd.db
df_ibrd_lending.to_sql('ibrd_lending', con=sqlite3.connect('assets/dbs/ibrd.db'), if_exists='replace', index=False)

# End of Period                       object
# Credit Number                       object
# Region                              object
# Country / Economy Code              object
# Country / Economy                   object
# Borrower                            object
# Credit Status                       object
# Service Charge Rate                float64
# currency of Commitment              object
# Project ID                          object
# Project Name                        object
# Original Principal Amount (US$)    float64
# Cancelled Amount (US$)             float64
# Undisbursed Amount (US$)           float64
# Disbursed Amount (US$)             float64
# Repaid to IDA (US$)                float64
# Due to IDA (US$)                   float64
# Exchange Adjustment (US$)          float64
# Borrowers Obligation (US$)         float64
# Sold 3rd Party (US$)               float64
# Repaid 3rd Party (US$)             float64
# Due 3rd Party (US$)                float64
# Credits Held (US$)                 float64
# First Repayment Date                object
# Last Repayment Date                 object
# Agreement Signing Date              object
# Board Approval Date                 object
# Effective Date (Most Recent)        object
# Closed Date (Most Recent)           object
# Last Disbursement Date              object
df_ida_lending['End of Period'] = pd.to_datetime(df_ida_lending['End of Period'])
df_ida_lending['First Repayment Date'] = pd.to_datetime(df_ida_lending['First Repayment Date'])
df_ida_lending['Last Repayment Date'] = pd.to_datetime(df_ida_lending['Last Repayment Date'])
df_ida_lending['Agreement Signing Date'] = pd.to_datetime(df_ida_lending['Agreement Signing Date'])
df_ida_lending['Board Approval Date'] = pd.to_datetime(df_ida_lending['Board Approval Date'])
df_ida_lending['Last Disbursement Date'] = pd.to_datetime(df_ida_lending['Last Disbursement Date'])
df_ida_lending['Effective Date (Most Recent)'] = pd.to_datetime(df_ida_lending['Effective Date (Most Recent)'])
df_ida_lending['Closed Date (Most Recent)'] = pd.to_datetime(df_ida_lending['Closed Date (Most Recent)'])

#rename columns to snake case lower and replace spaces with underscores and remove ()
df_ida_lending.columns = df_ida_lending.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

#replace $ with d in column names
df_ida_lending.columns = df_ida_lending.columns.str.replace('$', 'd', regex=False)

conn = sqlite3.connect('assets/dbs/ibrd.db')
#save to sqlite, create table name ida_lending if not exists, database name ibrd.db
df_ida_lending.to_sql('ida_lending', con=conn, if_exists='replace', index=False)
conn.close()


#IFC INVESTMENT SERVICES PROJECTS
# >>> df_ifc_investments.dtypes
# Date Disclosed                                               object
# Project Name                                                 object
# Document Type                                                object
# Project Number                                                int64
# Project Url                                                  object
# Product Line                                                 object
# Company Name                                                 object
# Country                                                      object
# IFC Country Code                                             object
# Industry                                                     object
# Environmental Category                                       object
# Department                                                   object
# Status                                                       object
# Projected Board Date                                         object
# IFC Approval Date                                            object
# IFC Signed Date                                              object
# IFC Invested Date                                            object
# IFC investment for Risk Management(Million - USD)           float64
# IFC investment for Guarantee(Million - USD)                 float64
# IFC investment for Loan(Million - USD)                      float64
# IFC investment for Equity(Million - USD)                    float64
# Total IFC investment as approved by Board(Million - USD)    float64
# WB Country Code                                              object
# As of Date                                                   object

df_ifc_investments['Date Disclosed'] = pd.to_datetime(df_ifc_investments['Date Disclosed'])
df_ifc_investments['Projected Board Date'] = pd.to_datetime(df_ifc_investments['Projected Board Date'])
df_ifc_investments['IFC Approval Date'] = pd.to_datetime(df_ifc_investments['IFC Approval Date'])
df_ifc_investments['IFC Signed Date'] = pd.to_datetime(df_ifc_investments['IFC Signed Date'])
df_ifc_investments['IFC Invested Date'] = pd.to_datetime(df_ifc_investments['IFC Invested Date'])
df_ifc_investments['As of Date'] = pd.to_datetime(df_ifc_investments['As of Date'])

#rename columns to snake case lower and replace spaces with underscores and remove ()
df_ifc_investments.columns = df_ifc_investments.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

#save to sqlite, create table name ifc_investments if not exists, database name ifc.db
conn = sqlite3.connect('assets/dbs/ifc.db') 
df_ifc_investments.to_sql('ifc_investments', con=conn, if_exists='replace', index=False)
conn.close()

#IFC ADVISORY SERVICES PROJECTS
# >>> df_ifc_advisory.dtypes
# Disclosure Date                object
# Project Number                  int64
# Project Name                   object
# Project URL                    object
# Country                        object
# IFC Country Code               object
# IFC Region                     object
# Business Line                  object
# Estimated Total Budget ($)    float64
# Department                    float64
# Status                         object
# IFC Approval Date              object
# Projected Start Date           object
# WB Country Code                object
# As of Date                     object

df_ifc_advisory['Disclosure Date'] = pd.to_datetime(df_ifc_advisory['Disclosure Date'])
df_ifc_advisory['IFC Approval Date'] = pd.to_datetime(df_ifc_advisory['IFC Approval Date'])
df_ifc_advisory['Projected Start Date'] = pd.to_datetime(df_ifc_advisory['Projected Start Date'])
df_ifc_advisory['As of Date'] = pd.to_datetime(df_ifc_advisory['As of Date'])

#rename columns to snake case lower and replace spaces with underscores and remove ()
df_ifc_advisory.columns = df_ifc_advisory.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

#save to sqlite, create table name ifc_advisory if not exists, database name ifc.db
conn = sqlite3.connect('assets/dbs/ifc.db')
df_ifc_advisory.to_sql('ifc_advisory', con=conn, if_exists='replace', index=False)
conn.close()

#MIGA ISSUED PROJECTS
# >>> df_miga_guarantees.dtypes
# Issued FY                      int64
# Project ID                     int64
# Project Name                  object
# Host Country                  object
# Host Country Code             object
# Host Region                   object
# Host Region Code              object
# Business Sector               object
# Product Line                  object
# Gross Issued count (US$)    float64
# Project Start Date            object
# Project Expiration Date       object
# Project Links                 object
# Issued Proj Count              int64

df_miga_guarantees['Project Start Date'] = pd.to_datetime(df_miga_guarantees['Project Start Date'])
df_miga_guarantees['Project Expiration Date'] = pd.to_datetime(df_miga_guarantees['Project Expiration Date'])

#rename columns to snake case lower and replace spaces with underscores and remove ()
df_miga_guarantees.columns = df_miga_guarantees.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '') 
#replace $ with d in column names
df_miga_guarantees.columns = df_miga_guarantees.columns.str.replace('$', 'd', regex=False)

#save to sqlite, create table name miga_guarantees if not exists, database name miga.db
conn = sqlite3.connect('assets/dbs/miga.db')
df_miga_guarantees.to_sql('miga_guarantees', con=conn, if_exists='replace', index=False)
conn.close()




# connect to ibrd.db and get all tables
conn = sqlite3.connect('assets/dbs/ibrd.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(tables.fetchall())

# get columns for each ibrd_lending table
ibrd_lending_columns = conn.execute("PRAGMA table_info(ibrd_lending)").fetchall()
print(ibrd_lending_columns)

#change 'guarantor_country_/_economy_code' to 'guarantor_country_economy_code'
# ALTER TABLE ibrd_lending RENAME COLUMN "guarantor_country_/_economy_code" TO "guarantor_country_economy_code";
conn.execute("ALTER TABLE ibrd_lending RENAME COLUMN 'guarantor_country_/_economy_code' TO 'guarantor_country_economy_code'")

#change 4, 'country_/_economy' to 'country_economy'
# ALTER TABLE ibrd_lending RENAME COLUMN "country_/_economy" TO "country_economy";
conn.execute("ALTER TABLE ibrd_lending RENAME COLUMN 'country_/_economy' TO 'country_economy'")

#change 3, 'country_/_economy_code' to 'country_economy_code'
# ALTER TABLE ibrd_lending RENAME COLUMN "country_/_economy_code" TO "country_economy_code";
conn.execute("ALTER TABLE ibrd_lending RENAME COLUMN 'country_/_economy_code' TO 'country_economy_code'")



#commit changes
conn.commit()


# get columns for each ida_lending table
ida_lending_columns = conn.execute("PRAGMA table_info(ida_lending)").fetchall()
print(ida_lending_columns)

#change 4, 'country_/_economy' to 'country_economy'
# ALTER TABLE ibrd_lending RENAME COLUMN "country_/_economy" TO "country_economy";
conn.execute("ALTER TABLE ida_lending RENAME COLUMN 'country_/_economy' TO 'country_economy'")

#change 3, 'country_/_economy_code' to 'country_economy_code'
# ALTER TABLE ibrd_lending RENAME COLUMN "country_/_economy_code" TO "country_economy_code";
conn.execute("ALTER TABLE ida_lending RENAME COLUMN 'country_/_economy_code' TO 'country_economy_code'")



conn.close()



#connect to ifc.db and get all tables
conn = sqlite3.connect('assets/dbs/ifc.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(tables.fetchall())


# ifc_investments
ifc_investments_columns = conn.execute("PRAGMA table_info(ifc_investments)").fetchall()
print(ifc_investments_columns)



# change 19, 'ifc_investment_for_loanmillion_-_usd' to 'ifc_investment_for_loan_million_usd'
conn.execute("ALTER TABLE ifc_investments RENAME COLUMN 'ifc_investment_for_loanmillion_-_usd' TO 'ifc_investment_for_loan_million_usd'")

#17, 'ifc_investment_for_risk_managementmillion_-_usd'
conn.execute("ALTER TABLE ifc_investments RENAME COLUMN 'ifc_investment_for_risk_managementmillion_-_usd' TO 'ifc_investment_for_risk_management_million_usd'")

#18, 'ifc_investment_for_guaranteemillion_-_usd'
conn.execute("ALTER TABLE ifc_investments RENAME COLUMN 'ifc_investment_for_guaranteemillion_-_usd' TO 'ifc_investment_for_guarantee_million_usd'")

#20, 'ifc_investment_for_equitymillion_-_usd'
conn.execute("ALTER TABLE ifc_investments RENAME COLUMN 'ifc_investment_for_equitymillion_-_usd' TO 'ifc_investment_for_equity_million_usd'")

#21, 'total_ifc_investment_as_approved_by_boardmillion_-_usd'
conn.execute("ALTER TABLE ifc_investments RENAME COLUMN 'total_ifc_investment_as_approved_by_boardmillion_-_usd' TO 'total_ifc_investment_as_approved_by_board_million_usd'")

# ifc_advisory
ifc_advisory_columns = conn.execute("PRAGMA table_info(ifc_advisory)").fetchall()
print(ifc_advisory_columns)

# estimated_total_budget_$
conn.execute("ALTER TABLE ifc_advisory RENAME COLUMN 'estimated_total_budget_$' TO 'estimated_total_budget_usd'")





































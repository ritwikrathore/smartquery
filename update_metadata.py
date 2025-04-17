import sqlite3
import json
import os
import argparse
import sys

def get_column_types_from_db(db_path, table_name):
    """Fetches column names and their data types from an SQLite table."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Use PRAGMA table_info to get column details
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns_info = cursor.fetchall()

        if not columns_info:
            print(f"Error: Table '{table_name}' not found or has no columns in {db_path}")
            return None

        # Return a dictionary of {column_name: column_type}
        # Column info tuple index: 1=name, 2=type
        return {info[1]: info[2] for info in columns_info}
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_create_table_statement(db_path, table_name):
    """Fetches the CREATE TABLE statement for a given table."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            print(f"Error: Could not retrieve CREATE TABLE statement for table '{table_name}'.")
            return None
    except sqlite3.Error as e:
        print(f"SQLite error while fetching CREATE statement: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_db_schema(db_path):
    """Fetches all table names and their column schemas from an SQLite database."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None

    conn = None
    schema = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()

        if not tables:
            print(f"Warning: No user tables found in database {db_path}")
            return {}

        # Get schema for each table
        for table_tuple in tables:
            table_name = table_tuple[0]
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns_info = cursor.fetchall()
            # Store as {column_name: column_type}
            # Use 'TEXT' as default if type is empty/null
            schema[table_name] = {
                info[1]: info[2] if info[2] else 'TEXT'
                for info in columns_info
            }
        return schema

    except sqlite3.Error as e:
        print(f"SQLite error while reading schema from {db_path}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def update_metadata_for_db(metadata_path, db_path):
    """Loads metadata JSON, updates or adds the entry for the given db_path based on its DB schema, and saves it back."""
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file {metadata_path}: {e}")
        return
    except IOError as e:
        print(f"Error opening file {metadata_path}: {e}")
        return

    print(f"Read metadata from {metadata_path}")

    # --- Determine the key for this database ---
    db_filename = os.path.basename(db_path)
    db_key = os.path.splitext(db_filename)[0]

    if 'databases' not in metadata:
        metadata['databases'] = {}

    # --- Add entry if it doesn't exist ---
    if db_key not in metadata['databases']:
        print(f"Database key '{db_key}' not found in metadata. Adding new entry.")
        metadata['databases'][db_key] = {
            'database_path': db_path,
            'tables': {}
        }
        metadata_changed = True
    else:
        metadata_changed = False
        # Update the path if it's different
        if metadata['databases'][db_key].get('database_path') != db_path:
            print(f"Updating database_path for '{db_key}' to '{db_path}'")
            metadata['databases'][db_key]['database_path'] = db_path
            metadata_changed = True

    db_entry = metadata['databases'][db_key]

    # --- Get the actual schema from the database --- 
    actual_schema = get_db_schema(db_path)

    if actual_schema is None:
        print("Failed to retrieve database schema. Aborting update.")
        return

    if not actual_schema:
        print(f"No tables found in database '{db_path}'. No updates made to metadata for '{db_key}'.")
        return

    # --- Synchronize schema with metadata --- 
    if 'tables' not in db_entry:
        db_entry['tables'] = {}
        metadata_changed = True # Added the tables key

    metadata_tables = db_entry['tables']

    # Iterate through tables found in the database
    for table_name, actual_columns in actual_schema.items():
        print(f"  Syncing table: '{table_name}'...")

        # Add table to metadata if it doesn't exist
        if table_name not in metadata_tables:
            print(f"    Table '{table_name}' not found in JSON, adding it.")
            metadata_tables[table_name] = {
                "description": f"Details for {table_name} (auto-added)",
                "columns": {}
            }
            metadata_changed = True

        # Ensure columns dict exists
        if 'columns' not in metadata_tables[table_name]:
             metadata_tables[table_name]['columns'] = {}
             metadata_changed = True # Added columns key

        metadata_columns = metadata_tables[table_name]['columns']

        # Iterate through columns found in the database table
        for col_name, actual_type in actual_columns.items():
            # Add column to metadata if it doesn't exist
            if col_name not in metadata_columns:
                print(f"      Column '{col_name}' not found in JSON for table '{table_name}', adding with type '{actual_type}'.")
                metadata_columns[col_name] = {
                    "type": actual_type,
                    "description": "(auto-added)"
                }
                metadata_changed = True
            else:
                # Update column type if DB type is different and not empty/defaulted
                current_type = metadata_columns[col_name].get('type', None)
                if current_type != actual_type:
                    print(f"      Updating type for column '{col_name}': '{current_type}' -> '{actual_type}'")
                    metadata_columns[col_name]['type'] = actual_type
                    metadata_changed = True

    # --- Save updated metadata if changes were made --- 
    if metadata_changed:
        print(f"Metadata for '{db_key}' was updated. Saving changes to {metadata_path}...")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Successfully updated metadata file: {metadata_path}")
        except IOError as e:
            print(f"Error writing updated metadata to {metadata_path}: {e}")
        except TypeError as e:
            print(f"Error serializing metadata to JSON: {e}")
    else:
        print(f"No changes needed for '{db_key}'. Metadata file remains unchanged.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update or add database schema metadata for a given SQLite DB.")
    parser.add_argument("db_path", help="Path to the SQLite database file.")
    parser.add_argument("--metadata", default=r"C:\Users\rrathore1\Projects\smartquery\assets\database_metadata.json", help="Path to the metadata JSON file (default: database_metadata.json)")
    args = parser.parse_args()

    print(f"Starting metadata update process for database: '{args.db_path}'...")
    update_metadata_for_db(args.metadata, args.db_path)
    print("Metadata update process finished.") 
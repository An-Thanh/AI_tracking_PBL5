import psycopg2
from ..config import Config

def setup_database(config: Config):
    """Set up the database schema from a SQL file."""
    try:
        conn = psycopg2.connect(**config.db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        with open('src/schema_db.sql', 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        cursor.execute(sql_script)
        config.logger.info("Database schema created successfully!")
    except Exception as e:
        config.logger.error(f"Failed to create schema: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
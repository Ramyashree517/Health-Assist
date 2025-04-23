import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Database connection
def get_db_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

def create_appointments_table():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) NOT NULL,
        phone VARCHAR(20) NOT NULL,
        date DATE NOT NULL,
        time TIME NOT NULL,
        comments TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    cur.close()
    conn.close()
    print("Appointments table created successfully.")

if __name__ == "__main__":
    create_appointments_table()
    
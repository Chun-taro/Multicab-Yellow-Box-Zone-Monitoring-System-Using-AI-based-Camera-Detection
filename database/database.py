import sqlite3
from config.config import config

# Determine DATABASE_PATH from config (supports dict or module), fallback to 'database.db'
if isinstance(config, dict):
    DATABASE_PATH = config.get('DATABASE_PATH', 'database.db')
else:
    DATABASE_PATH = getattr(config, 'DATABASE_PATH', 'database.db')


class Database:
    def __init__(self, db_path=None):
        # allow overriding path for tests or runtime
        path = db_path or DATABASE_PATH
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            timestamp TEXT,
            stop_time REAL,
            image TEXT
        )
        '''
        self.conn.execute(query)
        self.conn.commit()

    def insert_violation(self, vehicle_id, timestamp, stop_time, image):
        query = 'INSERT INTO violations (vehicle_id, timestamp, stop_time, image) VALUES (?, ?, ?, ?)'
        self.conn.execute(query, (vehicle_id, timestamp, stop_time, image))
        self.conn.commit()

    def get_all_violations(self):
        query = 'SELECT * FROM violations ORDER BY timestamp DESC'
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()

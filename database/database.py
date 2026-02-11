import sqlite3
from config.config import config
import json
from datetime import datetime

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
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.create_tables()
        self.initialize_defaults()

    def create_tables(self):
        """Create all necessary database tables."""
        
        # Vehicle types lookup table
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_name TEXT NOT NULL UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Monitoring zones
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_name TEXT NOT NULL,
            coordinates TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Main violations table
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            detection_id TEXT UNIQUE,
            vehicle_type_id INTEGER,
            zone_id INTEGER DEFAULT 1,
            stop_duration REAL,
            image_path TEXT,
            image_blob BLOB,
            confidence REAL DEFAULT 0.0,
            notes TEXT,
            reviewed BOOLEAN DEFAULT 0,
            status TEXT DEFAULT 'recorded',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vehicle_type_id) REFERENCES vehicle_types(id),
            FOREIGN KEY (zone_id) REFERENCES zones(id)
        )
        ''')
        
        # Detection history (optional, for detailed tracking)
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            frame_id TEXT,
            tracking_id INTEGER,
            vehicle_type_id INTEGER,
            zone_id INTEGER,
            centroid_x INTEGER,
            centroid_y INTEGER,
            confidence REAL,
            is_in_zone BOOLEAN,
            is_stopped BOOLEAN,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vehicle_type_id) REFERENCES vehicle_types(id),
            FOREIGN KEY (zone_id) REFERENCES zones(id)
        )
        ''')
        
        # Statistics table for reporting
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            total_violations INTEGER DEFAULT 0,
            violations_by_type TEXT,
            peak_hour INTEGER,
            total_vehicles_detected INTEGER DEFAULT 0,
            system_uptime_minutes INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for common queries
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(violation_timestamp)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_violations_vehicle_type ON violations(vehicle_type_id)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_detection_tracking_id ON detection_history(tracking_id)')
        
        self.conn.commit()

    def initialize_defaults(self):
        """Initialize default vehicle types and zone."""
        
        # Add default vehicle types if they don't exist
        vehicle_types = ['car', 'truck', 'bus', 'motorcycle']
        for vtype in vehicle_types:
            try:
                self.conn.execute(
                    'INSERT INTO vehicle_types (type_name) VALUES (?)',
                    (vtype,)
                )
            except sqlite3.IntegrityError:
                pass  # Already exists
        
        # Add default zone if it doesn't exist
        try:
            self.conn.execute('''
            INSERT INTO zones (zone_name, coordinates, is_active)
            VALUES (?, ?, ?)
            ''', ('Yellow Box Zone - Main', '[]', 1))
        except sqlite3.IntegrityError:
            pass  # Already exists
        
        self.conn.commit()

    def get_vehicle_type_id(self, vehicle_type_name):
        """Get vehicle type ID from name."""
        cursor = self.conn.execute(
            'SELECT id FROM vehicle_types WHERE type_name = ?',
            (vehicle_type_name.lower(),)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def insert_violation(self, vehicle_type, timestamp, image_path, image_blob=None, 
                        detection_id=None, stop_duration=None, confidence=0.0, notes=None, zone_id=1):
        """
        Insert a new violation record.
        
        Args:
            vehicle_type (str): Type of vehicle (car, truck, bus, motorcycle)
            timestamp (str): Violation timestamp (YYYY-MM-DD HH:MM:SS)
            image_path (str): Path to violation image
            image_blob (bytes): Optional binary image data
            detection_id (str): Optional detection identifier
            stop_duration (float): Optional duration vehicle was stopped
            confidence (float): YOLOv8 detection confidence (0.0-1.0)
            notes (str): Optional additional notes
            zone_id (int): Zone ID (default: 1)
        """
        vehicle_type_id = self.get_vehicle_type_id(vehicle_type)
        
        query = '''
        INSERT INTO violations 
        (vehicle_type_id, violation_timestamp, image_path, image_blob, 
         detection_id, stop_duration, confidence, notes, zone_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.conn.execute(query, (
            vehicle_type_id, timestamp, image_path, image_blob,
            detection_id, stop_duration, confidence, notes, zone_id
        ))
        self.conn.commit()

    def get_all_violations(self):
        """Get all violations ordered by timestamp (newest first)."""
        query = '''
        SELECT v.id, v.violation_timestamp as timestamp, vt.type_name as label, 
               v.image_path, v.stop_duration, v.confidence, v.status
        FROM violations v
        LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
        ORDER BY v.violation_timestamp DESC
        '''
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def get_violations_by_date(self, date):
        """Get violations for a specific date (YYYY-MM-DD)."""
        query = '''
        SELECT v.id, v.violation_timestamp as timestamp, vt.type_name as label,
               v.image_path, v.stop_duration, v.confidence, v.status
        FROM violations v
        LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
        WHERE DATE(v.violation_timestamp) = ?
        ORDER BY v.violation_timestamp DESC
        '''
        cursor = self.conn.execute(query, (date,))
        return cursor.fetchall()

    def get_violations_by_vehicle_type(self, vehicle_type, limit=None):
        """Get all violations of a specific vehicle type."""
        vehicle_type_id = self.get_vehicle_type_id(vehicle_type)
        if not vehicle_type_id:
            return []
        
        query = '''
        SELECT v.id, v.violation_timestamp as timestamp, vt.type_name as label,
               v.image_path, v.stop_duration, v.confidence, v.status
        FROM violations v
        LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
        WHERE v.vehicle_type_id = ?
        ORDER BY v.violation_timestamp DESC
        '''
        if limit:
            query += f' LIMIT {limit}'
        
        cursor = self.conn.execute(query, (vehicle_type_id,))
        return cursor.fetchall()

    def get_violation_by_id(self, violation_id):
        """Get a specific violation by ID."""
        query = '''
        SELECT v.id, v.violation_timestamp as timestamp, vt.type_name as label,
               v.image_path, v.image_blob, v.stop_duration, v.confidence, 
               v.status, v.notes
        FROM violations v
        LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
        WHERE v.id = ?
        '''
        cursor = self.conn.execute(query, (violation_id,))
        return cursor.fetchone()

    def update_violation_status(self, violation_id, status, notes=None):
        """Update violation status (recorded, reviewed, processed, dismissed)."""
        query = '''
        UPDATE violations
        SET status = ?, notes = ?, reviewed = 1
        WHERE id = ?
        '''
        self.conn.execute(query, (status, notes, violation_id))
        self.conn.commit()

    def get_statistics(self, date):
        """Get or create statistics for a date."""
        query = 'SELECT * FROM statistics WHERE date = ?'
        cursor = self.conn.execute(query, (date,))
        return cursor.fetchone()

    def count_violations_by_type(self, date=None):
        """Count violations grouped by vehicle type for a date or all time."""
        if date:
            query = '''
            SELECT vt.type_name, COUNT(*) as count
            FROM violations v
            LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
            WHERE DATE(v.violation_timestamp) = ?
            GROUP BY v.vehicle_type_id
            '''
            cursor = self.conn.execute(query, (date,))
        else:
            query = '''
            SELECT vt.type_name, COUNT(*) as count
            FROM violations v
            LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
            GROUP BY v.vehicle_type_id
            '''
            cursor = self.conn.execute(query)
        
        return cursor.fetchall()

    def record_detection(self, tracking_id, vehicle_type, centroid_x, centroid_y,
                        confidence, is_in_zone, is_stopped, frame_id=None, zone_id=1):
        """
        Record a vehicle detection for analytics (optional).
        
        Args:
            tracking_id (int): YOLOv8 object tracking ID
            vehicle_type (str): Type of vehicle
            centroid_x (int): X coordinate of vehicle center
            centroid_y (int): Y coordinate of vehicle center
            confidence (float): Detection confidence
            is_in_zone (bool): Whether vehicle is in monitored zone
            is_stopped (bool): Whether vehicle appears stopped
            frame_id (str): Optional frame identifier
            zone_id (int): Zone ID (default: 1)
        """
        vehicle_type_id = self.get_vehicle_type_id(vehicle_type)
        
        query = '''
        INSERT INTO detection_history
        (tracking_id, vehicle_type_id, centroid_x, centroid_y, confidence,
         is_in_zone, is_stopped, frame_id, zone_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.conn.execute(query, (
            tracking_id, vehicle_type_id, centroid_x, centroid_y, confidence,
            is_in_zone, is_stopped, frame_id, zone_id
        ))
        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


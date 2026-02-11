# Database Usage Guide

This guide explains how to use the updated database module for the Multicab Yellow Box Zone Monitoring System.

---

## Overview

The database module has been redesigned with a comprehensive schema supporting:
- **Violation tracking** with vehicle type classification
- **Detection history** for analytics
- **Zone management** for multiple monitoring areas
- **Violation review workflow** with status tracking
- **Statistics aggregation** for reporting

---

## Basic Usage

### Initializing the Database

```python
from database.database import Database

# Create database instance (auto-creates all tables)
db = Database()

# Or specify custom path
db = Database(db_path='/custom/path/violations.db')
```

### Recording Violations

When a vehicle violates the yellow box rule:

```python
# Simple violation recording
db.insert_violation(
    vehicle_type='car',
    timestamp='2026-02-11 14:30:15',
    image_path='violations/violation_20260211_143015_car_22.jpg'
)

# Full violation with all details
db.insert_violation(
    vehicle_type='truck',
    timestamp='2026-02-11 14:35:22',
    image_path='violations/violation_20260211_143522_truck_45.jpg',
    image_blob=image_bytes,
    detection_id='20260211_143522_45',
    stop_duration=18.5,
    confidence=0.94,
    notes='Multiple violations on same vehicle',
    zone_id=1
)
```

### Querying Violations

```python
# Get all violations (newest first)
all_violations = db.get_all_violations()
for v in all_violations:
    print(f"{v['timestamp']}: {v['label']} - {v['image_path']}")

# Get violations for a specific date
today_violations = db.get_violations_by_date('2026-02-11')

# Get all cars that violated
cars = db.get_violations_by_vehicle_type('car')

# Get specific violation
violation = db.get_violation_by_id(1)
print(f"Confidence: {violation['confidence']}")
print(f"Status: {violation['status']}")
```

### Violation Status Management

```python
# Review a violation
db.update_violation_status(
    violation_id=1,
    status='reviewed',
    notes='Validated by operator'
)

# Mark as processed
db.update_violation_status(
    violation_id=1,
    status='processed',
    notes='Assigned to enforcement'
)

# Dismiss a violation
db.update_violation_status(
    violation_id=1,
    status='dismissed',
    notes='False positive - not a violation'
)
```

### Recording Detection History (Analytics)

For detailed tracking of all vehicle detections (useful for analytics):

```python
db.record_detection(
    tracking_id=22,
    vehicle_type='car',
    centroid_x=512,
    centroid_y=384,
    confidence=0.92,
    is_in_zone=True,
    is_stopped=False,
    frame_id='frame_20260211_143015',
    zone_id=1
)
```

### Statistics

```python
# Get statistics for a date
stats = db.get_statistics('2026-02-11')
if stats:
    print(f"Total violations: {stats['total_violations']}")
    print(f"Peak hour: {stats['peak_hour']}:00")

# Count violations by vehicle type
counts = db.count_violations_by_type('2026-02-11')
for row in counts:
    print(f"{row['type_name']}: {row['count']}")

# Count for all time
all_counts = db.count_violations_by_type()
```

---

## Flask Integration Example

### Dashboard Routes

```python
from flask import Blueprint, render_template
from database.database import Database

dashboard_bp = Blueprint('dashboard', __name__)
db = Database()

@dashboard_bp.route('/dashboard')
def dashboard():
    violations = db.get_all_violations()
    return render_template('dashboard.html', violations=violations)

@dashboard_bp.route('/violations/<int:violation_id>')
def view_violation(violation_id):
    violation = db.get_violation_by_id(violation_id)
    return render_template('violation_detail.html', violation=violation)

@dashboard_bp.route('/api/statistics/<date>')
def get_stats(date):
    stats = db.get_statistics(date)
    counts = db.count_violations_by_vehicle_type(date)
    return jsonify({
        'stats': dict(stats) if stats else None,
        'vehicle_counts': [dict(row) for row in counts]
    })
```

### API Routes

```python
from flask import Blueprint, request, jsonify
from database.database import Database

api_bp = Blueprint('api', __name__)
db = Database()

@api_bp.route('/violations', methods=['GET'])
def list_violations():
    date = request.args.get('date')
    vehicle_type = request.args.get('vehicle_type')
    limit = request.args.get('limit', 10, type=int)
    
    if date and vehicle_type:
        violations = db.get_violations_by_vehicle_type(vehicle_type, limit=limit)
    elif date:
        violations = db.get_violations_by_date(date)
    else:
        violations = db.get_all_violations()[:limit]
    
    return jsonify([dict(v) for v in violations])

@api_bp.route('/violations/<int:violation_id>/status', methods=['PUT'])
def update_status(violation_id):
    data = request.json
    db.update_violation_status(
        violation_id=violation_id,
        status=data.get('status'),
        notes=data.get('notes')
    )
    return jsonify({'success': True})
```

---

## Data Model Reference

### Violation Object (dict-like)
```python
{
    'id': 1,
    'timestamp': '2026-02-11 14:30:15',
    'label': 'car',                    # vehicle_type
    'image_path': 'violations/...',
    'image_blob': b'...',              # Optional binary data
    'stop_duration': 18.5,             # Seconds stopped
    'confidence': 0.94,                # Detection confidence (0.0-1.0)
    'status': 'recorded',              # recorded|reviewed|processed|dismissed
    'notes': 'Object ID: 22',
    'reviewed': 0,                     # Boolean
    'created_at': '2026-02-11 14:30:15'
}
```

### Statistics Object
```python
{
    'id': 1,
    'date': '2026-02-11',
    'total_violations': 15,
    'violations_by_type': '{"car": 10, "truck": 3, "bus": 2}',
    'peak_hour': 14,                   # Hour 14 (2 PM) had most violations
    'total_vehicles_detected': 127,
    'system_uptime_minutes': 480,
    'created_at': '2026-02-11 23:59:59'
}
```

---

## Performance Considerations

### Indexes
The following indexes are automatically created for performance:
- `violation_timestamp` - Fast date range queries
- `vehicle_type_id` - Fast filtering by vehicle type
- `status` - Fast review workflow queries
- `tracking_id` - Fast detection history lookups

### Query Examples

**Get violations from last 7 days:**
```python
from datetime import datetime, timedelta
start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
# Query using database.conn directly:
query = f"SELECT * FROM violations WHERE date(violation_timestamp) >= '{start_date}'"
cursor = db.conn.execute(query)
results = cursor.fetchall()
```

**Get violations with high confidence:**
```python
query = "SELECT * FROM violations WHERE confidence > 0.90 ORDER BY violation_timestamp DESC"
cursor = db.conn.execute(query)
results = cursor.fetchall()
```

---

## Maintenance

### Backup Database

```python
import shutil
from datetime import datetime

backup_name = f"violations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
shutil.copy('database.db', f'backups/{backup_name}')
```

### Archive Old Violations

```python
from datetime import datetime, timedelta

# Archive violations older than 30 days
archive_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
query = f"""
INSERT INTO violations_archive 
SELECT * FROM violations 
WHERE date(violation_timestamp) < '{archive_date}'
"""
db.conn.execute(query)
db.conn.commit()

# Delete archived violations
query = f"DELETE FROM violations WHERE date(violation_timestamp) < '{archive_date}'"
db.conn.execute(query)
db.conn.commit()
```

---

## Troubleshooting

### Issue: "database is locked"
**Cause:** Multiple processes accessing the database simultaneously

**Solution:** Either use a more robust database (PostgreSQL) or ensure single-threaded access:
```python
# In database.py initialization
self.conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
```

### Issue: Missing vehicle type conversions
**Cause:** Vehicle type name doesn't match exactly

**Solution:** Check vehicle type names are lowercase:
```python
vehicle_type = 'Car'  # Wrong
vehicle_type = 'car'  # Correct
```

---

## Future Improvements

1. **Confidence Tracking**: Currently confidence is stored but not extracted per-violation. Enhance tracker to maintain confidence mapping.

2. **PostgreSQL Support**: For production deployments, migrate from SQLite to PostgreSQL for better concurrency.

3. **Image Storage**: Consider moving image blobs to object storage (S3, MinIO) instead of database for better performance.

4. **Real-time Sync**: Implement WebSocket support for real-time violation notifications.

5. **Analytics Dashboard**: Add pre-computed statistics for faster reporting.


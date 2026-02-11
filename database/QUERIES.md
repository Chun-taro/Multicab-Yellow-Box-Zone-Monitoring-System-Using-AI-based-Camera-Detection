# Database Query Examples

This file contains commonly used SQL queries for the violation system. These can be used directly with `db.conn.execute()` or as reference for API endpoints.

---

## Dashboard Queries

### Get Recent Violations (Last 24 Hours)
```sql
SELECT 
    v.id,
    v.violation_timestamp,
    vt.type_name as vehicle_type,
    v.image_path,
    v.stop_duration,
    v.confidence,
    v.status
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.violation_timestamp > datetime('now', '-1 day')
ORDER BY v.violation_timestamp DESC
LIMIT 50;
```

### Get Violations Requiring Review
```sql
SELECT 
    v.id,
    v.violation_timestamp,
    vt.type_name as vehicle_type,
    v.image_path,
    v.confidence,
    CAST((julianday('now') - julianday(v.violation_timestamp)) * 24 AS INTEGER) as hours_ago
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.reviewed = 0
ORDER BY v.violation_timestamp ASC
LIMIT 20;
```

### Get Today's Statistics
```sql
SELECT 
    COUNT(*) as total_violations,
    COUNT(DISTINCT CASE WHEN confidence > 0.9 THEN id END) as high_confidence,
    COUNT(DISTINCT CASE WHEN confidence > 0.7 AND confidence <= 0.9 THEN id END) as medium_confidence,
    AVG(stop_duration) as avg_stop_time,
    MAX(stop_duration) as max_stop_time,
    ROUND(AVG(confidence), 3) as avg_confidence
FROM violations
WHERE DATE(violation_timestamp) = DATE('now');
```

---

## Vehicle Type Analysis

### Violations by Vehicle Type (Today)
```sql
SELECT 
    vt.type_name,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage,
    ROUND(AVG(v.stop_duration), 2) as avg_duration,
    MAX(v.stop_duration) as max_duration
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE DATE(v.violation_timestamp) = DATE('now')
GROUP BY vt.type_name
ORDER BY count DESC;
```

### Violations by Vehicle Type (Weekly)
```sql
SELECT 
    DATE(v.violation_timestamp) as date,
    vt.type_name as vehicle_type,
    COUNT(*) as count
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.violation_timestamp > datetime('now', '-7 days')
GROUP BY DATE(v.violation_timestamp), vt.type_name
ORDER BY date DESC, count DESC;
```

### Vehicle Type Statistics (All Time)
```sql
SELECT 
    vt.type_name,
    COUNT(*) as total_violations,
    COUNT(DISTINCT DATE(v.violation_timestamp)) as days_with_violations,
    ROUND(AVG(v.stop_duration), 2) as avg_duration,
    ROUND(AVG(v.confidence), 3) as avg_confidence,
    MAX(v.violation_timestamp) as most_recent
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
GROUP BY vt.type_name
ORDER BY total_violations DESC;
```

---

## Time-Based Analysis

### Peak Violation Hours (Today)
```sql
SELECT 
    CAST(STRFTIME('%H', v.violation_timestamp) AS INTEGER) as hour,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / 
        (SELECT COUNT(*) FROM violations 
         WHERE DATE(violation_timestamp) = DATE('now')), 1) as percentage
FROM violations v
WHERE DATE(v.violation_timestamp) = DATE('now')
GROUP BY CAST(STRFTIME('%H', v.violation_timestamp) AS INTEGER)
ORDER BY hour;
```

### Violations by Day of Week (Last 30 Days)
```sql
SELECT 
    CASE CAST(STRFTIME('%w', v.violation_timestamp) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day_of_week,
    COUNT(*) as count,
    ROUND(AVG(v.stop_duration), 2) as avg_duration
FROM violations v
WHERE v.violation_timestamp > datetime('now', '-30 days')
GROUP BY STRFTIME('%w', v.violation_timestamp)
ORDER BY CAST(STRFTIME('%w', v.violation_timestamp) AS INTEGER);
```

### Hourly Trend (Last 7 Days)
```sql
SELECT 
    DATE(v.violation_timestamp) as date,
    CAST(STRFTIME('%H', v.violation_timestamp) AS INTEGER) as hour,
    COUNT(*) as count
FROM violations v
WHERE v.violation_timestamp > datetime('now', '-7 days')
GROUP BY DATE(v.violation_timestamp), 
         CAST(STRFTIME('%H', v.violation_timestamp) AS INTEGER)
ORDER BY date DESC, hour DESC;
```

---

## Quality & Confidence Analysis

### High Confidence Violations (>90%)
```sql
SELECT 
    v.id,
    v.violation_timestamp,
    vt.type_name,
    v.confidence,
    v.stop_duration,
    v.image_path
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.confidence > 0.9
ORDER BY v.confidence DESC, v.violation_timestamp DESC;
```

### Low Confidence Violations (<70%) - Need Review
```sql
SELECT 
    v.id,
    v.violation_timestamp,
    vt.type_name,
    v.confidence,
    v.stop_duration,
    v.reviewed,
    v.image_path
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.confidence < 0.7 AND v.confidence IS NOT NULL
ORDER BY v.confidence ASC, v.violation_timestamp DESC;
```

### Confidence Distribution
```sql
SELECT 
    CASE 
        WHEN v.confidence >= 0.9 THEN '90-100%'
        WHEN v.confidence >= 0.8 THEN '80-89%'
        WHEN v.confidence >= 0.7 THEN '70-79%'
        WHEN v.confidence >= 0.6 THEN '60-69%'
        ELSE '<60%'
    END as confidence_range,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / 
        (SELECT COUNT(*) FROM violations 
         WHERE confidence IS NOT NULL), 1) as percentage
FROM violations v
WHERE v.confidence IS NOT NULL
GROUP BY confidence_range
ORDER BY 
    CASE 
        WHEN v.confidence >= 0.9 THEN 1
        WHEN v.confidence >= 0.8 THEN 2
        WHEN v.confidence >= 0.7 THEN 3
        WHEN v.confidence >= 0.6 THEN 4
        ELSE 5
    END;
```

---

## Review Status Tracking

### Violations by Status
```sql
SELECT 
    v.status,
    COUNT(*) as count,
    COUNT(DISTINCT DATE(v.violation_timestamp)) as days_involved,
    MIN(v.violation_timestamp) as oldest,
    MAX(v.violation_timestamp) as newest
FROM violations v
GROUP BY v.status
ORDER BY count DESC;
```

### Review Backlog (Oldest Unreviewed First)
```sql
SELECT 
    v.id,
    v.violation_timestamp,
    vt.type_name,
    v.confidence,
    CAST((julianday('now') - julianday(v.violation_timestamp)) * 24 AS INTEGER) as hours_waiting,
    v.image_path
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.reviewed = 0
ORDER BY v.violation_timestamp ASC
LIMIT 30;
```

### Reviews Completed (Last 24 Hours)
```sql
SELECT 
    COUNT(*) as reviews_completed,
    COUNT(DISTINCT DATE(v.violation_timestamp)) as violations_from_days,
    ROUND(AVG(CASE WHEN v.status = 'dismissed' THEN 1 ELSE 0 END) * 100, 1) as dismiss_rate
FROM violations v
WHERE v.reviewed = 1 
AND v.created_at > datetime('now', '-1 day');
```

---

## Detection History Analytics

### Vehicle Detections in Zone (Last Hour)
```sql
SELECT 
    dh.tracking_id,
    vt.type_name,
    COUNT(*) as detections,
    MAX(dh.timestamp) as last_seen,
    ROUND(AVG(dh.confidence), 3) as avg_confidence,
    MAX(dh.centroid_x) as last_x,
    MAX(dh.centroid_y) as last_y
FROM detection_history dh
LEFT JOIN vehicle_types vt ON dh.vehicle_type_id = vt.id
WHERE dh.timestamp > datetime('now', '-1 hour')
AND dh.is_in_zone = 1
GROUP BY dh.tracking_id
ORDER BY last_seen DESC;
```

### Stopped Vehicles (Currently)
```sql
SELECT 
    dh.tracking_id,
    vt.type_name,
    dh.timestamp,
    dh.centroid_x,
    dh.centroid_y,
    dh.confidence,
    CAST((julianday('now') - julianday(dh.timestamp)) * 3600 AS INTEGER) as seconds_ago
FROM detection_history dh
LEFT JOIN vehicle_types vt ON dh.vehicle_type_id = vt.id
WHERE dh.is_stopped = 1
AND dh.is_in_zone = 1
AND dh.timestamp > datetime('now', '-10 minutes')
ORDER BY dh.timestamp DESC;
```

---

## Data Export Queries

### Export Violations as CSV Format
```sql
SELECT 
    v.id,
    v.violation_timestamp,
    vt.type_name as vehicle_type,
    v.stop_duration,
    v.confidence,
    v.status,
    v.image_path
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE DATE(v.violation_timestamp) = ?
ORDER BY v.violation_timestamp DESC;
```

### Export Daily Summary Report
```sql
SELECT 
    DATE(v.violation_timestamp) as report_date,
    'Daily Summary' as report_type,
    COUNT(*) as total_violations,
    COUNT(DISTINCT vt.type_name) as vehicle_types_involved,
    ROUND(AVG(v.stop_duration), 2) as avg_stop_duration,
    ROUND(AVG(v.confidence), 3) as avg_confidence,
    COUNT(DISTINCT CASE WHEN v.reviewed = 1 THEN v.id END) as reviewed_count,
    COUNT(DISTINCT CASE WHEN v.status = 'processed' THEN v.id END) as processed_count
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.violation_timestamp > datetime('now', '-30 days')
GROUP BY DATE(v.violation_timestamp)
ORDER BY report_date DESC;
```

---

## Maintenance Queries

### Count Violations by Month
```sql
SELECT 
    STRFTIME('%Y-%m', v.violation_timestamp) as month,
    COUNT(*) as count,
    ROUND(SUM(LENGTH(v.image_blob)) / (1024.0 * 1024.0), 2) as storage_mb
FROM violations v
WHERE v.image_blob IS NOT NULL
GROUP BY STRFTIME('%Y-%m', v.violation_timestamp)
ORDER BY month DESC;
```

### Identify and Delete Duplicates (Same Detection ID)
```sql
-- Find duplicates
SELECT 
    v.detection_id,
    COUNT(*) as count,
    GROUP_CONCAT(v.id) as ids
FROM violations v
WHERE v.detection_id IS NOT NULL
GROUP BY v.detection_id
HAVING COUNT(*) > 1;

-- Delete duplicates (keep only first)
DELETE FROM violations
WHERE id NOT IN (
    SELECT MIN(id)
    FROM violations
    WHERE detection_id IS NOT NULL
    GROUP BY detection_id
);
```

### Archive Old Violations (Older than 90 Days)
```sql
-- Create archive table first (if not exists)
CREATE TABLE IF NOT EXISTS violations_archive AS 
SELECT * FROM violations WHERE 1=0;

-- Move old violations
INSERT INTO violations_archive
SELECT * FROM violations
WHERE violation_timestamp < datetime('now', '-90 days');

DELETE FROM violations
WHERE violation_timestamp < datetime('now', '-90 days');
```

### Database Statistics
```sql
SELECT 
    'violations' as table_name,
    COUNT(*) as row_count,
    ROUND(SUM(LENGTH(json_array(
        id, violation_timestamp, image_path, notes
    ))) / (1024.0 * 1024.0), 2) as estimated_size_mb
FROM violations
UNION ALL
SELECT 
    'detection_history',
    COUNT(*),
    ROUND(SUM(LENGTH(json_array(
        id, timestamp, tracking_id
    ))) / (1024.0 * 1024.0), 2)
FROM detection_history
UNION ALL
SELECT 
    'statistics',
    COUNT(*),
    ROUND(SUM(LENGTH(json_array(
        date, total_violations
    ))) / (1024.0 * 1024.0), 2)
FROM statistics;
```

---

## Python Example: Running These Queries

```python
from database.database import Database

db = Database()

# Run custom query
query = """
SELECT vt.type_name, COUNT(*) as count
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE DATE(v.violation_timestamp) = DATE('now')
GROUP BY vt.type_name
ORDER BY count DESC
"""

cursor = db.conn.execute(query)
results = cursor.fetchall()

for row in results:
    print(f"{row['type_name']}: {row['count']} violations")

db.close()
```


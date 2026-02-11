# Database Schema Documentation

## Overview
This document describes the SQLite database schema for the Multicab Yellow Box Zone Monitoring System.

---

## Tables

### 1. `vehicle_types`
Lookup table for supported vehicle classifications from YOLOv8 detection.

| Column | Type | Constraints | Description |
|--------|------|-----------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier |
| type_name | TEXT | NOT NULL, UNIQUE | Vehicle type (car, truck, bus, motorcycle) |
| created_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Creation timestamp |

**Sample Data:**
```
id | type_name | created_at
1  | car | 2026-02-11 12:00:00
2  | truck | 2026-02-11 12:00:00
3  | bus | 2026-02-11 12:00:00
4  | motorcycle | 2026-02-11 12:00:00
```

---

### 2. `zones`
Monitoring zones (yellow box areas) configuration.

| Column | Type | Constraints | Description |
|--------|------|-----------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier |
| zone_name | TEXT | NOT NULL | Name of the zone (e.g., "Yellow Box - Main Intersection") |
| coordinates | TEXT | NOT NULL | JSON string of zone polygon coordinates |
| is_active | BOOLEAN | DEFAULT 1 | Whether zone monitoring is enabled |
| created_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Creation timestamp |
| updated_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Last update timestamp |

**Sample Data:**
```
id | zone_name | coordinates | is_active | created_at
1  | Yellow Box Zone | [[x1,y1], [x2,y2], ...] | 1 | 2026-02-11 12:00:00
```

---

### 3. `violations`
Main violation records when vehicles stop in yellow box zone beyond time limit.

| Column | Type | Constraints | Description |
|--------|------|-----------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier |
| violation_timestamp | TEXT | NOT NULL, DEFAULT CURRENT_TIMESTAMP | When violation occurred |
| detection_id | TEXT | UNIQUE | Camera frame identification (timestamp_objid) |
| vehicle_type_id | INTEGER | FOREIGN KEY | Reference to vehicle_types table |
| zone_id | INTEGER | FOREIGN KEY | Reference to zones table (default: 1) |
| stop_duration | REAL | - | How long (seconds) vehicle remained stopped |
| image_path | TEXT | - | Relative path to violation image (static/violations/...) |
| image_blob | BLOB | - | Binary image data |
| confidence | REAL | DEFAULT 0.0 | YOLOv8 detection confidence (0.0-1.0) |
| notes | TEXT | - | Additional information about violation |
| reviewed | BOOLEAN | DEFAULT 0 | Whether violation has been reviewed |
| status | TEXT | DEFAULT 'recorded' | Status (recorded, reviewed, processed, dismissed) |
| created_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Record creation timestamp |

**Indexes:**
- `violation_timestamp` (for quick time-range queries)
- `vehicle_type_id` (for filtering by vehicle type)
- `status` (for filtering by review status)

**Sample Data:**
```
id | violation_timestamp | detection_id | vehicle_type_id | zone_id | stop_duration | image_path | confidence | status
1  | 2026-02-11 14:30:15 | 20260211_143015_22 | 1 | 1 | 18.5 | violations/violation_20260211_143015_car_22.jpg | 0.92 | recorded
```

---

### 4. `detection_history`
Detailed tracking of all vehicle detections (optional, for analytics).

| Column | Type | Constraints | Description |
|--------|------|-----------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier |
| timestamp | TEXT | NOT NULL, DEFAULT CURRENT_TIMESTAMP | Detection timestamp |
| frame_id | TEXT | - | Frame identifier from camera feed |
| tracking_id | INTEGER | - | YOLOv5/8 object tracking ID |
| vehicle_type_id | INTEGER | FOREIGN KEY | Reference to vehicle_types |
| zone_id | INTEGER | FOREIGN KEY | Reference to zones |
| centroid_x | INTEGER | - | Vehicle center X coordinate |
| centroid_y | INTEGER | - | Vehicle center Y coordinate |
| confidence | REAL | - | Detection confidence score |
| is_in_zone | BOOLEAN | - | Whether detected inside monitored zone |
| is_stopped | BOOLEAN | - | Whether vehicle appeared stopped |
| created_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Record creation timestamp |

---

### 5. `statistics`
Daily/weekly aggregated violation statistics for reporting.

| Column | Type | Constraints | Description |
|--------|------|-----------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier |
| date | TEXT | NOT NULL, UNIQUE | Date (YYYY-MM-DD) |
| total_violations | INTEGER | DEFAULT 0 | Total violations for the day |
| violations_by_type | TEXT | - | JSON: {"car": 5, "truck": 2, ...} |
| peak_hour | INTEGER | - | Hour with most violations (0-23) |
| total_vehicles_detected | INTEGER | DEFAULT 0 | Total vehicles seen |
| system_uptime_minutes | INTEGER | DEFAULT 0 | Minutes system was running |
| created_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Record creation timestamp |
| updated_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Last update timestamp |

---

## Relationships

```
vehicle_types (1) ──── (*) violations
vehicle_types (1) ──── (*) detection_history

zones (1) ──── (*) violations
zones (1) ──── (*) detection_history

violations ──── detection_history (many detections lead to one violation)
```

---

## SQL Queries (Common Use Cases)

### Get recent violations (last 10)
```sql
SELECT v.id, v.violation_timestamp, vt.type_name, v.image_path, v.stop_duration
FROM violations v
JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
ORDER BY v.violation_timestamp DESC
LIMIT 10;
```

### Violations by vehicle type (today)
```sql
SELECT vt.type_name, COUNT(*) as count
FROM violations v
JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE DATE(v.violation_timestamp) = DATE('now')
GROUP BY vt.type_name;
```

### High-confidence violations (>90%)
```sql
SELECT v.id, v.violation_timestamp, vt.type_name, v.confidence
FROM violations v
JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.confidence > 0.9
ORDER BY v.violation_timestamp DESC;
```

### Statistics for a specific date
```sql
SELECT * FROM statistics
WHERE date = '2026-02-11';
```

---

## Notes

- **Timestamps** use ISO 8601 format (YYYY-MM-DD HH:MM:SS) for consistent database operations
- **Image paths** are relative to the Flask app root (static/violations/) for portability
- **Image blob** is optional to reduce database size; image_path is the primary reference
- **Detection IDs** combine timestamp and tracking object ID for uniqueness
- **Status field** allows for violation review workflows
- Consider adding backups for the database and archiving old violations periodically


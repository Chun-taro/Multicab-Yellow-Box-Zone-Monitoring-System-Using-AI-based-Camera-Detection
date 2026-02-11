# Database Schema - Entity Relationship Diagram

## ER Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         VIOLATIONS SYSTEM SCHEMA                         │
└──────────────────────────────────────────────────────────────────────────┘

                              vehicle_types
                         ┌──────────────────────┐
                         │ id (PK)              │
                         │ type_name (UNIQUE)   │
                         │ created_at           │
                         └──────────────────────┘
                                    △
                                    │ (FK)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        │                           │                           │
    violations              detection_history            statistics
 ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────┐
 │ id (PK)          │   │ id (PK)              │   │ id (PK)      │
 │ violation_ts     │   │ timestamp            │   │ date (UQ)    │
 │ detection_id(UQ) │   │ frame_id             │   │ total_viols  │
 │ vehicle_type_id *├──→│ tracking_id          │   │ by_type (J)  │
 │ zone_id         *├──→│ vehicle_type_id     *├──→│ peak_hour    │
 │ stop_duration    │   │ zone_id             *├──→│ vehicles_det │
 │ image_path       │   │ centroid_x           │   │ uptime_min   │
 │ image_blob       │   │ centroid_y           │   │ created_at   │
 │ confidence       │   │ confidence           │   │ updated_at   │
 │ notes            │   │ is_in_zone           │   └──────────────┘
 │ reviewed         │   │ is_stopped           │
 │ status           │   │ created_at           │
 │ created_at       │   └──────────────────────┘
 └──────────────────┘            △
        △                        │ (FK)
        │ (FK)                   │
        └────────────────────────┘
                 │
                 │ (FK)
                 │
            ┌─────────┐
            │ zones   │
            │─────────│
            │ id (PK) │
            │ name    │
            │ coords  │
            │ active  │
            │ created │
            │ updated │
            └─────────┘

```

---

## Table Relationships

### Primary Keys (PK)
- `vehicle_types.id`
- `zones.id`
- `violations.id`
- `detection_history.id`
- `statistics.id`

### Unique Constraints (UQ)
- `vehicle_types.type_name`
- `violations.detection_id`
- `statistics.date`

### Foreign Keys (FK)
- `violations.vehicle_type_id` → `vehicle_types.id`
- `violations.zone_id` → `zones.id`
- `detection_history.vehicle_type_id` → `vehicle_types.id`
- `detection_history.zone_id` → `zones.id`

### Indexes
```
violations:
  - idx_violations_timestamp (on violation_timestamp)
  - idx_violations_vehicle_type (on vehicle_type_id)
  - idx_violations_status (on status)

detection_history:
  - idx_detection_tracking_id (on tracking_id)
```

---

## Data Flow

```
Camera Feed
    │
    ├──→ YOLOv8 Detection
    │        │
    │        ├──→ Vehicle Type Classification
    │        └──→ Confidence Score
    │
    ├──→ CentroidTracker
    │        │
    │        └──→ Tracking ID (persists across frames)
    │
    ├───────┬─────────────────────────────────────┐
    │       │                                     │
    │   (Optional)                            (Mandatory)
    │   Record Detection                      Check Violation
    │   to detection_history                  Criteria
    │       │                                     │
    │       │                                     ├──→ In Zone?
    │       │                                     ├──→ Stopped?
    │       │                                     └──→ Time > Limit?
    │       │                                         │
    │       │                                         ├──→ YES: VIOLATION
    │       │                                         └──→ NO: Continue tracking
    │       │
    └──────→ violations table
                    │
                    ├──→ Crop & Save Image
                    ├──→ Create Detection ID
                    └──→ Record with Type & Timestamp

violations table
    │
    ├──→ Dashboard Display
    ├──→ Status Review Workflow
    ├──→ Export to CSV/Backend
    └──→ Analytics Report
```

---

## Record Lifecycle

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Violation    │──(review)→│  Reviewed    │──(process)→│ Processed  │
│ Recorded     │           │              │           │            │
└──────────────┘           └──────────────┘           └──────────────┘
       │
       │ (false positive)
       ▼
┌──────────────┐
│  Dismissed   │
└──────────────┘

Status field tracks current state:
- 'recorded': Initial violation detection
- 'reviewed': Operator has verified the violation
- 'processed': Violation assigned to enforcement/action
- 'dismissed': False positive or not a violation
```

---

## Statistics Aggregation

```
┌─────────────────────────────────────────┐
│  Daily Statistics Computation            │
├─────────────────────────────────────────┤
│ Date: 2026-02-11                        │
│                                         │
│ Total Violations: 15                    │
│   - Car: 10                             │
│   - Truck: 3                            │
│   - Bus: 2                              │
│   - Motorcycle: 0                       │
│                                         │
│ Peak Hour: 14 (2:00 PM - 3:00 PM)       │
│   Hour 14: 5 violations                 │
│   Hour 13: 3 violations                 │
│   Hour 15: 2 violations                 │
│                                         │
│ Total Vehicles Detected: 127            │
│ System Uptime: 480 minutes (8 hours)    │
└─────────────────────────────────────────┘
```

---

## Queries by Use Case

### Operational Dashboards
```sql
-- Today's violations
SELECT v.id, v.violation_timestamp, vt.type_name, v.confidence
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE DATE(v.violation_timestamp) = DATE('now')
ORDER BY v.violation_timestamp DESC;

-- Violations needing review
SELECT v.id, v.violation_timestamp, vt.type_name, v.image_path
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.reviewed = 0
ORDER BY v.violation_timestamp ASC;
```

### Analytics & Reporting
```sql
-- Violations by vehicle type (weekly)
SELECT vt.type_name, COUNT(*) as count, AVG(v.stop_duration) as avg_duration
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE DATE(v.violation_timestamp) BETWEEN DATE('now', '-7 days') AND DATE('now')
GROUP BY vt.type_name;

-- High-risk times (peak violation hours)
SELECT STRFTIME('%H', v.violation_timestamp) as hour, COUNT(*) as count
FROM violations v
WHERE DATE(v.violation_timestamp) = DATE('now')
GROUP BY hour
ORDER BY count DESC;
```

### Compliance & Enforcement
```sql
-- High-confidence violations (reportable)
SELECT v.id, v.violation_timestamp, vt.type_name, v.image_path, v.confidence
FROM violations v
LEFT JOIN vehicle_types vt ON v.vehicle_type_id = vt.id
WHERE v.confidence > 0.90 AND v.status = 'recorded'
ORDER BY v.confidence DESC, v.violation_timestamp DESC;

-- Repeat offenders (same vehicle by tracking ID)
SELECT detection_id, COUNT(*) as count, 
       DATE(violation_timestamp) as last_violation
FROM violations
WHERE SUBSTR(detection_id, -3) IN (       -- Filter by object ID suffix
    SELECT SUBSTR(detection_id, -3)
    FROM violations
    GROUP BY SUBSTR(detection_id, -3)
    HAVING COUNT(*) > 3
)
GROUP BY SUBSTR(detection_id, -3);
```

---

## Storage Estimates

For a 24/7 operation with average 20 violations/hour:

```
Per Day:
  - 480 violations
  - ~1.5 GB images (if storing blobs @ 3.5 MB/image)
  
Per Month:
  - ~14,400 violations
  - ~45 GB images

Per Year:
  - ~175,200 violations
  - ~540 GB images
  
Recommendation:
  - Store image blobs externally (S3, MinIO)
  - Keep image_path as reference in database
  - Archive violations older than 6 months
```


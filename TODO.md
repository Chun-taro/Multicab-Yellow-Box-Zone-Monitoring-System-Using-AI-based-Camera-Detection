# TODO List for Multicab Yellow Box Zone Monitoring System

## Project Configuration
- [x] Create requirements.txt with Python dependencies (AI components)
- [x] Create config/config.py (system settings)
- [x] Create README.md (project description)

## AI & Camera Module (Python)
- [x] Create ai_model/detect.py (YOLOv5 detection code)
- [x] Create ai_model/tracker.py (object tracking)
- [x] Create ai_model/stop_timer.py (stop-time computation)
- [x] Create camera/camera.py (camera/video handler)
- [x] Create utils/zone.py (yellow box zone coordinates)
- [x] Create utils/helpers.py (helper functions)
- [x] Create tests/test_detection.py (basic testing)
- [x] Update AI module to send detection data to Node.js Backend or MongoDB (with SQLite fallback)

## Backend (Node.js & MongoDB)
- [x] Initialize `backend/` directory
- [x] Create `backend/package.json` (Node dependencies)
- [x] Create `backend/server.js` (Main entry point)
- [x] Create `backend/config/db.js` (MongoDB connection handled in server.js)
- [x] Create `backend/models/Violation.js` (Mongoose Schema)
- [x] Create `backend/routes/apiRoutes.js` (API endpoints)
- [x] Create `backend/controllers/violationController.js` (Logic for fetching/storing logs)

## Frontend (React)
- [x] Initialize `frontend/` directory (React App)
- [x] Create `frontend/package.json`
- [x] Create `frontend/src/App.js` (Main component)
- [x] Create `frontend/src/components/Dashboard.jsx` (Real-time monitoring view)
- [x] Create `frontend/src/components/Logs.jsx` (Violation history view)
- [x] Create `frontend/src/services/api.js` (Axios service for backend communication)
- [x] Create `frontend/src/styles/dashboard.css` (Dashboard styling)

## Integration & Deployment
- [ ] Install Python dependencies
- [ ] Install Node.js dependencies (Backend & Frontend)
- [ ] Run MongoDB service
- [ ] Test full system integration (Camera -> AI -> Backend -> Frontend)

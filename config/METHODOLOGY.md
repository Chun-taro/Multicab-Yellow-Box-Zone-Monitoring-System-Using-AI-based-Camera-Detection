# Methodology

This document describes the materials, data sources, research design, and procedures used to develop the **Multicab Yellow Box Zone Monitoring System Using AI-based Camera Detection**.

## 1. Materials

### Software
The proposed system will be developed using the following software tools:
*   **Python 3.10**: Primary programming language for developing the AI and computer vision components.
*   **OpenCV**: For video frame processing, object tracking, and motion detection.
*   **YOLOv5 Framework (PyTorch-based)**: For real-time multicab detection and classification.
*   **React & NodeJS**: For backend and web dashboard development.
*   **MongoDB**: For storing logs, detection results, and timestamps.
*   **LabelImg**: For annotating training images.
*   **Google Colab**: For GPU-accelerated model training.
*   **CSS**: For dashboard interface design.

### Hardware
The hardware setup will simulate a real-world monitoring environment:
*   **High-definition CCTV Camera**: 1080p, 30 fps, for capturing multicab behavior at intersections and yellow box zones.
*   **Desktop Computer**: Intel i7, 16GB RAM, NVIDIA GTX 1660 GPU, for model training and testing.
*   **Network Router and Ethernet Connection**: To link CCTV input to the AI system.
*   **Monitor Display**: For the Traffic Management Center (TMC) Dashboard to review detected violations.

## 2. Data

The dataset will consist of real-world traffic footage collected from Malaybalay City intersections.

*   **Source of Data**:
    *   **TMC CCTV Footage**: Video recordings from existing traffic cameras installed at key intersections in Malaybalay City, Bukidnon.
    *   **Supplementary Public Datasets**: Open-source traffic video datasets available online for training augmentation.
*   **Type of Data**: Video frames and annotated images of multicabs, tricycles, and cars.
*   **Year of Acquisition**: 2025 – 2026.
*   **Data Volume**: Approximately 8,000 annotated images from 10 hours of footage.
*   **Data Processing**: Manual labeling using LabelImg for YOLOv5 training.

## 3. Research Design

This study uses a **Developmental Research Design**, focusing on the design, implementation, and evaluation of an AI-based system prototype. The research follows the **Waterfall Model**, including sequential phases:
1.  Plan
2.  Develop
3.  Implement
4.  Evaluate
5.  Maintenance

## 4. Procedures

### Phase 1: System Analysis and Requirements Gathering
*   Conduct interviews with TMC personnel to identify violations and challenges.
*   Assess existing CCTV infrastructure for feasibility.
*   Document functional (accuracy, response time) and non-functional requirements.

### Phase 2: System Design and Modeling
*   **Use Case Diagram**: Outlines interactions between the system and TMC officers.
*   **Data Flow Diagram (DFD)**: Illustrates data flow from CCTV to the dashboard.
*   **System Architecture**: Shows communication between Hardware, AI Unit, Backend, and Dashboard.
*   **Flowchart**: Represents the process from video input to alert generation.
*   **Database Schema**: Defines tables for violations, timestamps, and vehicle classes.

### Phase 3: Data Preprocessing
*   Segment CCTV footage into frames.
*   Perform image preprocessing (resizing, normalization, noise reduction).
*   Annotate images using LabelImg (classes: multicabs, tricycles, private cars).
*   Augment dataset (rotation, brightness, flipping) for robustness.

### Phase 4: Model Training and System Implementation
*   Train YOLOv5 model on Google Colab (GPU).
*   Optimize hyperparameters (batch size, learning rate, epochs).
*   Evaluate using Precision, Recall, F1-score, and mAP.
*   Integrate tracking algorithms (**DeepSORT** or **ByteTrack**) for object identity.
*   Develop stop-time computation module (15-second limit).

### Phase 5: System Integration and Implementation
*   **Backend**: Node.js service for data storage and API communication.
*   **Database**: MongoDB for storing detection results and images.
*   **Dashboard**: React-based web interface for TMC officers to:
    *   View real-time feeds with overlays.
    *   Review logged violations.
    *   Access historical data and generate reports.
*   **Scheduling**:
    *   **Peak Hours (Real-time detection)**: 6:00–9:00 AM and 4:00–7:00 PM.
    *   **Off-peak (Trend analysis)**: 10:00 AM–3:00 PM.

## 5. Violation Documentation (NCAP-Based)

The system adopts a **No Contact Apprehension Policy (NCAP)** approach:
1.  **Detection**: System detects a multicab stopping in the yellow box > 15 seconds.
2.  **Recording**: Automatically records timestamped visual evidence (frames, duration, location).
3.  **Storage**: Records are stored in MongoDB and sent to the TMC dashboard.
4.  **Verification**: Authorized TMC personnel verify the violation.
5.  **Enforcement**: Local government handles the issuance of notices based on verified records.

## 6. Handling Multiple Multicabs

The system uses multi-object tracking (YOLO + Tracker) to:
*   Assign unique IDs to each vehicle.
*   Track stop duration independently for each vehicle.
*   Ensure overlapping vehicles do not interfere with tracking.

## 7. Evaluation

The system is evaluated by TMC personnel based on **Functionality**, **Usability**, and **Reliability** using a 5-point Likert scale.

### Evaluation Criteria
*   **Functionality**: Accuracy of detection, stop-time measurement, and report generation.
*   **Usability**: Ease of navigation, user confidence, and learning curve.
*   **Reliability**: Performance under peak traffic and varying environmental conditions (rain, night, occlusion).

### Testing Procedures
*   **Unit Testing**: Individual components (detection, backend).
*   **Integration Testing**: Data flow between modules.
*   **Field Testing**: Real-world performance using Malaybalay CCTV footage.
*   **Metrics**: Detection Accuracy, Stop-Time Accuracy, FPS/Latency.

---
*Document generated based on project methodology.*
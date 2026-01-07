import cv2
import sys
import os

# Add project root to path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import config

def select_coordinates():
    # Open video source
    cap = cv2.VideoCapture(config.camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {config.camera_source}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame from video.")
        return

    # Resize frame to match the processing dimensions defined in config
    # This is crucial because coordinates must match the frame size used by the AI
    frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

    print(f"Frame resized to {config.FRAME_WIDTH}x{config.FRAME_HEIGHT} for coordinate selection.")
    print("Click on the 4 corners of the zone in order (e.g., Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left).")
    print("Press any key to exit when done.")

    coordinates = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            print(f"Clicked: ({x}, {y})")
            
            # Draw a circle and text
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(len(coordinates)), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw lines to visualize the polygon
            if len(coordinates) > 1:
                cv2.line(frame, coordinates[-2], coordinates[-1], (255, 0, 0), 2)
            if len(coordinates) == 4:
                # Close the loop
                cv2.line(frame, coordinates[-1], coordinates[0], (255, 0, 0), 2)
                
            cv2.imshow("Select Zone Coordinates", frame)

    cv2.imshow("Select Zone Coordinates", frame)
    cv2.setMouseCallback("Select Zone Coordinates", click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(coordinates) == 4:
        print("\n--- COPY THIS TO config/config.py ---")
        print("YELLOW_BOX_ZONE = [")
        for x, y in coordinates:
            print(f"    ({x}, {y}),")
        print("]")
        print("-------------------------------------")
    else:
        print(f"\nWarning: You selected {len(coordinates)} points. 4 points are expected.")

if __name__ == "__main__":
    select_coordinates()
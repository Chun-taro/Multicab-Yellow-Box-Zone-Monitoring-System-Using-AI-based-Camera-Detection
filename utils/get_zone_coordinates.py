import cv2
import numpy as np
import sys
import os

# Add project root to path to allow sibling imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.camera import CameraHandler
from config.config import config

# Global variables
points = []
img = None

def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point
        points.append([x, y])
        
        # Visual feedback: Draw a red dot
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        
        # Draw lines between points
        if len(points) > 1:
            cv2.line(img, tuple(points[-2]), tuple(points[-1]), (255, 0, 0), 2)
        
        # If 4 points are clicked, close the loop and print the code
        if len(points) == 4:
            cv2.line(img, tuple(points[-1]), tuple(points[0]), (255, 0, 0), 2)
            
            # Fill polygon to show the selected zone clearly
            pts = np.array(points, np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            print("\n" + "="*50)
            print("   COPY THE COORDINATES INTO YOUR CONFIGURATION FILE")
            print("="*50)

            # Format for config/config.py
            print("\n# For config/config.py:")
            print("YELLOW_BOX_ZONE = [")
            for p in points:
                print(f"    ({p[0]}, {p[1]}),")
            print("]")
            
            # Format for routes/dashboard_routes.py
            print("\n# For routes/dashboard_routes.py:")
            print("yellow_zone = np.array([")
            for p in points:
                print(f"        [{p[0]}, {p[1]}],")
            print("    ], np.int32).reshape((-1, 1, 2))")
            print("="*50 + "\n")
            print("Done! Press 'q' to quit.")

        cv2.imshow('Set Coordinates', img)

def main():
    global img, points

    # Use the same CameraHandler as the main application for consistency
    camera = CameraHandler()
    if camera.use_placeholder:
        print(f"Error: Could not open camera source '{config.camera_source}'.")
        print("Please check your camera or the 'camera_source' in 'config/config.py'.")
        return

    print(f"Camera source '{config.camera_source}' opened.")
    print("1. Position your camera.")
    print("2. Press SPACEBAR to freeze the frame and start drawing.")
    print("3. Press 'q' to quit without saving.")
    
    while True:
        frame = camera.read_frame()
        if frame is None:
            break
            
        cv2.imshow('Set Coordinates', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Space bar to freeze
            img = frame.copy()
            break
        elif key == ord('q'):
            camera.close()
            cv2.destroyAllWindows()
            return

    print(">> CLICK THE 4 CORNERS OF THE YELLOW BOX NOW <<")
    cv2.setMouseCallback('Set Coordinates', click_event)
    
    while True:
        cv2.imshow('Set Coordinates', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
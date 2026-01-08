import cv2
import numpy as np

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
            print("   COPY THE CODE BELOW INTO routes/dashboard_routes.py")
            print("="*50)
            print("    yellow_zone = np.array([")
            for p in points:
                print(f"        [{p[0]}, {p[1]}],")
            print("    ], np.int32).reshape((-1, 1, 2))")
            print("="*50 + "\n")
            
            print("Done! Press 'q' to quit.")

        cv2.imshow('Set Coordinates', img)

def main():
    global img, points
    
    # Try opening camera (Source 1 first, then 0)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera 1 failed, trying Camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            return

    print("Camera opened.")
    print("1. Position your camera.")
    print("2. Press SPACEBAR to freeze the frame and start drawing.")
    print("3. Press 'q' to quit without saving.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Set Coordinates', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Space bar to freeze
            img = frame.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    print(">> CLICK THE 4 CORNERS OF THE YELLOW BOX NOW <<")
    cv2.setMouseCallback('Set Coordinates', click_event)
    
    while True:
        cv2.imshow('Set Coordinates', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
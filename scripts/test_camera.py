import cv2, time

def test_indices(max_idx=5):
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read()
        print(f"index {i}: opened={cap.isOpened()}, read_ok={ok}")
        cap.release()

if __name__ == "__main__":
    test_indices(5)
    # test configured default
    cap = cv2.VideoCapture(0)
    print("Default 0 opened:", cap.isOpened())
    cap.release()
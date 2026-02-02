import cv2
import time
from multiprocessing import Process, Queue

def capture_frames(frame_queue, running):
    cap = cv2.VideoCapture(0)
    while running.value:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
        time.sleep(2)  # Simulate frame rate
    cap.release()

if __name__ == "__main__":
    pass  # This file is meant to be imported

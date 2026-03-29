import cv2
import time
from multiprocessing import Process, Queue, Value
from camera_capture import capture_frames
from frame_processor import process_frames

def display_frames(output_queue, running):
    while running.value:
        if not output_queue.empty():
            frame = output_queue.get()
            cv2.imshow('Processed Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running.value = False
        time.sleep(0.03)  # Simulate display rate

if __name__ == "__main__":
    frame_queue = Queue(maxsize=10)  # Queue for raw frames
    processed_queue = Queue(maxsize=10)  # Queue for processed frames
    running = Value('b', True)  # Shared boolean flag

    # Create processes
    capture_process = Process(
        target=capture_frames, args=(frame_queue, running)
    )
    process_process = Process(
        target=process_frames, args=(frame_queue, processed_queue, running)
    )
    display_process = Process(
        target=display_frames, args=(processed_queue, running)
    )

    # Start processes
    capture_process.start()
    process_process.start()
    display_process.start()

    # Wait for processes to finish
    capture_process.join()
    process_process.join()
    display_process.join()

    cv2.destroyAllWindows()

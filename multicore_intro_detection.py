import sys
import cv2
import time
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Queue, Array, Lock, Process, Value

N_CORES_USABLE = 4
FRAMES_TO_COUNT = 1000
DEBUG = False

def consumer(process_idx: int, reference, frame_queue: Queue, qs: Value, qs_lock: Lock, scores: Array):
    if DEBUG:
        sys.stdout.write(f"Consumer: This is consumer process {process_idx}\n")
        sys.stdout.flush()

    frames_read = 0

    while True:
        try:
            try:
                st = time.time()
                idx, frame_bytes = frame_queue.get(timeout=1)

                if idx is None:
                    if DEBUG:
                        sys.stdout.write(f"Consumer: Process {process_idx} received sentinel, stopping\n")
                        sys.stdout.flush()
                    frame_queue.put((None, None))
                    break

                frames_read += 1
                if DEBUG:
                    sys.stdout.write(f"Consumer: Process {process_idx}: Read frame {idx}\n")
                    sys.stdout.flush()

                try:
                    ssim_index, _ = ssim(reference, frame_bytes, full=True)
                    if DEBUG:
                        sys.stdout.write(f"Consumer: Process {process_idx}: Frame {idx} SSIM = {ssim_index:.4f}\n")
                        sys.stdout.flush()
                    scores[idx] = ssim_index
                    et = time.time()
                    if DEBUG:
                        print(f"Consumer: took {et-st} seconds to read and process a frame")
                except Exception as ssim_error:
                    if DEBUG:
                        sys.stdout.write(f"Consumer: Process {process_idx}: SSIM error for frame {idx}: {ssim_error}\n")
                        sys.stdout.flush()
                        # Print shapes for debugging
                        sys.stdout.write(f"Consumer: Reference shape: {reference.shape}, Frame shape: {frame_bytes.shape}\n")
                        sys.stdout.flush()

            except:
                with qs_lock:
                    if qs.value:
                        sys.stdout.write(f"Consumer: Process {process_idx} - producer done, exiting\n")
                        sys.stdout.flush()
                        break
                continue
            pass
        except Exception as e:
            sys.stdout.write(f"Consumer: Process {process_idx} error: {e}\n")
            sys.stdout.flush()
            break
    sys.stdout.write(f"Consumer: Process {process_idx} finished, read {frames_read} frames\n")
    sys.stdout.flush()

    sys.stdout.write(f"Consumer: queue empty, process {process_idx} read {frames_read} frames\n")
    sys.stdout.flush()


def parse_args():
    if len(sys.argv) != 3:
        exit("Please provide file name and end frame picture path")

    filename = sys.argv[1]
    ef_path = sys.argv[2]

    return filename, ef_path

def find_intro(fname: str, path: str):
    end_frame = cv2.imread(path)
    if end_frame is None:
        print("Error: Could not load reference image")
        return

    gray_png = cv2.cvtColor(end_frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale_reference_frame.png", gray_png)

    frame_queue = Queue()
    queue_sentinel = Value("b", False)
    scores = Array("f", [0]*FRAMES_TO_COUNT)
    queue_sentinel_lock = Lock()

    cap = cv2.VideoCapture(fname)
    frame_offset = 0
    cur = 0
    prev = 0
    found_frame = False

    frame_queue.put((None, None))
    while True:
        frame_queue.get()
        for i in range(FRAMES_TO_COUNT):
            st = time.time()
            ret, frame = cap.read()

            if not ret:
                if DEBUG:
                    print(f"Producer: Ran out of frames at frame {i}")
                    sys.stdout.flush()
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray_frame.shape != end_frame.shape:
                gray_frame = cv2.resize(gray_frame, (end_frame.shape[1], end_frame.shape[0]))

            try:
                frame_queue.put((i, gray_frame))
                et = time.time()
                if DEBUG:
                    print(f"Prodcuer: Frame {i+frame_offset} took {et-st} seconds to push to quueue")
                    sys.stdout.flush()
            except Exception as e:
                print(f"Producer: Failed to put frame {i}: {e}")
                sys.stdout.flush()
                break
        frame_queue.put((None, None), timeout=2)
        consumer_processes = [Process(target=consumer, args=(i, gray_png, frame_queue, queue_sentinel, queue_sentinel_lock, scores)) for i in range(N_CORES_USABLE)]

        for p in consumer_processes:
            p.start()

        for p in consumer_processes:
            p.join()

        for i in range(FRAMES_TO_COUNT):
            print(f"Score at index {i+frame_offset}: {scores[i]}")
            prev = cur
            cur = scores[i]
            f = i + frame_offset

            if prev - cur > 0.5:
                fps = cap.get(cv2.CAP_PROP_FPS)
                timestamp = int(f / fps)
                print(f"Intro end timestamp {timestamp//60}:{timestamp - ((timestamp//60)*60)}")
                print(f"End Frame found: Frame {f-1}")
                found_frame = True
                break
        if found_frame:
            break
        frame_offset += FRAMES_TO_COUNT

    print("Done")


if __name__ == '__main__':
    fname, path = parse_args()
    find_intro(fname, path)

import sys
import cv2
from skimage.metrics import structural_similarity as ssim

FRAME_COUNT = 2500
filename = ""
ff_path = ""
scores = []

def parse_args():
    if len(sys.argv) != 3:
        exit("Please provide file name and end frame picture path")

    filename = sys.argv[1]
    ff_path = sys.argv[2]
    return filename, ff_path


def find_intro(fname: str, path: str):
    cap = cv2.VideoCapture(fname)
    first_frame = cv2.imread(path)

    i = 0
    cur = 0
    while i < FRAME_COUNT:
        ret, frame = cap.read()

        if not ret:
            print("Ran out of frames")
            break

        if frame.shape != first_frame.shape:
            frame = cv2.resize(frame, (first_frame.shape[1], first_frame.shape[0]))

        gray_png = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ssim_index, _ = ssim(gray_png, gray_frame, full=True)
        print(f"{i},{ssim_index}")
        cur = ssim_index

        scores.append(ssim_index)
        i += 1

        if cur > 0.8:
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp = int(i / fps)
            print(f"Intro start timestamp {timestamp//60}:{timestamp - (timestamp//60)}")
            print(f"End Frame found: Frame {i-2}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    fname, path = parse_args()
    find_intro(fname, path)

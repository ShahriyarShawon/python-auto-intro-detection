import sys
import cv2
from skimage.metrics import structural_similarity as ssim

scores = []

def parse_args():
    if len(sys.argv) != 3:
        exit("Please provide file name and end frame picture path")

    filename = sys.argv[1]
    ef_path = sys.argv[2]

    return filename, ef_path

def find_intro(fname: str, path: str):
    cap = cv2.VideoCapture(fname)
    end_frame = cv2.imread(path)

    i = 0
    prev = 0
    cur = 0

    gray_png = cv2.cvtColor(end_frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale_reference_frame.png", gray_png)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ran out of frames")
            break

        if frame.shape != end_frame.shape:
            frame = cv2.resize(frame, (end_frame.shape[1], end_frame.shape[0]))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ssim_index, _ = ssim(gray_png, gray_frame, full=True)
        print(f"{i},{ssim_index}")
        prev = cur
        cur = ssim_index

        scores.append(ssim_index)
        i += 1

        diff = cur - prev
        if diff < -0.4:
            cv2.imwrite("grayscale_frame.png", gray_frame)
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp = int(i / fps)
            print(f"Intro end timestamp {timestamp//60}:{timestamp - ((timestamp//60)*60)}")
            print(f"End Frame found: Frame {i-2}")
            break

    with open("out.csv", "w") as f:
        for i in range(len(scores)):
            f.write(f"{i},{scores[i]}\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fname, path = parse_args()
    find_intro(fname, path)

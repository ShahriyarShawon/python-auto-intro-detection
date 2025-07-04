import sys
import cv2
from skimage.metrics import structural_similarity as ssim
# from pytorch_msssim import ssim

# FRAME_COUNT = 2500
if len(sys.argv) != 3:
    exit("Please provide file name and end frame picture path")

filename = sys.argv[1]
ef_path = sys.argv[2]
scores = []

cap = cv2.VideoCapture(filename)
end_frame = cv2.imread(ef_path)

i = 0
prev = 0
cur = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Ran out of frames")
        break

    if frame.shape != end_frame.shape:
        frame = cv2.resize(frame, (end_frame.shape[1], end_frame.shape[0]))

    gray_png = cv2.cvtColor(end_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ssim_index, _ = ssim(gray_png, gray_frame, size_average=False)
    ssim_index, _ = ssim(gray_png, gray_frame, full=True)
    print(f"{i},{ssim_index}")
    prev = cur
    cur = ssim_index

    scores.append(ssim_index)
    i += 1

    diff = cur - prev
    if diff < -0.4:
        fps = 23.976023976023978
        timestamp = int(i / fps)
        print(f"Intro end timestamp {timestamp}")
        print(f"End Frame found: Frame {i-2}")
        break

with open("out.csv", "w") as f:
    for i in range(len(scores)):
        f.write(f"{i},{scores[i]}\n")
cap.release()
cv2.destroyAllWindows()

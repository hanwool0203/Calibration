import cv2
import numpy as np
import os
import re
import yaml

# 1. 체커보드 내부 코너 수와 칸의 물리 크기(mm)
CHECKERBOARD = (10, 7)
square_size = 24  # mm

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D world points
imgpoints = []  # 2D image points

# 2. 이미지 파일명 자동 인식 (image_00.jpg, image_01.jpg 등)
image_folder = "useimg"
pattern = re.compile(r'image_\d+\.(jpg|JPG|jpeg|JPEG)')
file_list = [f for f in os.listdir(image_folder) if pattern.match(f)]
file_list.sort(key=lambda fname: int(re.search(r'_(\d+)', fname).group(1)))
image_paths = [os.path.join(image_folder, f) for f in file_list]

if not image_paths:
    raise RuntimeError("images 폴더에 image_00.jpg, image_01.jpg 등의 파일이 없습니다. 파일명을 확인하세요.")

gray = None
for fname in image_paths:
    img = cv2.imread(fname)
    if img is None:
        print(f"이미지를 불러오지 못함: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret:
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
        objpoints.append(objp)
        draw_img = img.copy()
        cv2.drawChessboardCorners(draw_img, CHECKERBOARD, corners, ret)
        cv2.imshow('Corners', draw_img)
        cv2.waitKey(100)
cv2.destroyAllWindows()

N_OK = len(objpoints)
if N_OK < 5:
    raise RuntimeError("체커보드를 정상적으로 찾은 이미지가 5장 미만입니다. 다양한 각도·위치·조명으로 여러 장을 다시 촬영해 보세요.")

print(f"Found {N_OK} valid images for calibration")

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

img_shape = gray.shape[::-1]

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    img_shape,
    K,
    D,
    rvecs,
    tvecs,
    calib_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print("RMS error:", rms)
print("Camera Matrix (K):\n", K)
print("Distortion Coefficients (D):\n", D.ravel())

# 파라미터 yaml로 저장
calib_data = {
    'K': K.tolist(),
    'D': D.ravel().tolist(),
    'image_width': img_shape[0],
    'image_height': img_shape[1],
    'rms': float(rms),
    'CHECKERBOARD': CHECKERBOARD,
    'square_size': square_size
}
with open('fisheye_calib.yaml', 'w') as f:
    yaml.dump(calib_data, f, sort_keys=False)

print("캘리브레이션 결과를 fisheye_calib.yaml에 저장했습니다.")

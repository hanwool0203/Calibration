import cv2
import numpy as np
import yaml

# (1) 입력 읽기 (쌍의 개수, shape 등 점검)
lidar_points = np.loadtxt('clicked_points.csv', delimiter=',').astype(np.float32)      # (N,3)
image_points = np.loadtxt('clicked_img_points.csv', delimiter=',').astype(np.float32)  # (N,2)

assert lidar_points.shape[0] == image_points.shape[0] and lidar_points.shape[0] >= 4
assert lidar_points.shape[1] == 3 and image_points.shape[1] == 2

with open('fisheye_calib.yaml') as f:
    calib = yaml.load(f, Loader=yaml.FullLoader)
K = np.array(calib['K'])

# (2) solvePnP (rvec: 3x1, tvec: 3x1)
success, rvec, tvec = cv2.solvePnP(
    lidar_points, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE
)
print("solvePnP success:", success)
print("rvec(3x1):", rvec.ravel())
print("tvec(3x1):", tvec.ravel())

# (3) rvec(3x1) → R(3x3 회전행렬) 변환
R, _ = cv2.Rodrigues(rvec)
print("R(3x3):\n", R)

# (4) 외부 파라미터 전체(3x4) 행렬로 결합
extrinsic = np.hstack([R, tvec])
print("Extrinsic matrix (3x4):\n", extrinsic)

# rvec(3x1): [ 1.21786007 -1.14826187  1.17055874]
# tvec(3x1): [-0.02434201  0.10524078  0.18248737]
# R(3x3):
#  [[ 0.06264942 -0.9980284  -0.00378974]
#  [ 0.02298057  0.00523874 -0.99972219]
#  [ 0.99777099  0.06254492  0.02326346]]
# Extrinsic matrix (3x4):
#  [[ 0.06264942 -0.9980284  -0.00378974 -0.02434201]
#  [ 0.02298057  0.00523874 -0.99972219  0.10524078]
#  [ 0.99777099  0.06254492  0.02326346  0.18248737]]

# 왜곡 보정
# solvePnP success: True
# rvec(3x1): [ 1.19614343 -1.26892236  1.26731285]
# tvec(3x1): [0.0508615  0.09402993 0.17726318]
# R(3x3):
#  [[-0.07423452 -0.99712001  0.01552157]
#  [-0.01678712 -0.01431283 -0.99975664]
#  [ 0.99709951 -0.07447702 -0.01567627]]
# Extrinsic matrix (3x4):
#  [[-0.07423452 -0.99712001  0.01552157  0.0508615 ]
#  [-0.01678712 -0.01431283 -0.99975664  0.09402993]
#  [ 0.99709951 -0.07447702 -0.01567627  0.17726318]]

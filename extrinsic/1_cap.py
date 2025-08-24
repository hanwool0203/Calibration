import cv2
import numpy as np
cap = cv2.VideoCapture(0)  # 카메라 인덱스 0, 필요시 변경

# calibration config
# 캘리브레이션 설정
img_size = (640, 480)
warp_img_w, warp_img_h, warp_img_mid = 650, 120, 60

# 한울님이 측정한 camera intrinsic params (from fisheye_calib.yaml )
# 한울님이 측정한 카메라 내부 파라미터 (fisheye_calib.yaml 파일에서 가져옴)
mtx = np.array([[329.7890410051319, 0.000000, 320.6057545861696],
                        [0.000000, 329.76662478183584, 229.13605526487663],
                        [0.000000, 0.000000, 1.000000]])
dist = np.array([-0.00845590148493767, -0.020724735072096084, 0.007699567026492894, 0.004404855849468008])

# 어안 렌즈 캘리브레이션을 위해 cv2.fisheye 모듈 사용
# cv2.fisheye.initUndistortRectifyMap 함수를 사용하여 undistortion 및 rectification 맵 계산
cal_mtx = mtx # 어안 렌즈 캘리브레이션에서는 새로운 카메라 행렬을 별도로 계산하지 않고 기존 mtx를 사용하거나 필요에 따라 조정합니다.
new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, img_size, np.eye(3), balance=0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), new_mtx, img_size, cv2.CV_32FC1)

def to_calibrated(img, show=False):
    # 어안 렌즈 이미지 왜곡 보정
    # cv2.remap 함수와 initUndistortRectifyMap에서 계산된 맵을 사용하여 왜곡 보정 수행
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if show:
        cv2.imshow('calibrated', img_undistorted)
    return img_undistorted

# perspective config
# 원근 변환 설정
warpx_mid, warpx_margin_hi, warpx_margin_lo, warpy_hi, warpy_lo, tilt = 320, 240, 319, 345, 375, 0
h_bias = -25
# 원근 변환 소스 포인트 설정 (어안 렌즈 왜곡 보정 후 이미지에 맞게 조정 필요)
# 이 부분은 실제 환경에서 캘리브레이션 마커 등을 사용하여 정확하게 설정해야 합니다.
warp_src  = np.array([[warpx_mid+tilt-warpx_margin_hi, warpy_hi+h_bias], [warpx_mid+tilt+warpx_margin_hi, warpy_hi+h_bias],
                    [warpx_mid+warpx_margin_lo, warpy_lo+h_bias], [warpx_mid-warpx_margin_lo,  warpy_lo+h_bias]], dtype=np.float32)
# 원근 변환 결과 이미지의 목적지 포인트 설정
warp_dist = np.array([[50, 0], [649-50, 0],
                [649-50, 119], [50, 119]], dtype=np.float32)
M = cv2.getPerspectiveTransform(warp_src, warp_dist)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    undistorted_frame = to_calibrated(frame, show=False)
    cv2.imshow('Camera', frame)
    cv2.imshow('Undistorted', undistorted_frame)
    key = cv2.waitKey(1)
    if key == ord('c'):  # c키 누를 때마다 저장
        cv2.imwrite(f'/home/xytron/calibration/extrinsic/image_{count:02d}.jpg', undistorted_frame)
        count += 1
        print(f"Saved image_{count:02d}.jpg")
    elif key == 27:  # ESC누르면 종료
        break
cap.release()
cv2.destroyAllWindows()




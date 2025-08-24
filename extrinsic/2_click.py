import cv2

# ① 이미지 불러오기
img = cv2.imread('click.jpg')  # 원하는 이미지 파일명으로 수정

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 시
        clicked_points.append((x, y))
        print(f"Clicked point: {x}, {y}")
        # 클릭 지점에 작은 원 그려서 피드백
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('img', img)

cv2.imshow('img', img)
cv2.setMouseCallback('img', mouse_callback)

print("마우스로 (특이점 순서대로) 원하는 점을 클릭하세요. Enter(또는 아무 키)로 종료.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("선택된 이미지 좌표 (u,v) 리스트:", clicked_points)

# 필요시 파일로 저장
import numpy as np
np.savetxt('clicked_img_points.csv', np.array(clicked_points), fmt='%d', delimiter=',')

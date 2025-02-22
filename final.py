import cv2
import numpy as np


def sort_corners(corners):
    """ 四边形角点排序函数 """
    points = corners.reshape(4, 2)
    sorted_y = points[np.argsort(points[:, 1])]
    top = sorted_y[:2]
    bottom = sorted_y[2:]
    top_sorted = top[np.argsort(top[:, 0])]
    tl, tr = top_sorted[0], top_sorted[1]
    bottom_sorted = bottom[np.argsort(-bottom[:, 0])]
    br, bl = bottom_sorted[0], bottom_sorted[1]
    return np.array([tl, tr, br, bl]).reshape(4, 1, 2)


def ShapeDetection(img, imgContour):
    """ 形状检测与姿态解算主函数 """
    # 相机内参配置
    fx = 1120.0
    fy = 1120.0
    cx = 2240 / 2
    cy = 1260 / 2
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # 图像预处理
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 60, 60)

    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    quadrilaterals = []

    for obj in contours:
        perimeter = 0.03 * cv2.arcLength(obj, True)
        approx = cv2.approxPolyDP(obj, perimeter, True)
        CornerNum = len(approx)
        area = cv2.contourArea(approx)

        if CornerNum == 4 and area > 15000:
            approx = sort_corners(approx)
            x, y, w, h = cv2.boundingRect(approx)

            # 绘制检测图形
            cv2.drawContours(imgContour, [approx], -1, (255, 0, 0), 4)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imgContour, "Tag", (x + (w // 2), y + (h // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 姿态解算
            tag_size = 150.0  # mm
            half_size = tag_size / 2.0
            object_points = np.array([
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            image_points = approx.astype(np.float32).reshape(4, 2)

            success, rvec, tvec = cv2.solvePnP(object_points,
                                               image_points,
                                               camera_matrix,
                                               dist_coeffs)

            if success:
                # 计算三维坐标
                pos_mm = tvec.flatten()
                distance = np.linalg.norm(tvec)

                # 计算法线夹角
                R, _ = cv2.Rodrigues(rvec)
                normal_vector = R[:, 2]
                cos_theta = np.dot(normal_vector, [0, 0, 1])
                angle_deg = np.degrees(np.arccos(cos_theta))

                height, width, _ = imgContour.shape

                # 标注结果
                cv2.putText(imgContour, f"Dist: {distance:.2f}mm", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)
                cv2.putText(imgContour, f"Angle: {angle_deg:.2f}deg", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

                print(f"三维坐标（mm）: X={pos_mm[0]}, Y={pos_mm[1]}, Z={pos_mm[2]}")
                print(f"距离: {distance}mm")
                print(f"法线夹角: {angle_deg}°\n")

    return imgContour


# 主程序
if __name__ == "__main__":
    # 读取原始图像并检测红色区域
    img_bgr = cv2.imread('nav.jpg')
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 红色阈值范围
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # 创建红色掩膜
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 应用掩膜
    img_tag = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_red)
    img_contour = img_tag.copy()

    # 执行检测与解算
    result_img = ShapeDetection(img_tag, img_contour)

    # 显示并保存结果
    cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Detection Result", result_img)
    cv2.imwrite("final_result.jpg", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
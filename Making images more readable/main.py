import cv2


img = cv2.imread("page.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (560, 900))

# Thresholding with single global vairable
_, result = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

# Adaptive thresholding
adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 41, 5)


cv2.imshow("result", result)
cv2.imshow("original", img)
cv2.imshow("adaptive threshold", adaptive_result)

import cv2 as cv
import numpy as np

# Read the image
image_path = "C:\\Users\Prasanna Kumar\\Downloads\\much_better_images\\dark_fullcolor_l.png"
image = cv.imread(image_path)

# Convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
gray_blur = cv.GaussianBlur(gray, (9, 9), 0)

# Enhance edges using bilateral filtering
edges = cv.bilateralFilter(gray_blur, d=7, sigmaColor=75, sigmaSpace=75)
edges = cv.Canny(edges, 50, 150)

# Detect lines using the probabilistic Hough transform
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

# Draw detected lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Concatenate the original image and the image with detected lines horizontally
concatenated_image = np.concatenate((image, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)), axis=1)

# Display the concatenated image
cv.imshow('Original and Detected Lines', concatenated_image)
cv.waitKey(0)
cv.destroyAllWindows()


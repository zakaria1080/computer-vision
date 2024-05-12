import cv2

# reading the image
img = cv2.imread(r"C:\Users\zakar\Documents\Cours\Projets\Projet MA1\mug\Mug_3.jpg")
# convert to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create SIFT feature extractor
sift = cv2.SIFT_create()

# detect features from the image
keypoints, descriptors = sift.detectAndCompute(gray, None)

# draw the detected key points on the original image
sift_image = cv2.drawKeypoints(gray, keypoints, img)

# Redimensionner l'image pour la visualisation
# Vous pouvez ajuster les dimensions (width, height) selon vos besoins
height, width = sift_image.shape[:2]
resized_image = cv2.resize(sift_image, (int(width * 0.3), int(height * 0.3)))

# show the resized image
cv2.imshow('Resized Image', resized_image)

# save the resized image
cv2.imwrite("table-sift-resized.jpg", resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

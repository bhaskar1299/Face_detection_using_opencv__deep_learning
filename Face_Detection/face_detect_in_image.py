# TO RUN THE SCRIPT USE
# python face_detect_in_image.py --image images/img_1.jpg --prototxt face_detector/deploy.prototxt.txt --model face_detector/res10_300x300_ssd_iter_140000.caffemodel
# Make sure you have downloaded pre trained model files

# import necessary packages
import numpy as np
import argparse
import cv2

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to deploy prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained caffe model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="probability to filter weak detections of face")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load image and create blob for the image and resize it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections 
	if confidence > args["confidence"]:
		# compute the (x, y) coordinates of the bounding box
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# probability of detected face
		prob_text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		# draw bounding box along the face
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
		# write probability obtained
		cv2.putText(image, prob_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# show output
cv2.imshow("Output", image)
cv2.waitKey(0)

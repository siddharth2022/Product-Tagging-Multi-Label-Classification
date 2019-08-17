# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import csv

with open('../Configuration/param.csv', newline='') as csvfile :
	reader = csv.reader(csvfile)
	param = {'param':'value'}
	for row in reader:
		param.update({row[0]:row[1]})


# construct the argument parse and parse the arguments


model=param['model']
labelbin=param['label']
image=param['image']
outputfile=param['output']
image_name=image
# load the image
image = cv2.imread(image)
output = imutils.resize(image, width=400)
 
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model(model)
mlb = pickle.loads(open(labelbin, "rb").read())

# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

out=[]
out.append(image_name)

for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	out.append(mlb.classes_[j])
	print(label)	


with open(outputfile,'a') as f:
    writer = csv.writer(f)
    writer.writerow(out)

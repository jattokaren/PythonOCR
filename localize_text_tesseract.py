# USAGE
# python localize_text_tesseract.py --image 'apple_support2.png'
# python localize_text_tesseract.py --image 'apple_support2.png' --min-conf 20

# import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-c", "--min-conf", type=int, default=0,
	help="mininum confidence value to filter weak text detection")
args = vars(ap.parse_args())

# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to localize each area of text in the input image
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
	# extract the bounding box coordinates of the text region from
	# the current result
	x = results["left"][i]
	y = results["top"][i]
	w = results["width"][i]
	h = results["height"][i]

	# extract the OCR text itself along with the confidence of the
	# text localization
	text = results["text"][i]
	conf = int(results["conf"][i])

	# filter out weak confidence text localizations
	if conf > args["min_conf"]:
		# display the confidence and text to our terminal
		print("Confidence: {}".format(conf))
		print("Text: {}".format(text))
		print("")

		# strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw a bounding box around the text along
		# with the text itself
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(image, str(text + ' (' + str(conf) +')'), (x, y - 10), 
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)


### Testing ###
#Input File
fullfilename = args["image"]
print(fullfilename)
filename = fullfilename[:-4]     # Exclude last 4 characters
print(filename)

# OCR Processing Results
print('pytesseract output dictionary')
print(len(results))
print(results.keys())
print(results.values())

# dict_items([
# 	('level', [1, 2, 3, 4, 5, 5, 4, 5, 2, 3, 4, 5, 5, 4, 5]), 
# 	('page_num', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 
# 	('block_num', [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]), 
# 	('par_num', [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]), 
# 	('line_num', [0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 1, 2, 2]), 
# 	('word_num', [0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 1, 2, 0, 1]), 
# 	('left', [0, 104, 104, 104, 104, 276, 104, 104, 103, 103, 104, 104, 279, 103, 103]), 
# 	('top', [0, 52, 52, 52, 52, 52, 92, 92, 176, 176, 176, 176, 176, 216, 216]), 
# 	('width', [554, 317, 317, 317, 160, 145, 133, 133, 258, 258, 257, 160, 82, 193, 193]), 
# 	('height', [322, 72, 72, 31, 31, 25, 32, 32, 71, 71, 31, 31, 24, 31, 31]), 
# 	('conf', ['-1', '-1', '-1', '-1', 95, 96, '-1', 96, '-1', '-1', '-1', 96, 96, '-1', 96]), 
# 	('text', ['', '', '', '', 'Payment', 'Amount', '', '$593.83', '', '', '', 'Payment', 'Date', '', '03/24/2021'])
# 	])

myOCRoutputDictionary = {
'level': [1, 2, 3, 4, 5, 4, 5, 5, 4, 5], 
'page_num': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
'block_num': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
'par_num': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1], 
'line_num': [0, 0, 0, 1, 1, 2, 2, 2, 3, 3], 
'word_num': [0, 0, 0, 0, 1, 0, 1, 2, 0, 1], 
'left': [0, 88, 88, 88, 88, 197, 197, 390, 200, 200], 
'top': [0, 39, 39, 39, 39, 82, 83, 82, 161, 161], 
'width': [644, 541, 541, 37, 37, 432, 174, 239, 335, 335], 
'height': [256, 161, 161, 42, 42, 64, 63, 64, 39, 39], 
'conf': ['-1', '-1', '-1', '-1', 26, '-1', 96, 96, '-1', 96], 
'text': ['', '', '', '', 'a', '', 'Apple', 'Support', '', '1-800-275-2273']
}

import pandas as pd
import numpy as np
#Output
cv2.imwrite(str(filename+'_pytesseract.png'), image)
print(results)
df = pd.DataFrame.from_dict(results)
print(df)
print(args["min_conf"])

print("list all dictionary 'text' values:")
print(results['text'])
OCRalltext =  " ".join(results['text'])
print(OCRalltext)
test = []
for i in results['text']:
	if i != "":
		test.append(i)
		#print(test)
		#print(i)
print(test)

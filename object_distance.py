'''
MIT License

Copyright (c) 2017, Adam Geitgey

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import cv2
import time
import math
from PIL import Image
import face_recognition

dis = 0.0
font = cv2.FONT_HERSHEY_DUPLEX

start = time.time()


#vid = cv2.VideoCapture('http://192.168.11.7:8000/stream.mjpg')
vid = cv2.VideoCapture(0)

while(vid.isOpened()):

	#resize the jpg file and load into a numpy array
	#img = cv2.imread("002_darken.jpg")
	ret, img = vid.read()
   
	
	#you can keep the ratio of height and width to resize the image
	x = int(img.shape[1]/1)
	r = x/img.shape[1]
	dim = (int(x), int(img.shape[0]*r))
	img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
	#print(img.shape)

	cv2.imwrite('output.jpg', img)
	image = face_recognition.load_image_file("output.jpg")
	
	# Find all the faces in the image using a pre-trained convolutional neural network.
	face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")

	print("I found {} face(s) in this photograph.".format(len(face_locations)))

	for face_location in face_locations:
		
		# Print the location of each face in this image
		top, right, bottom, left = face_location
		print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
	
		w = right - left
		h = bottom - top 
		
		#dis = focal_length * face_height/ (h * your_value)
		dis= 0.367 * 14 / (h * 0.00064)
		
		dis = math.floor(dis)
		
		# You can access the actual face itself like this:
		face_image = image[top:bottom, left:right]
		pil_image = Image.fromarray(face_image)
		#pil_image.show()
		
		cv2.rectangle(image, (left,top), (right,bottom), (0, 255, 0), 3)
	
	end = time.time()
	elapsed = end - start
	msg = 'Time taken: %.2f seconds' % elapsed
	print(msg)	
		
	start = end	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.putText(image,'Distance = ' + str(dis) + ' cm', (325,450),font,1,(255,255,255),2)
	cv2.imshow("face detection", image)
	if cv2.waitKey(1) == 13:
		break
cv2.destroyAllWindows()


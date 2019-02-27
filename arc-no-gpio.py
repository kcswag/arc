
#from collections import deque
from imutils.video import VideoStream
from scipy.spatial import distance
import argparse
import cv2
import imutils
import numpy as np
import time
import math
import dlib
#import RPi.GPIO as GPIO


#initialize pin
# pin = 12
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(pin,GPIO.OUT)
# GPIO.output(pin, GPIO.LOW)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
ap.add_argument("-r","--radius", type=float,default=28.5,help="Radius of roller")
ap.add_argument("-c","--chord", type=float,default=40,help="chord length")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space
greenLower = np.array([55,55,55])
greenUpper = np.array([80,255,255])

blueLower = np.array([86, 31, 4])
blueUpper = np.array([220, 88, 50])

centroidLower = np.array([(165, 145, 100)])
centroidUpper = np.array([(250, 210, 160)])


referenceX = None
referenceY = None
blueInitX = None
blueInitY = None
referenceCenter = None

actualRadius = args.get("radius")
actualChordLength = args.get('chord')


if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(1.0)

font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=800)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	kernel = np.ones((5,5),'int')
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, kernel, iterations=2)

	blueMask = cv2.inRange(hsv,blueLower,blueUpper)
	blueMask = cv2.erode(blueMask,None,iterations=2)
	blueMask = cv2.dilate(blueMask,kernel,iterations=2)

	centroidMask = cv2.inRange(hsv,centroidLower,centroidUpper)   # red
	centroidMask2 = cv2.inRange(hsv,(0, 145, 100), (10, 210, 160)) # red2
	centroidMask = cv2.add(centroidMask,centroidMask2)  #combine 2 limits for red
	centroidMask = cv2.dilate(centroidMask,kernel,iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	#blue contours
	blueCnts = cv2.findContours(blueMask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	blueCnts = imutils.grab_contours(blueCnts)
	blueCenter = None

	#redCnts = cv2.findContours(centroidMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#redCnts = imutils.grab_contours(redCnts)
	#redCenter = None

	# only proceed if at least one contour was found
	cntsLen = len(cnts)
	#redCntsLen = len(redCnts)
	blueCntsLen = len(blueCnts)

	#blue point
	if blueCntsLen > 0:
		b = max(blueCnts, key=cv2.contourArea)
		((blueX,blueY),blueRadius) = cv2.minEnclosingCircle(b)

		blueM = cv2.moments(b)
		if blueM["m00"] == 0:
			continue
		blueCenter = (int(blueM["m10"] / blueM["m00"]), int(blueM["m01"] / blueM["m00"]))
		if blueInitX is None or blueInitY is None:
			blueInitX = blueCenter[0]
			blueInitY = blueCenter[1]

		positionblue = (blueCenter[0] - 70, blueCenter[1] + 70)
		cv2.putText(frame,"("+str(blueCenter[0])+","+str(blueCenter[1])+")",positionblue,font,1,(0,0,255) )

		if blueRadius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(blueX), int(blueY)), int(blueRadius), (0, 255, 255), 2)
			cv2.circle(frame, blueCenter, 5, (0, 0, 255), -1)

	#green point
	if cntsLen > 0 :
		c = max(cnts, key=cv2.contourArea)

		((x, y), radius) = cv2.minEnclosingCircle(c)

		M = cv2.moments(c)
		if M["m00"] == 0:
			continue
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if referenceX is None or referenceY is None or referenceCenter is None:
			referenceX = center[0]
			referenceY = center[1]
			referenceCenter = (referenceX, referenceY)

		cv2.circle(frame, (int(referenceX), int(referenceY)), int(radius), (0, 255, 255), 2)
		cv2.circle(frame, referenceCenter, 5, (0, 0, 255), -1)

		#print(center)
		position1 = (center[0] - 70, center[1] + 70)

		cv2.putText(frame, "("+str(center[0])+","+str(center[1])+")",position1,font,1,(0,0,255))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	#red contours, blue contours and green contours
	if cntsLen and blueCntsLen:
		#draw line from reference center to green center
		cv2.line(frame,referenceCenter,center,(255,0,0),4,cv2.LINE_AA)

		#draw reference center line to red center
		#cv2.line(frame,referenceCenter,redCenter,(255,0,0),4,cv2.LINE_AA)

		#measure blue green radian
		#cv2.line(frame,blueCenter,redCenter,(255,0,0),4,cv2.LINE_AA)

		blueGreenDistance = round(distance.euclidean(center,blueCenter),3)
		pixelsPerMetric = blueGreenDistance / actualChordLength

		#cv2.line(frame, center, redCenter, (255, 0, 0), 4, cv2.LINE_AA)

		#actualRadius = 28.5

		chordLength = round(distance.euclidean(referenceCenter, center),3)
		realChordLength = round(chordLength / pixelsPerMetric,3)

		L_2R = realChordLength/(2 * actualRadius)
		if referenceCenter != center and -1 < L_2R < 1:
			arcLength = 2 * actualRadius * math.asin(realChordLength/(2 * actualRadius))

			if arcLength >= 30.01:
				#GPIO.output(pin,GPIO.HIGH)
				print(arcLength)
				#exit(77)
			
			#else:
				#GPIO.output(pin, GPIO.LOW)

			#position3 = (redCenter[0] - 100, redCenter[1] + 110)

			dataDic = {
				"green X":center[0],
				"green Y":center[1],
				#"centroid X":redCenter[0],
				#"centroid Y":redCenter[1],
				"blue green distance":blueGreenDistance,
				"chord length":chordLength,
				"real chord length":realChordLength,
				"arc length":arcLength
			}
			print(dataDic)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

if not args.get("video", False):
	vs.stop()

else:
	vs.release()

# GPIO.output(pin,GPIO.LOW)
# GPIO.cleanup()
cv2.destroyAllWindows()


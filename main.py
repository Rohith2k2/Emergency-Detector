
import cv2
import imutils
import numpy as np
import whatsapp
from playsound import playsound
from sklearn.metrics import pairwise

bg = None

def run_avg(image, accumWeight):
    global bg
    
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    global bg
    # background-current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None when no contour
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def count(thresholded, segmented):

    chull = cv2.convexHull(segmented)
    #extreme points in convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
    #euclidean distance
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    radius = int(0.8 * maximum_distance)

    circumference = (2 * np.pi * radius)
    # take out the circular ROI
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize count
    count = 0
    for c in cnts:
        # compute the bounding box for hand
        (x, y, w, h) = cv2.boundingRect(c)

        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

if __name__ == "__main__":
    accumWeight = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    calibrated = False
    while(True):
        #current frame captured
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]
        #get the region of interest
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("Loading")
            elif num_frames == 29:
                print("Success")
        #when frames >=30        
        else:
            hand = segment(gray)
            #check for hand
            if hand is not None:
                (thresholded, segmented) = hand
 
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # find the hand
                fingers = count(thresholded, segmented)
                	
               	cv2.putText(clone, str(''), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                cv2.imshow("Thesholded", thresholded)
                #check for the required pattern
                if fingers==4:
                    #send whatsapp message
                	whatsapp.mes()
                    #play the alarm
                	playsound(r"open.mp3")
					
        #draw the hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        cv2.imshow("Video Feed", clone)
        #stop the loop
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break
#release memory
camera.release() 
cv2.destroyAllWindows()
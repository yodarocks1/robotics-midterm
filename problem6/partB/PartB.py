 
# base from: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
# square and on image display: https://www.geeksforgeeks.org/facial-expression-recognizer-using-fer-using-deep-neural-net/
# documentation for fer: ' https://pypi.org/project/fer/
# fps from https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/

# also found at the bottom of the jupityr script.

# run ' pip install fer ' & ' pip install tensorflow ' if you run into import errors

# imports for fer
from fer import FER
import os
import sys
import pandas as pd

# import the opencv library
import cv2

# time for fps
import time

# clear prev
#cv2.destroyAllWindows()

# fer detector
detector = FER()


# define a video capture object
vid = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()



    if ret:
        try:
            result = detector.detect_emotions(frame)
            bounding_box = result[0]["box"]
            emotions = result[0]["emotions"]

            # adds box and emotions
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 155, 255), 2,)
            emotion_name, score = detector.top_emotion(frame)
            for index, (emotion_name, score) in enumerate(emotions.items()):
                color = (255, 0, 0) if score < 0.01 else (211, 211,211)
                emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))

                cv2.putText(frame,emotion_score,(bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)


        except:
            pass

        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)


    # Display the resulting frame
        cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

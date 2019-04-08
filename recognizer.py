from enum import Enum
import cv2
import numpy as np
import time
import guicontroller as gui
from keras.models import load_model

MODEL_PATH = "model/hand_model_gray.hdf5"
HAND_BOX = (20, 50, 250, 250)
NULL_POS = (170, 200)
CLASSES = {
    -1: 'NONE',
    0: 'FIST',
    1: 'HAND',
    2: 'POINTER',
    3: 'SCROLL'
}

POSITIONS = {
    'gesture_text': (15, 40), # hand pose text
    'fps': (15, 20), # fps counter
    'null_pos': NULL_POS # used as null point for mouse control
}

class GestureRecognizer:

    def __init__(self, capture=0, model_path=MODEL_PATH):
        self.bg = None
        self.frame = None # current frame
        self.tracker = cv2.TrackerKCF_create()
        self.kernel = np.ones((3, 3), np.uint8)
        self.recognizer = load_model(model_path)
        self.is_tracking = False
        self.hand_bbox = HAND_BOX
        self.best_prediction_index = -1

        # Begin capturing video
        self.video = cv2.VideoCapture(capture)
        if not self.video.isOpened():
            print("Could not open video")

    def __del__(self):
        cv2.destroyAllWindows()
        self.video.release()

    # Helper function for applying a mask to an array
    def mask_array(self, array, imask):
        if array.shape[:2] != imask.shape:
            raise Exception("Shapes of input and imask are incompatible")
        output = np.zeros_like(array, dtype=np.uint8)
        for i, row in enumerate(imask):
            output[i, row] = array[i, row]
        return output

    def extract_foreground(self):
        # Find the absolute difference between the background frame and current frame
        diff = cv2.absdiff(self.bg, self.frame)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold the mask
        th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        # Opening, closing and dilation
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        img_dilation = cv2.dilate(closing, self.kernel, iterations=1)

        # Get mask indexes
        imask = img_dilation > 0

        # Get foreground from mask
        foreground = self.mask_array(self.frame, imask)
        return foreground, mask, img_dilation

    def run(self):   
        while True:
            ok, self.frame = self.video.read()
            self.frame = cv2.resize(self.frame, (0,0), fx=0.6, fy=0.6)
            display = self.frame.copy()
            if not ok:
                break

            timer = cv2.getTickCount()

            if self.bg is None:
                cv2.putText(display,
                            "(R) Reset foreground extraction", 
                            POSITIONS['gesture_text'], 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.75, (0, 127, 64), 2)
                cv2.imshow("Display", display)

                k = cv2.waitKey(1) & 0xff
                if k == 27: break # ESC pressed
                elif k == 114 or k == 108: 
                    # r pressed
                    self.bg = self.frame.copy()
                    self.hand_bbox = HAND_BOX
                    self.is_tracking = False
            else:
                # Extract the foreground
                foreground, mask, dilatation = self.extract_foreground()
                hand_crop_display = dilatation[int(self.hand_bbox[1]):int(self.hand_bbox[1]+self.hand_bbox[3]), 
                                                int(self.hand_bbox[0]):int(self.hand_bbox[0]+self.hand_bbox[2])]
                    
                # Get hand from mask using the bounding box
                hand_crop = mask[int(self.hand_bbox[1]):int(self.hand_bbox[1]+self.hand_bbox[3]), 
                                 int(self.hand_bbox[0]):int(self.hand_bbox[0]+self.hand_bbox[2])]

                # Update tracker
                #if self.is_tracking:
                    #tracking, self.hand_bbox = self.tracker.update(foreground)

                try:
                    # Resize cropped hand and make prediction on gesture
                    hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
                    prediction = self.recognizer.predict(hand_crop_resized)                    
                    mean = cv2.mean(hand_crop_display)
                    if ((mean[0] + mean[1] + mean[2]) / 3) < 10:
                        self.best_prediction_index = -1
                    else:
                        self.best_prediction_index = prediction[0].argmax() # Get the index of the greatest confidence
                    gesture = CLASSES[self.best_prediction_index]

                    data_display = np.zeros((300, 500), dtype=np.uint8) # Black screen to display data
                    for i, pred in enumerate(prediction[0]):
                        # Draw confidence bar for each gesture
                        barx = POSITIONS['gesture_text'][0]
                        bary = 60 + i*60
                        bar_height = 20
                        bar_length = int(280 * pred) + barx # calculate length of confidence bar
                        
                        # Make the most confidence prediction green
                        color = (0, 0, 255)
                        if i == self.best_prediction_index:
                            color = (0, 255, 0)
                        
                        cv2.putText(data_display, "{}: {:.0%} ".format(CLASSES[i], pred), (POSITIONS['gesture_text'][0], 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                        cv2.rectangle(data_display, (barx, bary), (bar_length, bary - bar_height), color, cv2.FILLED)
                    
                    cv2.putText(display, 
                                "GESTURE: {}".format(gesture), 
                                POSITIONS['gesture_text'], 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.75, (0, 0, 255), 2)
                except Exception as ex:
                    cv2.putText(display, 
                                "GESTURE: error", 
                                POSITIONS['gesture_text'], 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.75, (0, 0, 255), 2)    
                
                # Draw bounding box
                p1 = (int(self.hand_bbox[0]), int(self.hand_bbox[1]))
                p2 = (int(self.hand_bbox[0] + self.hand_bbox[2]), int(self.hand_bbox[1] + self.hand_bbox[3]))
                cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

                # Calculate difference in hand position
                hand_pos = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
                mouse_change = ((p1[0] + p2[0])//2 - POSITIONS['null_pos'][0], POSITIONS['null_pos'][0] - (p1[1] + p2[1])//2)

                if self.is_tracking and pred > 0.5:
                    if self.best_prediction_index == -1:
                        print("NONE")
                    elif self.best_prediction_index == 0:
                        gui.take_screenshot()
                    elif self.best_prediction_index == 1:
                        gui.left_click()
                        time.sleep(2)
                    #elif self.best_prediction_index == 2:
                        #gui.mouse_move(10, 10) # TODO: implement move logics
                    elif self.best_prediction_index == 3:
                        gui.scroll()
                        time.sleep(1)

                # Draw hand moved difference
                # cv2.circle(display, POSITIONS['null_pos'], 5, (0,0,255), -1)
                # cv2.circle(display, hand_pos, 5, (0,255,0), -1)
                # cv2.line(display, POSITIONS['null_pos'], hand_pos, (255,0,0),5)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                # Display FPS on frame
                cv2.putText(display, "FPS : " + str(int(fps)), POSITIONS['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)

                cv2.imshow("Hand mask", hand_crop_display)
                cv2.imshow("Display", display)
                # Display result
                cv2.imshow("Results", data_display)

                k = cv2.waitKey(1) & 0xff
                if k == 27: break # ESC pressed
                elif k == 114 or k == 108: # r pressed
                    self.bg = self.frame.copy()
                    self.hand_bbox = HAND_BOX
                    self.is_tracking = False
                elif k == 116: # t pressed
                    # Initialize tracker with first frame and bounding box
                    self.tracker = cv2.TrackerKCF_create()
                    self.tracker.init(self.frame, self.hand_bbox)
                    self.is_tracking = True
                elif k == 112: # p pressed
                    # Reset to paused state
                    self.is_tracking = False
                    self.bg = None
                    cv2.destroyAllWindows()
                elif k != 255: print(k)
                
if __name__ == "__main__":
    recognizer = GestureRecognizer(0)
    recognizer.run()
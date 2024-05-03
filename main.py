from abc import abstractmethod
from typing import Type, List
import tensorflow as tf
import numpy as np
import cv2
# import tensorflow_hub as hub
import webbrowser as wb
import pyautogui

class Observable:
    def addObserver(self, obs: Type['Observer']):
        self.observers.append(obs)

    def notifyObservers(self):
        for observer in self.observers:
            observer.update(self)

class Observer:
    def __init__(self):
        pass

    @abstractmethod
    def update(self, obs: Observable):
        pass

class VideoStream(Observable):
    observers :List[Observer] = []
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path='model/singlepose-lightning.tflite')
        self.interpreter.allocate_tensors()
        # self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192
        self.vid = cv2.VideoCapture(0)

    def start(self):
        self.addObserver(Game())
        while self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.flip(frame,1)
            h, w  = frame.shape[:2]
            self.drawLines(frame, w, h)

            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
            input_image = tf.cast(img, dtype=tf.float32)

            # # Setup input and output 
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            # # Make predictions 
            self.interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            self.interpreter.invoke()
            keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])[0, 0]

            # Run model inference.
            # keypoints_with_scores = self.movenet(input_image)

            orig_w, orig_h = frame.shape[:2]
            M = self.get_affine_transform_to_fixed_sizes_with_padding((orig_w, orig_h), (192, 192))
            # M has shape 2x3 but we need square matrix when finding an inverse
            M = np.vstack((M, [0, 0, 1]))
            M_inv = np.linalg.inv(M)[:2]
            xy_keypoints = keypoints_with_scores[:, :2] * 192
            xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
            keypoints_with_scores = np.hstack((xy_keypoints, keypoints_with_scores[:, 2:]))

            self.handle_keypoints(frame, keypoints_with_scores, 0.4)

            cv2.imshow('MoveNet Lightning', frame)
    
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
        
        self.vid.release()
        cv2.destroyAllWindows()

    def get_affine_transform_to_fixed_sizes_with_padding(self,size, new_sizes):
        width, height = new_sizes
        scale = min(height / float(size[1]), width / float(size[0]))
        M = np.float32([[scale, 0, 0], [0, scale, 0]])
        M[0][2] = (width - scale * size[0]) / 2
        M[1][2] = (height - scale * size[1]) / 2
        return M

    def drawLines(self, frame, w, h):
        cv2.line(frame, (0,int(h/2 - 20)),(w,int(h/2 - 20)),(255,255,255),2) 
        cv2.line(frame, (0,int(h/2 + 100)),(w,int(h/2 + 100)),(255,255,255),2) 
        cv2.line(frame,(int(w/2 - 200),0),(int(w/2 - 200),h),(255,255,255),2)
        cv2.line(frame,(int(w/2 + 200),0),(int(w/2 + 200),h),(255,255,255),2)

    def movenet(self,input_image):
        """Runs detection on an input image.

        Args:
            input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
            A [1, 1, 17, 3] float numpy array representing the predicted keypoint
            coordinates and scores.
        """
        model = self.module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

    def handle_keypoints(self,frame, keypoints, confidence_threshold):
        h, w, c = frame.shape
        # shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))[5:7] #take points up to shoulder point
    
        sumx = 0
        sumy = 0
        for keypoint in keypoints[5:7]:
            y,x,conf = keypoint
            sumx += x
            sumy += y
            if conf > confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 10, (0,0,255),-1)
        avgx = sumx/2
        avgy = sumy/2

        cv2.circle(frame,(int(avgx),int(avgy)),20, (0,255,0),-1) # midpoint
        self.calculate_pos(w, h, avgx,avgy)
    
    def calculate_pos(self, width, height, x, y): 
        # cv2.line(frame, (0,int(h/2 - 20)),(w,int(h/2 - 20)),(255,255,255),2) jump
        # cv2.line(frame, (0,int(h/2 + 100)),(w,int(h/2 + 100)),(255,255,255),2) slide
        # cv2.line(frame,(int(w/2 - 100),0),(int(w/2 - 100),h),(255,255,255),2) left
        # cv2.line(frame,(int(w/2 + 100),0),(int(w/2 + 100),h),(255,255,255),2) right

        self.pos = ""
        self.state = ""
        if x >= (width/2 + 200):
            self.pos = "right"
        elif x <= (width/2 - 200):
            self.pos = "left"
        else:
            self.pos = "middle"

        if y <= (height/2 - 20):
            self.state = "jump"
        elif y >= (height/2 +100):
            self.state = "slide"
        else:
            self.state = "run"
        
        self.notifyObservers()

class Game(Observer):
    def __init__(self):
        super()
        self.currentPos = "middle"
        self.currentState = "run"
        wb.open("https://subwaysurfersgame.io/subway-surfers-game.embed", new=1)

    def update(self, obs:Observable):
        prevPos = self.currentPos
        if self.currentPos != obs.pos:
            self.currentPos = obs.pos
            self.updatePos(prevPos, self.currentPos)

        if self.currentState != obs.state:
            self.currentState = obs.state
            self.updateState(self.currentState)


    def updatePos(self, prevPos, newPos):
        match newPos:
            case "middle":
                if prevPos == "left":
                    pyautogui.press("right") 
                elif prevPos == "right":
                    pyautogui.press("left")
            case "left":
                pyautogui.press("left")
            case "right":
                pyautogui.press("right")

    def updateState(self, newState):
        match newState:
            case "jump":
                pyautogui.press("up")
            case "slide":
                pyautogui.press("down")
            case _:
                pass


if __name__ == "__main__":
    VideoStream().start()
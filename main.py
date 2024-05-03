"""Main class"""

from abc import abstractmethod
from typing import Type, List
import webbrowser as wb
import tensorflow as tf
import numpy as np
from cv2 import cv2
import pyautogui

class Observable:
    """Represents an Abstract Observable class"""

    def add_observer(self, obs: Type['Observer']):
        """
        Adds an observer to the list of observers
        """
        self.observers.append(obs)

    def notify_observers(self):
        """
        Updates each observer 
        """
        for observer in self.observers:
            observer.update(self)

class Observer:
    """Represents an Abstract Observer class"""
    def __init__(self):
        pass

    @abstractmethod
    def update(self, obs: Observable):
        """
        Updates the observer based on Observable's state
        """

class VideoStream(Observable):
    """Handles Webcam feed and pose detection."""
    observers :List[Observer] = []

    def __init__(self):
        """Constructor loads MoveNet model and starts webcam capture"""
        self.interpreter = tf.lite.Interpreter(model_path='model/singlepose-lightning.tflite')
        self.interpreter.allocate_tensors()
        # self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192
        self.vid = cv2.VideoCapture(0)
        self.pos = ""
        self.state = ""

    def start(self):
        """Starts the pose detection"""
        self.add_observer(Game())
        while self.vid.isOpened():
            _, frame = self.vid.read()
            frame = cv2.flip(frame,1)
            h, w  = frame.shape[:2]
            self.draw_lines(frame, w, h)

            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
            input_image = tf.cast(img, dtype=tf.float32)

            # Setup input and output
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            # Make predictions
            self.interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            self.interpreter.invoke()
            keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])[0, 0]

            orig_w, orig_h = frame.shape[:2]
            mat = self.get_affine_transform((orig_w, orig_h), (192, 192))
            # M has shape 2x3 but we need square matrix when finding an inverse
            mat = np.vstack((mat, [0, 0, 1]))
            m_inv = np.linalg.inv(mat)[:2]
            xy_keypoints = keypoints_with_scores[:, :2] * 192
            xy_keypoints = cv2.transform(np.array([xy_keypoints]), m_inv)[0]
            keypoints_with_scores = np.hstack((xy_keypoints, keypoints_with_scores[:, 2:]))

            self.handle_keypoints(frame, keypoints_with_scores, 0.4)

            cv2.imshow('MoveNet Lightning', frame)

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break

        self.vid.release()
        cv2.destroyAllWindows()

    def get_affine_transform(self,size, new_sizes):
        """
        Convert model output back to actual coordinates
        for accurate keypoint drawing.
        """
        width, height = new_sizes
        scale = min(height / float(size[1]), width / float(size[0]))
        mat = np.float32([[scale, 0, 0], [0, scale, 0]])
        mat[0][2] = (width - scale * size[0]) / 2
        mat[1][2] = (height - scale * size[1]) / 2
        return mat

    def draw_lines(self, frame, w, h):
        """
        Draw border lines on webcam output
        """
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
        """
        Draw each keypoint and calculate the position
        """
        h, w, _ = frame.shape

        sumx = 0
        sumy = 0
        for keypoint in keypoints[5:7]:
            y,x,conf = keypoint
            sumx += x
            sumy += y
            if conf > confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 10, (0,0,255),-1)
        x_mid = sumx/2
        y_mid = sumy/2

        cv2.circle(frame,(int(x_mid),int(y_mid)),20, (0,255,0),-1) # midpoint
        self.calculate_pos(w, h, x_mid,y_mid)

    def calculate_pos(self, width, height, x, y):
        """
        Determines player's new position and state given current x,y position.
        """
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

        self.notify_observers()

class Game(Observer):
    """Represents the Game itself"""

    def __init__(self):
        """
        Constructor opens the game in a new default browser window
        """
        super()
        self.current_pos = "middle"
        self.current_state = "run"
        wb.open("https://subwaysurfersgame.io/subway-surfers-game.embed", new=1)

    def update(self, obs:Observable):
        """
        Updates player's position and state
        """
        prev_pos = self.current_pos
        if self.current_pos != obs.pos:
            self.current_pos = obs.pos
            self.update_pos(prev_pos, self.current_pos)

        if self.current_state != obs.state:
            self.current_state = obs.state
            self.update_state(self.current_state)


    def update_pos(self, prev_pos, new_pos):
        """
        Updates player's position 
        """
        match new_pos:
            case "middle":
                if prev_pos == "left":
                    pyautogui.press("right")
                elif prev_pos == "right":
                    pyautogui.press("left")
            case "left":
                pyautogui.press("left")
            case "right":
                pyautogui.press("right")

    def update_state(self, new_state):
        """
        Updates player's state 
        """
        match new_state:
            case "jump":
                pyautogui.press("up")
            case "slide":
                pyautogui.press("down")
            case _:
                pass


if __name__ == "__main__":
    VideoStream().start()

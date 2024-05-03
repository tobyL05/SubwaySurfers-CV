## Play Subway Surfers in real life!

Using TensorFlow's MoveNet Pose detection model, get active while playing Subway Surfers.

### Requirements
- A webcam
- Fairly big open space
- Single Player only
- Default browser with hardware acceeleration enabled

### Usage
- Clone the repository to your local machine.
- Install the dependencies
- Launch main.py 

### How To Play
- The game will pop up on a new browser window and the webcam feed will pop up in a new window
- On the webcam feed, there are lines, red dots, and a green dot. The code reads the position of the green dot each frame.
- The area in which the green dot is located determines the position (left, middle, right) and state (jump, run, slide) of the player.
- You will need a mouse to pause, press "play", close the game, etc.

### Demo

### Future
- Bring this to the web?
- Improve robustness

import pydirectinput as py
import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
import pickle


# Emulate keyboard and mouse function
def emulator_request_decoder(predict, binds):
    request = binds[predict]
    for req_index in range(len(request)):
        device, command, button = request[req_index]
        if device == 'kb':
            try:
                if command == 'keyUp':
                    py.keyUp(button)
                elif command == 'keyDown':
                    py.keyDown(button)
                elif command == 'press':
                    py.press(button)
                else:
                    print(f'Unknown Command Error: {command} is not available for {device}')
            except TypeError:
                print(f'Unknown Button Error: {button}')
        elif device == 'ms':
            try:
                if command == 'click':
                    py.click(button=button)
                elif command == 'mouseUp':
                    py.mouseUp(0, 0, button)
                elif command == 'mouseDown':
                    py.mouseDown(0, 0, button)
                elif command == 'move':
                    x, y = button.split('_')
                    py.moveRel(int(x), int(y))
                else:
                    print(f'Unknown Command Error: {command} is not available for {device}')
            except TypeError:
                print(f'Unknown Button Error: {button}')
        else:
            print(f'Unknown Device Error: {device}. Only "kb", "ms" devices are available')
        # print(predict)


# Mediapipe import
mp_holistic = mp.solutions.holistic

# Load model
with open('gb_PaB_body1.pkl', 'rb') as f:
    model = pickle.load(f)

# Contains what we want to emulate for each pose
bind_settings = {
    'stay':       [['ms', 'mouseUp', 'right']],
    'left_hook':  [['ms', 'mouseUp', 'right'], ['ms', 'mouseDown', 'left'], ['ms', 'mouseUp', 'left']],
    'right_hook': [['ms', 'mouseUp', 'right'], ['ms', 'mouseDown', 'left'], ['ms', 'mouseUp', 'left']],
    'block':      [['ms', 'mouseDown', 'right']],
}

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        try:
            # Get landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Predict pose
            X = pd.DataFrame([pose_row])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            # Emulate prediction
            emulator_request_decoder(pred, bind_settings)
        except AttributeError:
            print('Detection error')
            pass

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()

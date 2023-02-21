import pydirectinput as py
import cv2
import pandas as pd
import mediapipe as mp
import pickle


# Helper function to classify the right and left hands according to their position
def which_hand_is(prediction_results):
    fst_hand = prediction_results.multi_hand_landmarks[0].landmark[0]
    sec_hand = prediction_results.multi_hand_landmarks[1].landmark[0]
    fst_hand_coord = (int(fst_hand.x * 640), int(fst_hand.y * 480))
    sec_hand_coord = (int(sec_hand.x * 640), int(sec_hand.y * 480))

    if fst_hand.x > sec_hand.x:
        labels = ['Right', 'Left']
    else:
        labels = ['Right', 'Left']

    coords = [fst_hand_coord, sec_hand_coord]
    return labels, coords


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


# Mediapipe setup
mp_hands = mp.solutions.hands

# Load models for each hand
with open('left_gb_mario.pkl', 'rb') as f:
    model_left = pickle.load(f)
with open('right_gb_mario.pkl', 'rb') as f:
    model_right = pickle.load(f)

hand_models = {'Left': model_left,
               'Right': model_right}

# Initializing gestures and linking them to hands
list_of_classes = ['l_stay', 'forward', 'back', 'r_stay', 'jump']
hand_for_class = ['Left', 'Left', 'Left', 'Right', 'Right']

# Contains what we want to emulate for each pose
bind_settings = {
    'l_stay':    [['kb', 'keyUp', 'a'], ['kb', 'keyUp', 'd']],
    'forward':   [['kb', 'keyUp', 'a'], ['kb', 'keyDown', 'd']],
    'back':      [['kb', 'keyUp', 'd'], ['kb', 'keyDown', 'a']],
    'r_stay':    [['kb', 'keyUp', 'space']],
    'jump':      [['kb', 'keyDown', 'space']]
}

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        try:
            for acc_hand in list(set(hand_for_class)):
                model = hand_models[acc_hand]
                hands_labels, hands_coords = which_hand_is(results)
                hand_landmarks = results.multi_hand_landmarks[hands_labels.index(acc_hand)].landmark

                row = []
                for landmark in hand_landmarks:
                    row += [landmark.x, landmark.y, landmark.z]
                X = pd.DataFrame([row])

                pred = model.predict(X)[0]

                emulator_request_decoder(pred, bind_settings)
        except IndexError:
            print('Detection Error')

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()

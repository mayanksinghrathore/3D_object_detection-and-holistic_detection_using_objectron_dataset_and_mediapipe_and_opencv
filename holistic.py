import cv2
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        success, image = cap.read()

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        #print(results.pose_landmarks)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


        end=time.time()
        totalTime=end-start

        try:
            fps = 1/totalTime
        except ZeroDivisionError:
            print("FPS",fps)

        cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('MediaPipe FaceMesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()



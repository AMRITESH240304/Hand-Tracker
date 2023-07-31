import cv2 as cv
import time
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    previous_time = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error reading frame")
            break

        imRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(imRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    # checking out height with and  channel
                    h, w, c = img.shape
                    # finding position of centre
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    if id == 8:
                        cv.circle(img, (cx, cy), 15, (0, 0, 255), cv.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # create a mask image from the landmark drawing
        mask = np.zeros_like(img)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    # checking out height with and  channel
                    h, w, c = img.shape
                    # finding position of centre
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    if id == 8:
                        cv.circle(mask, (cx, cy), 15, (0, 0, 255), cv.FILLED)
                mpDraw.draw_landmarks(mask, handLms, mpHands.HAND_CONNECTIONS)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv.putText(mask, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        bitwise_and = cv.bitwise_and(img, mask)

        ret, buffer = cv.imencode('.jpg', bitwise_and)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)

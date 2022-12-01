import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_detection_confidence)

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        landmark_list = []
        if results.multi_hand_landmarks:
            height, width, channels = img.shape
            for hand_landmarks in results.multi_hand_landmarks:
                for ix, landmark in enumerate(hand_landmarks.landmark):
                    center_x, center_y = int(landmark.x*width), int(landmark.y*height)
                    # landmark_list.append([ix, center_x, center_y])
                    landmark_list.append([center_x, center_y])
                    if draw:
                        cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        return landmark_list

    def which_fingers_up(self, img, lm=None):
        finger_list = [0, 0, 0, 0, 0]
        if lm is None:
            lm = self.find_position(img, draw=False)
        if lm:
            if math.hypot((lm[4][0] - lm[17][0]), (lm[4][1] - lm[17][1])) \
                    > math.hypot((lm[3][0] - lm[17][0]), (lm[3][1] - lm[17][1])):
                finger_list[0] = 1
            finger_tips = [8, 12, 16, 20]
            for finger_tip in finger_tips:
                if (math.hypot((lm[finger_tip][0] - lm[0][0]), (lm[finger_tip][1] - lm[0][1]))
                        > math.hypot((lm[finger_tip - 2][0] - lm[0][0]), (lm[finger_tip - 2][1] - lm[0][1]))):
                    finger_list[finger_tip//4 - 1] = 1

        return finger_list


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    c_time = 0
    detector = HandDetector()

    while True:
        success, frame = cap.read()
        frame = detector.find_hands(frame)
        landmark_list = detector.find_position(frame)
        if len(landmark_list) != 0:
            print(landmark_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import Hand_Tracking_Module
import math


cap = cv2.VideoCapture(0)
cam_width, cam_height = 1080, 1080
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = Hand_Tracking_Module.HandDetector(min_detection_confidence=0.7)

while True:
    success, frame = cap.read()
    # frame = detector.find_hands(frame)
    num_fingers = 0
    # landmark list
    lm = detector.find_position(frame, draw=False)
    if lm:
        if math.hypot((lm[4][0] - lm[17][0]), (lm[4][1] - lm[17][1])) \
                > math.hypot((lm[3][0] - lm[17][0]), (lm[3][1] - lm[17][1])):
            num_fingers += 1
        finger_tips = [8, 12, 16, 20]
        for finger_tip in finger_tips:
            if (math.hypot((lm[finger_tip][0] - lm[0][0]), (lm[finger_tip][1] - lm[0][1]))
                    > math.hypot((lm[finger_tip - 2][0] - lm[0][0]), (lm[finger_tip - 2][1] - lm[0][1]))):
                num_fingers += 1
        # line_length = math.hypot((x2 - x1), (y2 - y1))

    cv2.putText(frame, str(int(num_fingers)), (40, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

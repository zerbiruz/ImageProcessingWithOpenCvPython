import cv2

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()

    if not ret:
        break
    else:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "OpenCV", (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
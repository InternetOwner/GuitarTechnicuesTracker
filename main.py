import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(
    "C:/Users/Administrator/Downloads/Telegram Desktop/Dave Mustaine - Spider Chord Hand Changes demonstration.flv [3zCKVTE3t6o].mp4")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 800, 600)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=4, min_tracking_confidence=0.4, min_detection_confidence=0.4)
mpDraw = mp.solutions.drawing_utils

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_filename = 'output.mp4'
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

count = 0
seconds = 0
while True:
    count += 1
    if seconds != count // fps:
        seconds = count // fps
        print(f"{seconds} second")
    success, img = cap.read()
    if not success:
        break

    # Обработка кадра
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                                  connection_drawing_spec=mpDraw.DrawingSpec(1, 2, 1),
                                  landmark_drawing_spec=mpDraw.DrawingSpec(color=mpDraw.RED_COLOR, thickness=1,
                                                                           circle_radius=1))
    out.write(img) # Запись обработанного кадра в выходной файл
    cv2.imshow("Image", img)
    cv2.waitKey(1)


cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Видео сохранено в файл: {output_filename}")

import mediapipe as mp
import cv2
import xml.etree.ElementTree as ET
import os


class HandTracking:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=4, min_tracking_confidence=0.5, min_detection_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def real_time_hands_detection(self, file_path: str | int = 0):
        cap = cv2.VideoCapture(file_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)
        while True:
            success, img = cap.read()
            if not success:
                break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               connection_drawing_spec=self.mpDraw.DrawingSpec(
                                                   color=self.mpDraw.BLACK_COLOR, thickness=2,
                                                   circle_radius=1),
                                               landmark_drawing_spec=self.mpDraw.DrawingSpec(
                                                   color=self.mpDraw.RED_COLOR, thickness=1,
                                                   circle_radius=1))
            cv2.imshow("Image", img)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()

    def hands_markup_to_xml(self, file_path: str, output_xml_filename: str):
        cap = cv2.VideoCapture(file_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

        count = 0
        frames = ET.Element("frames")
        while True:
            count += 1
            frame = ET.SubElement(frames, "frame", name=f"frame_{str(count)}")
            success, img = cap.read()
            if not success:
                break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for i, handLms in enumerate(results.multi_hand_landmarks):
                    hand = ET.SubElement(frame, "hand", name=f"hand_{str(i + 1)}", type="hand")
                    for j, landmark in enumerate(handLms.landmark):
                        landmark = ET.SubElement(hand, "landmark", name=f"landmark_{j + 1}", x=f"{landmark.x: .4f}",
                                                 y=f"{landmark.y: .4f}", z=f"{landmark.z: .4f}")
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               connection_drawing_spec=self.mpDraw.DrawingSpec(
                                                   color=self.mpDraw.BLACK_COLOR, thickness=2,
                                                   circle_radius=1),
                                               landmark_drawing_spec=self.mpDraw.DrawingSpec(
                                                   color=self.mpDraw.RED_COLOR, thickness=1,
                                                   circle_radius=1))
            out.write(img)
            cv2.waitKey(1)
        data = ET.tostring(frames)
        with open(output_xml_filename, "wb+") as videodata:
            videodata.write(data)
        cap.release()
        out.release()

    def make_dataset(self, dir_path: str):
        dataset_dirs = list(filter(lambda x: "." not in x, os.listdir(dir_path)))
        marked_dataset = f"marked_{dir_path}"
        os.mkdir(marked_dataset)
        for directory in dataset_dirs:
            os.mkdir(f"{marked_dataset}/{directory}")
            if os.listdir(f"{dir_path}/{directory}"):
                for video in os.listdir(f"{dir_path}/{directory}"):
                    self.hands_markup_to_xml(f"{dir_path}/{directory}/{video}", f"{marked_dataset}/{directory}/{video[:-4]}.xml")
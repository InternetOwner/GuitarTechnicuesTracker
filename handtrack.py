import mediapipe as mp
import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np


def read_dataset(dir_path: str) -> tuple[np.ndarray, list, list]:
    dataset_dirs = list(filter(lambda x: os.path.isdir(os.path.join(dir_path, x)), os.listdir(dir_path)))
    markdown = []
    labels = []
    for i, directory in enumerate(dataset_dirs):
        labels += [i] * len(os.listdir(os.path.join(dir_path, directory)))
        for xml_file in os.listdir(os.path.join(dir_path, directory)):
            doc = ET.parse(os.path.join(dir_path, directory, xml_file))
            root = doc.getroot()
            tmp_vid = []
            for frame in root:
                tmp_frame = []
                for hand in frame.findall("hand"):
                    tmp_hand = []
                    for landmark in hand.findall("landmark"):
                        tmp_hand.append(
                            [float(landmark.get("x")), float(landmark.get("y")), float(landmark.get("z"))])
                    tmp_hand = np.array(tmp_hand)
                    a, b = tmp_hand[0], tmp_hand[9]
                    linalg = np.linalg.norm(b - a)
                    tmp_frame.append(tmp_hand / linalg)
                tmp_vid.append(tmp_frame)
            markdown.append(tmp_vid[1:-1])
    markdown = np.array(markdown)
    return markdown, labels, dataset_dirs


class HandTracking:
    def __init__(self, max_num_hands: int, min_tracking_confidence: float, min_detection_confidence: float):
        self.__mpHands = mp.solutions.hands
        self.__max_num_hands = max_num_hands
        self.__min_tracking_confidence = min_tracking_confidence
        self.__min_detection_confidence = min_detection_confidence
        self.__hands_init()
        self.mpDraw = mp.solutions.drawing_utils

    def __hands_init(self):
        self.hands = self.__mpHands.Hands(max_num_hands=self.__max_num_hands,
                                          min_tracking_confidence=self.__min_tracking_confidence,
                                          min_detection_confidence=self.__min_detection_confidence)

    def change_min_tracking_confidence(self, new_value: int):
        self.__min_tracking_confidence = new_value
        self.__hands_init()

    def change_min_detection_confidence(self, new_value: int):
        self.__min_detection_confidence = new_value
        self.__hands_init()

    def change_max_num_hands(self, new_value: int):
        self.__max_num_hands = new_value
        self.__hands_init()

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
                    self.mpDraw.draw_landmarks(img, handLms, self.__mpHands.HAND_CONNECTIONS,
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
                for j, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                    landmark = ET.SubElement(frame, "landmark", name=f"landmark_{j + 1}", x=f"{landmark.x: .4f}",
                                                 y=f"{landmark.y: .4f}", z=f"{landmark.z: .4f}")
                    self.mpDraw.draw_landmarks(img, results.multi_hand_landmarks[0], self.__mpHands.HAND_CONNECTIONS,
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
        dataset_dirs = list(filter(lambda x: os.path.isdir(os.path.join(dir_path, x)), os.listdir(dir_path)))
        marked_dataset = f"marked_{dir_path}"
        os.mkdir(marked_dataset)
        for directory in dataset_dirs:
            os.mkdir(os.path.join(marked_dataset, directory))
            if os.listdir(os.path.join(dir_path, directory)):
                for video in os.listdir(os.path.join(dir_path, directory)):
                    self.hands_markup_to_xml(os.path.join(dir_path, directory, video),
                                             os.path.join(marked_dataset, directory, f"{video[:-4]}.xml"))

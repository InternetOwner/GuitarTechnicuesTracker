from handtrack import HandTracking, read_dataset
from classifier import TechniqueClassifier, SequenceDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import customtkinter as ctk
from PIL import Image
from collections import deque
import numpy as np


class ModernGUI(ctk.CTk):
    def __init__(self, classes_names: list):
        super().__init__()
        self.stream = None
        self.buffer = deque(maxlen=30)
        self.classes_names = classes_names

        INPUT_DIM = 63
        HIDDEN_DIM = 128
        OUTPUT_DIM = len(classes_names)
        NUM_LAYERS = 2

        self.model = TechniqueClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
        weights = torch.load("model/model.pth", weights_only=False)
        self.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.model.device)
        self.model.load_state_dict(weights)


        self.video_label = ctk.CTkLabel(self, text="Ожидание видео...")
        self.video_label.pack(pady=10)

        self.title("Guitar handtrack")
        self.geometry("800x700")
        self.entry_filename = ctk.CTkEntry(self, placeholder_text="Введите имя файла для записи...", width=300)
        self.entry_filename.pack(pady=10)

        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=5)

        self.btn_record = ctk.CTkButton(self.btn_frame, text="Начать", command=self.on_btn_click)
        self.btn_record.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(self.btn_frame, text="Остановить", fg_color="gray", command=self.on_stop_click)
        self.btn_stop.pack(side="left", padx=10)

    def __get_video_stream__(self, file_path: str | int = 0):
        self.stream = ht.real_time_hands_detection(file_path=file_path)
        self.update_video()

    def on_btn_click(self):
        entry = self.entry_filename.get()
        filename = 0 if not entry else entry
        self.__get_video_stream__(filename)

    def on_stop_click(self):
        print("Кнопка 'Остановить' нажата.")

    def neuralnet(self):
        if len(self.buffer) < 30:
            return

        data = list(self.buffer)

        max_hands = max(len(frame) for frame in data)

        if max_hands == 0:
            self.buffer.clear()
            return

        hands_to_process = []

        for hand_idx in range(max_hands):
            hand_sequence = []
            for frame in data:
                if len(frame) > hand_idx:
                    hand_sequence.append(frame[hand_idx])
                else:
                    hand_sequence.append([0.0] * 63)
            hands_to_process.append(hand_sequence)

        processed_data = []

        for hand_frames in hands_to_process:
            sequence = []

            for frame in hand_frames:
                tmp_frame = np.array(frame)

                a = tmp_frame[0:3]
                b = tmp_frame[27:30]
                distance = np.linalg.norm(b - a)

                tmp_frame = tmp_frame / distance if distance > 0 else tmp_frame
                sequence.append(tmp_frame.flatten())

            processed_data.append(sequence)

        dummy_labels = [0] * len(processed_data)

        video_data = SequenceDataset(processed_data, dummy_labels)
        video_dataloader = DataLoader(video_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

        self.model.eval_model(video_dataloader, self.classes_names)
        self.buffer.clear()

    def update_video(self):
        try:
            rgb_frame, hands = next(self.stream)
            self.buffer.append(hands)

            if len(hands) > 0:
                self.buffer.append(hands)
            else:
                self.buffer.clear()

            if len(self.buffer) == 30:
                self.neuralnet()

            img = Image.fromarray(rgb_frame)
            imgtk = ctk.CTkImage(light_image=img, size=(640, 480))

            self.video_label.configure(text="", image=imgtk)
            self.video_label.image = imgtk

            self.after(5, self.update_video)

        except StopIteration:
            self.video_label.configure(text="Ожидание видео...", image="")


if __name__ == "__main__":
    ht = HandTracking(1, 0.5, 0.5)
    print(
        "What do you want to do?\n 1. Real-time hand tracking\n 2. Make xml dataset\n 3. Read xml dataset\n 4. Neural network\n 5. Load model")
    match input():
        case "1":
            classes_names = read_dataset("marked_dataset")[-1]
            print(classes_names)
            ht = HandTracking(2, 0.5, 0.5)
            app = ModernGUI(classes_names)
            app.mainloop()
        case "2":
            ht.make_dataset(input())
        case "3":
            print(read_dataset("marked_" + input())[0])
        case "4":
            dataset, labels, classes = read_dataset("marked_dataset")
            # test_dataset, test_lables, test_classes = read_dataset("marked_test")

            dataset = SequenceDataset(dataset, labels)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

            # test_dataset = SequenceDataset(test_dataset, test_lables)
            # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

            INPUT_DIM = 63  # 21 * 3 - mediapipe landmarks
            HIDDEN_DIM = 128
            OUTPUT_DIM = len(classes)
            NUM_LAYERS = 2

            model = TechniqueClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
            model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(model.device)
            model.train_model(dataloader, epochs=100)
            # model.eval_model(test_dataloader)
            model.save("model/model.pth")
        case "5":
            test_dataset, test_lables, test_classes = read_dataset("marked_test")
            test_dataset = SequenceDataset(test_dataset, test_lables)
            test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

            INPUT_DIM = 63
            HIDDEN_DIM = 128
            OUTPUT_DIM = len(test_classes)
            NUM_LAYERS = 2

            model = TechniqueClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
            weights = torch.load("model/model.pth", weights_only=False)
            model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(model.device)
            model.load_state_dict(weights)
            model.eval_model(test_dataloader, test_classes)

from handtrack import HandTracking, read_dataset
from classifier import TechniqueClassifier, SequenceDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import customtkinter as ctk
from PIL import Image
from collections import deque
import numpy as np


class ModernGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.stream = None
        self.buffer = deque(maxlen=30)

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


    def __get_video_stream__(self, file_path: str):
        self.stream = ht.real_time_hands_detection(file_path=file_path)
        self.update_video()

    def on_btn_click(self):
        self.__get_video_stream__(self.entry_filename.get())

    def on_stop_click(self):
        print("Кнопка 'Остановить' нажата.")


    def update_video(self):
        try:
            rgb_frame = next(self.stream)

            img = Image.fromarray(rgb_frame)
            imgtk = ctk.CTkImage(light_image=img, size=(640, 480))

            self.video_label.configure(text="", image=imgtk)
            self.video_label.image = imgtk

            # задержка кадра
            self.after(5, self.update_video)

        except StopIteration:
            self.video_label.configure(text="Ожидание видео...", image="")



if __name__ == "__main__":
    ht = HandTracking(1, 0.5, 0.5)
    print(
        "What do you want to do?\n 1. Real-time hand tracking\n 2. Make xml dataset\n 3. Read xml dataset\n 4. Neural network\n 5. Load model")
    match input():
        case "1":
            ht = HandTracking(2, 0.5, 0.5)
            app = ModernGUI()
            app.mainloop()
        case "2":
            ht.make_dataset(input())
        case "3":
            print(read_dataset("marked_" + input())[0])
        case "4":
            dataset, labels, classes = read_dataset("marked_dataset")
            test_dataset, test_lables, test_classes = read_dataset("marked_test")

            dataset = SequenceDataset(dataset, labels)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

            test_dataset = SequenceDataset(test_dataset, test_lables)
            test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

            INPUT_DIM = 63  # 21 * 3 - mediapipe landmarks
            HIDDEN_DIM = 128
            OUTPUT_DIM = len(classes)
            NUM_LAYERS = 2

            model = TechniqueClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
            model.train_model(dataloader, epochs=100)
            model.eval_model(test_dataloader)
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
            model.load_state_dict(weights)
            model.eval_model(test_dataloader)

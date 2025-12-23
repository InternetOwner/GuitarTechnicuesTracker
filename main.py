from handtrack import HandTracking, read_dataset
from classifier import TechniqueClassifier, SequenceDataset, collate_fn
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    ht = HandTracking(1, 0.5, 0.5)
    print(
        "What do you want to do?\n 1. Real-time hand tracking\n 2. Make xml dataset\n 3. Read xml dataset\n 4. Neural network\n 5. Load model")
    match input():
        case "1":
            ht.real_time_hands_detection(input())
        case "2":
            ht.make_dataset(input())
        case "3":
            print(read_dataset("marked_" + input())[0])
        case "4":
            dataset, labels, classes = read_dataset("marked_dataset")
            test_dataset, _, __ = read_dataset("marked_test")

            dataset = SequenceDataset(dataset, labels)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

            test_dataset = SequenceDataset(test_dataset, _)
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
            test_dataset, _, __ = read_dataset("marked_test")
            test_dataset = SequenceDataset(test_dataset, _)
            test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

            INPUT_DIM = 63
            HIDDEN_DIM = 128
            OUTPUT_DIM = len(__)
            NUM_LAYERS = 2

            model = TechniqueClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
            weights = torch.load("model/model.pth", weights_only=False)
            model.load_state_dict(weights)
            model.eval_model(test_dataloader)

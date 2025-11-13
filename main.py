from handtrack import HandTracking, read_dataset
from classifier import TechniqueClassifier

if __name__ == "__main__":
    ht = HandTracking(1, 0.5, 0.5)
    print("What do you want to do?\n 1. Real-time hand tracking\n 2. Make xml dataset\n 3. Read xml dataset\n 4. Neural network")
    match input():
        case "1":
            ht.real_time_hands_detection(input())
        case "2":
            ht.make_dataset("dataset1")
        case "3":
            print(read_dataset("marked_dataset1")[0])
        case "4":
            dataset, labels, classes = read_dataset("marked_dataset1")
            ltsmc = TechniqueClassifier()

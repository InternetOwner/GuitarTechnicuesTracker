from handtrack import HandTracking, read_dataset

if __name__ == "__main__":
    ht = HandTracking()
    print("What do you want to do?\n 1. Real-time hand tracking\n 2. Make xml dataset\n 3. Read xml dataset")
    match input():
        case "1":
            ht.real_time_hands_detection(input())
        case "2":
            ht.make_dataset("dataset1")
        case "3":
            print(read_dataset("marked_dataset1")[0])

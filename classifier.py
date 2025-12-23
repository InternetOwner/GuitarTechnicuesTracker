import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s).float() for s in sequences]
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]

    def __len__(self):
        return len(self.sequences)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences, torch.stack(labels), lengths


class TechniqueClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(TechniqueClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Полносвязный слой для классификации

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, lengths):
        # x (batch_size, seq_length, input_dim)
        # out (batch_size, seq_length, hidden_dim * num_directions)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(packed_input)  # hn, cn - конечные скрытое и ячеечное состояния
        out = self.fc(hn[-1])  # Полносвязный слой
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def train_model(self, dataloader: DataLoader, epochs: int = 10, lr: float = 0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr)
        self.to(self.device)
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels, lengths in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs, lengths)  # Прямой проход (Forward pass)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}')

    def eval_model(self, dataloader: DataLoader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, lengths in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs, lengths)

                _, predicted = torch.max(outputs.data, 1)
                print(labels.tolist(), predicted.tolist())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')


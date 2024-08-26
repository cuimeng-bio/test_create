import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import os

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ConvResNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ConvResNet, self).__init__()
        self.one_hot_dict = {
            'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }

        self.layer1 = ResidualBlock(input_channels, 64, kernel_size=9, padding=4)
        self.layer2 = ResidualBlock(64, 128, kernel_size=7, padding=3)
        self.layer3 = ResidualBlock(128, 256, kernel_size=3, padding=1)
        self.layer4 = ResidualBlock(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        x = self.pool(self.layer4(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def one_hot_pad(self, sequence, len_w):
        original_sequence = sequence
        padding_length = max(0, len_w - len(original_sequence))
        padded_sequence = original_sequence + 'X' * padding_length
        coded_seq = self.AA_ONE_HOT(padded_sequence)
        return coded_seq

    def AA_ONE_HOT(self, AA):
        AA = ''.join(['X' if aa not in self.one_hot_dict else aa for aa in AA])
        coding_arr = np.zeros((len(AA), 20), dtype=np.float32)
        for i in range(len(AA)):
            coding_arr[i] = self.one_hot_dict[AA[i]]
        return coding_arr

class SequenceDataset(Dataset):
    def __init__(self, fasta_file, len_w):
        self.sequences = []
        self.labels = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            self.sequences.append(str(record.seq))  # 转换为字符串
            ll = record.description.strip().split("_")
            self.labels.append(int(ll[-1]))
        if len(self.labels) != len(self.sequences):
            raise ValueError("The number of labels and sequences are different.")
        self.len_w = len_w

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx])

def train_model(model, train_loader, n_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for sequences, labels in train_loader:
            sequences = [model.one_hot_pad(seq, model.len_w) for seq in sequences]
            sequences = torch.tensor(sequences).to(device).permute(0, 2, 1)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

    print("Training complete. Best Accuracy: {:.2f}%".format(best_accuracy))

def predict_sequence(model, sequence, len_w, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        sequence_padded = model.one_hot_pad(sequence, len_w)
        sequence_tensor = torch.tensor(sequence_padded.T, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(sequence_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    ref_data1 = "F:\\project\\噬菌体\\database\\phrog\\structual\\train.fasta"
    
    len_w = 1500
    batch_size = 1024
    n_epochs = 100

    dataset = SequenceDataset(fasta_file=ref_data1, len_w=len_w)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_classes = len(np.unique(dataset.labels))
    print(n_classes)

    model = ConvResNet(input_channels=20, num_classes=n_classes)
    model.len_w = len_w

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, n_epochs, device)

    torch.save(model.state_dict(), ref_data1 + '.res_conv.pth')
    print("The model has been saved.")

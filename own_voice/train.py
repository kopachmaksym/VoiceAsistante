# train.py

import os
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
from sklearn.model_selection import train_test_split

# 1. Налаштування

# Базова директорія, де зберігаються ваші дані
base_dir = 'MEGA'  # Замініть на ваш фактичний шлях, якщо інший

# 2. Підготовка даних

def load_data(base_dir):
    data = []
    for root, dirs, files in os.walk(base_dir):
        if 'txt.done.data' in files:
            output_file_path = os.path.join(root, 'txt.done.data')
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Розділяємо рядок на шлях та транскрипцію
                    parts = line.split('  ', 1)  # Розділяємо по двох пробілах
                    if len(parts) != 2:
                        continue  # Пропускаємо некоректні рядки
                    audio_path, transcript = parts
                    # Видаляємо перший символ '/' з шляху
                    if audio_path.startswith('/'):
                        audio_path = audio_path[1:]
                    # Формуємо повний шлях до аудіофайлу
                    full_audio_path = os.path.normpath(audio_path)
                    # Перевіряємо, чи файл існує
                    if os.path.isfile(full_audio_path):
                        data.append({'audio_path': full_audio_path, 'transcript': transcript})
                    else:
                        print(f"Файл не знайдено: {full_audio_path}")
    return pd.DataFrame(data)

# Завантажуємо дані
annotations = load_data(base_dir)

# Перевірка, чи дані завантажені
if annotations.empty:
    print("Дані не завантажені. Перевірте шлях до даних та структуру файлів.")
    exit()

# Збереження готового CSV файлу
annotations.to_csv('annotations.csv', index=False)

# Розподіл даних
train_annotations, val_annotations = train_test_split(annotations, test_size=0.1)

# Збереження розділених даних
train_annotations.to_csv('train_annotations.csv', index=False)
val_annotations.to_csv('val_annotations.csv', index=False)

# 3. Клас Dataset

class SpeechDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_path = self.annotations.iloc[idx]['audio_path']
        transcript = self.annotations.iloc[idx]['transcript']
        # Завантажуємо аудіо
        waveform, sample_rate = torchaudio.load(audio_path)
        # Якщо необхідно, перетворюємо частоту дискретизації
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, transcript

# 4. Трансформації аудіо

# Налаштовуємо параметри MFCC
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=40,
    melkwargs={
        'n_fft': 512,
        'hop_length': 160,
        'n_mels': 128,
        'f_max': 8000,
    }
)

# 5. Створення Dataset та DataLoader

train_dataset = SpeechDataset(annotations_file='train_annotations.csv', transform=mfcc_transform)
val_dataset = SpeechDataset(annotations_file='val_annotations.csv', transform=mfcc_transform)

def collate_fn(batch):
    waveforms = []
    transcripts = []
    input_lengths = []
    target_lengths = []

    for waveform, transcript in batch:
        waveform = waveform.squeeze(0).transpose(0, 1)  # (T, F)
        waveforms.append(waveform)
        transcript_indices = torch.tensor(text_to_indices(transcript), dtype=torch.long)
        transcripts.append(transcript_indices)
        input_lengths.append(waveform.shape[0])  # Довжина по часовій осі
        target_lengths.append(len(transcript_indices))

    waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    transcripts = torch.cat(transcripts)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return waveforms, transcripts, input_lengths, target_lengths

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# 6. Підготовка словника та токенізатора

# Об'єднуємо всі транскрипції
all_text = ''.join(annotations['transcript']).lower()
all_chars = list(set(all_text))
all_chars.sort()

# Створюємо словник символів
char2idx = {char: idx + 1 for idx, char in enumerate(all_chars)}  # +1, бо 0 зарезервовано для бланку в CTC
idx2char = {idx + 1: char for idx, char in enumerate(all_chars)}

def text_to_indices(text):
    return [char2idx[char] for char in text.lower() if char in char2idx]

# 7. Створення моделі з покращеною архітектурою

class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim + 1)  # +1 для бланку в CTC

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.log_softmax(dim=2)
        return x

input_dim = 40  # Кількість коефіцієнтів MFCC
hidden_dim = 256  # Збільшено розмірність
output_dim = len(char2idx)

model = SpeechRecognitionModel(input_dim, hidden_dim, output_dim)

# 8. Функція втрат та оптимізатор

criterion = nn.CTCLoss(blank=0)  # Бланк має індекс 0
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Зменшено швидкість навчання

# 9. Тренування моделі

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for batch_idx, (waveforms, transcripts, input_lengths, target_lengths) in enumerate(dataloader):
        waveforms = waveforms.to(device)
        transcripts = transcripts.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        # Прогін вперед
        outputs = model(waveforms)

        # Підготовка для CTC Loss
        outputs = outputs.permute(1, 0, 2)  # (T, N, C)

        # Обчислення втрат
        loss = criterion(outputs, transcripts, input_lengths, target_lengths)

        # Перевірка на NaN або Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Batch {batch_idx}: Loss is NaN or Inf")
            continue

        # Зворотне розповсюдження
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate_epoch(model, device, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, (waveforms, transcripts, input_lengths, target_lengths) in enumerate(dataloader):
            waveforms = waveforms.to(device)
            transcripts = transcripts.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(waveforms)
            outputs = outputs.permute(1, 0, 2)

            loss = criterion(outputs, transcripts, input_lengths, target_lengths)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

num_epochs = 20  # Збільшено кількість епох

for epoch in range(num_epochs):
    train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
    val_loss = validate_epoch(model, device, val_loader, criterion)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 10. Збереження моделі та словників

torch.save(model.state_dict(), 'speech_recognition_model.pth')
torch.save(char2idx, 'char2idx.pth')
torch.save(idx2char, 'idx2char.pth')

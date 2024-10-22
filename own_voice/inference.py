# inference.py

import torch
import torchaudio
import torch.nn as nn
import sounddevice as sd
import wave
import numpy as np


# 1. Завантаження моделі та словників

class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim + 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.log_softmax(dim=2)
        return x


# Завантаження словників
char2idx = torch.load('char2idx.pth')
idx2char = torch.load('idx2char.pth')

input_dim = 40
hidden_dim = 256  # Відповідає тренованій моделі
output_dim = len(char2idx)

model = SpeechRecognitionModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('speech_recognition_model.pth', map_location='cpu'))
model.eval()


# 2. Функція декодування виходу моделі з додатковим відображенням

def decode_output(output):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        prev_index = -1
        print(f"Предбачені індекси для зразка {i}: {args.tolist()}")
        for index in args:
            index = index.item()
            if index != prev_index and index != 0:
                decode.append(idx2char.get(index, ''))
            prev_index = index
        decoded_text = ''.join(decode)
        print(f"Розпізнаний текст для зразка {i}: {decoded_text}")
        decodes.append(decoded_text)
    return decodes


# 3. Запис аудіо з мікрофона та збереження у файл

def record_audio(duration=5, sample_rate=16000, filename='recorded_audio.wav'):
    print("Початок запису...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Запис завершено.")

    # Збереження запису у файл WAV
    recording_int16 = np.int16(recording * 32767)  # Перетворення в 16-бітний формат
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 біт = 2 байти
        wf.setframerate(sample_rate)
        wf.writeframes(recording_int16.tobytes())
    print(f"Аудіо збережено у файл {filename}")

    return torch.from_numpy(recording.T)


# 4. Передбачення

def predict(model, audio):
    with torch.no_grad():
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
        audio_mfcc = mfcc_transform(audio)
        audio_mfcc = audio_mfcc.transpose(1, 2)  # Потрібно (N, T, F)
        outputs = model(audio_mfcc)
        print(f"Форма виходу моделі: {outputs.shape}")
        decoded_preds = decode_output(outputs)
    return decoded_preds


# 5. Використання моделі для розпізнавання мовлення

if __name__ == "__main__":
    # Записуємо аудіо з мікрофона та зберігаємо у файл
    audio = record_audio(duration=5, filename='recorded_audio.wav')

    # Передбачення
    predictions = predict(model, audio)

    print("Розпізнаний текст:")
    for pred in predictions:
        print(pred)

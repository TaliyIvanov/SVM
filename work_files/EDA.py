import os
import numpy as np
from collections import Counter

# Функция для получения размеров спектрограмм
def get_spectrogram_sizes(directory):
    sizes = []

    for speaker in os.listdir(directory):
        speaker_path = os.path.join(directory, speaker)
        for spec_file in os.listdir(speaker_path):
            spec_path = os.path.join(speaker_path, spec_file)
            spec = np.load(spec_path)
            sizes.append(spec.flatten().shape[0])  # Добавляем размер вектора спектрограммы

    return sizes

# Директории с чистыми и зашумленными аудио
clean_data_path = '/media/talium/Новый том/Интервью/01. Собес в ГосЗнак/01. Тестовое/data/train/clean'
noise_data_path = '/media/talium/Новый том/Интервью/01. Собес в ГосЗнак/01. Тестовое/data/train/noisy'

# Получаем размеры спектрограмм для чистых и зашумленных данных
clean_sizes = get_spectrogram_sizes(clean_data_path)
noise_sizes = get_spectrogram_sizes(noise_data_path)

# Объединяем размеры всех файлов
all_sizes = clean_sizes + noise_sizes

# Вычисляем статистики
mean_size = np.mean(all_sizes)
min_size = np.min(all_sizes)
max_size = np.max(all_sizes)
most_common_size = Counter(all_sizes).most_common(1)[0][0]  # Наиболее часто встречающийся размер

# Вывод результатов
print(f"Средний размер: {mean_size}")
print(f"Минимальный размер: {min_size}")
print(f"Максимальный размер: {max_size}")
print(f"Наиболее часто встречающийся размер: {most_common_size}")

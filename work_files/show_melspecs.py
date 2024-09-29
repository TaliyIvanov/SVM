import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем бэкенд без GUI
import matplotlib.pyplot as plt

# Функция для визуализации одной спектрограммы
def plot_spectrogram(spec, title="Spectrogram", output_file="spectrogram.png"):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bands')
    plt.savefig(output_file)  # Сохраняем изображение
    plt.close()  # Закрываем фигуру, чтобы освободить память

# Функция для визуализации двух спектрограмм рядом
def compare_spectrograms_visual(file1, file2, output_file="comparison.png"):
    # Загрузка данных
    spec1 = np.load(file1)
    spec2 = np.load(file2)
    
    # Проверка на одинаковую форму спектрограмм
    if spec1.shape != spec2.shape:
        print("Файлы имеют разные размеры или форму.")
        return

    # Визуализация двух спектрограмм
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Первая спектрограмма
    axs[0].imshow(spec1, aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title('Spectrogram 1')
    axs[0].set_xlabel('Time Frames')
    axs[0].set_ylabel('Frequency Bands')
    axs[0].colorbar = plt.colorbar(axs[0].imshow(spec1, aspect='auto', origin='lower', cmap='viridis'), ax=axs[0])

    # Вторая спектрограмма
    axs[1].imshow(spec2, aspect='auto', origin='lower', cmap='viridis')
    axs[1].set_title('Spectrogram 2')
    axs[1].set_xlabel('Time Frames')
    axs[1].set_ylabel('Frequency Bands')
    axs[1].colorbar = plt.colorbar(axs[1].imshow(spec2, aspect='auto', origin='lower', cmap='viridis'), ax=axs[1])

    plt.tight_layout()
    plt.savefig(output_file)  # Сохраняем изображение
    plt.close()  # Закрываем фигуру, чтобы освободить память

# Пример использования
file1_path = '/media/talium/Новый том1/Интервью/01. Собес в ГосЗнак/01. Тестовое/data/train/clean/20/20_205_20-205-0004.npy'
file2_path = '/media/talium/Новый том1/Интервью/01. Собес в ГосЗнак/01. Тестовое/data/train/noise/20/20_205_20-205-0004.npy'

compare_spectrograms_visual(file1_path, file2_path)

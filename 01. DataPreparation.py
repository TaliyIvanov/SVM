# Импорты
import os
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Функция для извлечения признаков (среднее и стандартное отклонение)
def extract_features(spec):
    # Среднее по строкам
    mean_freq = spec.mean(axis=0)
    # Стандартное отклонение по строкам
    std_freq = spec.std(axis=0)
    return np.concatenate([mean_freq, std_freq])

# Загрузчик данных с извлечением признаков
def load_data(clean_dir, noise_dir):
    X = []
    y = []

    # Загрузка чистых аудио
    for speaker in os.listdir(clean_dir):
        speaker_path = os.path.join(clean_dir, speaker)
        for spec_file in os.listdir(speaker_path):
            spec_path = os.path.join(speaker_path, spec_file)
            spec = np.load(spec_path)
            # Извлечение признаков из спектрограммы
            features = extract_features(spec)
            X.append(features)
            y.append(0)  # чистые аудио - 0

    # Загрузка зашумленных аудио
    for speaker in os.listdir(noise_dir):
        speaker_path = os.path.join(noise_dir, speaker)
        for spec_file in os.listdir(speaker_path):
            spec_path = os.path.join(speaker_path, spec_file)
            spec = np.load(spec_path)
            # Извлечение признаков из спектрограммы
            features = extract_features(spec)
            X.append(features)
            y.append(1)  # зашумленные аудио - 1

    return np.array(X), np.array(y)

# Пути к данным
clean_data_path = 'Your_path_to_clean'
noise_data_path = 'our_path_to_noise'

# Загрузка данных
X, y = load_data(clean_data_path, noise_data_path)  # Вызов функции для загрузки данных

# Разбиение на тренировочную и тестовую выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Понижение размерности методом Randomized PCA
pca = PCA(n_components=100, svd_solver='randomized', random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Сохранение обработанных данных для дальнейшего использования
np.save('X_train_pca.npy', X_train_pca)
np.save('X_val_pca.npy', X_val_pca)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

# Сохранение scaler и PCA
import joblib
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(pca, 'pca.joblib')
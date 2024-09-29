# Импорты
import numpy as np
import joblib

# Функция для извлечения признаков (среднее и стандартное отклонение)
def extract_features(spec):
    # Среднее по строкам
    mean_freq = spec.mean(axis=0)
    # Стандартное отклонение по строкам
    std_freq = spec.std(axis=0)
    return np.concatenate([mean_freq, std_freq])

# Загрузка модели, scaler и PCA
svm_model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')

# Предсказание для новой спектрограммы
def predict_for_spec(spec_path):
    # Шаг 1: Загрузка спектрограммы
    spec = np.load(spec_path)
    
    # Шаг 2: Извлечение признаков
    features = extract_features(spec)
    
    # Шаг 3: Стандартизация признаков
    features_scaled = scaler.transform([features])
    
    # Шаг 4: Применение PCA для понижения размерности
    features_pca = pca.transform(features_scaled)
    
    # Шаг 5: Предсказание с помощью модели SVM
    prediction = svm_model.predict(features_pca)
    
    return prediction[0]  # Вернуть результат предсказания (0 - чистый звук, 1 - зашумленный звук)

# Пример использования
spec_path = 'Укажите путь к новой спектрограмме.npy'
prediction = predict_for_spec(spec_path)

# Вывод результата
if prediction == 0:
    print("Чистое аудио")
else:
    print("Зашумленное аудио")

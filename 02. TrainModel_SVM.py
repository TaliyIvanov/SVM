# Импорты
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np 

# Загрузка предобработанных данных
X_train = np.load('X_train_pca.npy')
X_val = np.load('X_val_pca.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Обучение модели SVM
svm_model = SVC(kernel='rbf', C=1, gamma='auto', random_state=42)
svm_model.fit(X_train, y_train)

# Оценка модели на валидационной выборке
y_val_pred = svm_model.predict(X_val)

# Вывод результатов
print(f'Accuracy: {accuracy_score(y_val, y_val_pred)}')
print(classification_report(y_val, y_val_pred))

# Сохранение модели

import joblib
joblib.dump(svm_model, 'svm_model.joblib')
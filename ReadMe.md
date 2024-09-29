### Описание файлов:

1. **01. DataPreparation.py**: 
   Этот скрипт загружает пары mel-спектрограмм из директорий с чистыми и зашумленными аудиофайлами, извлекает из них признаки (среднее и стандартное отклонение частотных компонентов) и разделяет данные на тренировочные и тестовые выборки.

2. **02. TrainModel_SVM.py**: 
   Скрипт обучает модель SVM для классификации аудиофайлов на чистые и зашумленные. Для этого он использует предварительно подготовленные признаки, а также применяет методы стандартизации данных и понижения размерности с использованием PCA. Обученная модель сохраняется для последующего использования.

3. **03. Predictions.py**: 
   Этот скрипт загружает обученную модель SVM, а также параметры стандартизации и PCA. Он принимает на вход путь к новой спектрограмме, извлекает признаки, нормализует их, применяет PCA и делает предсказание, является ли аудио чистым или зашумленным.

### Установка зависимостей:

Скрипты используют следующие библиотеки:
- `numpy`
- `sklearn`
- `joblib`

Чтобы установить необходимые зависимости, выполните команду:
`pip install numpy scikit-learn joblib`

### Запуск скриптов:

1. **Подготовка данных (01. DataPreparation.py)**

   Скрипт загружает спектрограммы и извлекает признаки для обучения модели. Перед запуском укажите правильные пути к директориям с чистыми и зашумленными данными.

   `python 01. DataPreparation.py`

   После запуска создаются массивы признаков и меток, которые можно использовать для обучения.

2. **Обучение модели (02. TrainModel_SVM.py)**

   После подготовки данных используйте этот скрипт для обучения модели SVM. Он выполнит стандартизацию данных, применит PCA и обучит модель на подготовленных данных. Модель будет сохранена в файл `svm_model.joblib`.

   `python 02. TrainModel_SVM.py`

   Вывод будет содержать точность классификации и отчёт о качестве модели.

3. **Предсказания для новых данных (03. Predictions.py)**

   После обучения модели можно использовать её для предсказаний. Для этого укажите путь к файлу спектрограммы в переменной `spec_path`, затем выполните скрипт:

   `python 03. Predictions.py`

   Скрипт выведет результат предсказания: "Чистое аудио" или "Зашумленное аудио".


### Структура данных:

- **Чистые данные** должны находиться в папке, путь к которой задаётся переменной `clean_data_path`.
- **Зашумленные данные** должны быть в папке, путь к которой задаётся переменной `noise_data_path`.
- Все файлы спектрограмм должны быть в формате `.npy`.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Задаем константы для воспроизводимости результатов
RANDOM_STATE = 42
FIGURE_SIZE = (12, 8)

# Пути
TITANIC_DATASET_PATH = r'data\\titanic.csv'
GRAPHICS_SAVE_PATH = r'report\\graphics\\'


class TitanicClassifier:
    """
    Класс для классификации выживших на Титанике с использованием различных моделей.
    """

    def __init__(self, data_path=TITANIC_DATASET_PATH):
        """
        Инициализация класса.
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None  # Объявляем preprocessor здесь
        self.models = {}  # Словарь для хранения обученных моделей


    def load_data(self):
        """
        Загружает данные из CSV файла.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print('Данные успешно загружены.')
        except FileNotFoundError:
            print(f'Ошибка: Файл {self.data_path} не найден.')
            exit()


    def preprocess_data(self):
        """
        Предварительная обработка данных: обработка пропусков, кодирование категориальных признаков, масштабирование.
        """
        # Определяем числовые и категориальные признаки
        numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
        categorical_features = ['Sex', 'Embarked', 'Pclass']

        # Создаем предобработчики для числовых и категориальных признаков
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Объединяем предобработчики с помощью ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Остальные столбцы удаляем
        )

        # Разделяем данные на признаки (X) и целевую переменную (y)
        X = self.data.drop('Survived', axis=1)
        y = self.data['Survived']

        # Разделяем данные на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Применяем предобработку к обучающей и тестовой выборкам
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)

        print('Данные успешно предобработаны.')


    def train_svm(self, kernel='rbf', C=1, gamma='scale'):
        """
        Обучает модель SVM.

        :param kernel: Ядро SVM ('linear', 'rbf', 'poly', 'sigmoid').
        :param C: Параметр регуляризации.
        :param gamma: Параметр ядра ('scale', 'auto' или числовое значение).
        """
        svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=RANDOM_STATE)
        svm_model.fit(self.X_train, self.y_train)
        self.models['svm'] = svm_model  # Сохраняем модель в словаре
        print(f'Модель SVM с ядром {kernel} успешно обучена.')


    def train_logistic_regression(self, C=1.0):
        """
        Обучает модель логистической регрессии.

        :param C: Параметр регуляризации.
        """
        logistic_model = LogisticRegression(C=C, random_state=RANDOM_STATE, max_iter=1000)
        logistic_model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = logistic_model  # Сохраняем модель
        print('Модель логистической регрессии успешно обучена.')


    def train_random_forest(self, n_estimators=100, max_depth=None):
        """
        Обучает модель случайного леса.

        :param n_estimators: Количество деревьев в лесу.
        :param max_depth: Максимальная глубина дерева.
        """
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE)
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf_model  # Сохраняем модель
        print('Модель случайного леса успешно обучена.')


    def evaluate_model(self, model_name='svm'):
        """
        Оценивает производительность модели на тестовой выборке.

        :param model_name: Название модели для оценки ('svm', 'logistic_regression', 'random_forest').
        """
        model = self.models.get(model_name)  # Получаем модель из словаря
        if model is None:
            print(f'Ошибка: Модель {model_name} не обучена.')
            return

        y_pred = model.predict(self.X_test)
        print(f'Отчет о классификации для модели {model_name}:\n', classification_report(self.y_test, y_pred))

        # Строим и отображаем матрицу ошибок
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Не выжил', 'Выжил']).plot(cmap='Blues')
        plt.title(f'Матрица ошибок для модели {model_name}', pad=15)
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Фактические значения')
        plt.savefig(f'{GRAPHICS_SAVE_PATH}confusion_matrix_{model_name}.png')
        plt.show()


    def tune_parameters_grid_search(self, model_name='svm', param_grid=None):
        """
        Настраивает параметры модели с использованием Grid Search.

        :param model_name: Название модели для настройки ('svm').
        :param param_grid: Словарь с параметрами для перебора.
        """
        model = self.models.get(model_name)  # Получаем модель из словаря
        if model is None:
            print(f'Ошибка: Модель {model_name} не обучена.')
            return

        if param_grid is None:
            if model_name == 'svm':
                param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
            elif model_name == 'logistic_regression':
                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
            elif model_name == 'random_forest':
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            else:
                print(f'Ошибка: Неизвестное имя модели: {model_name}')
                return

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        print('Лучшие параметры:', grid_search.best_params_)
        print('Лучший результат:', grid_search.best_score_)

        # Обновляем модель с лучшими параметрами
        self.models[model_name] = grid_search.best_estimator_


    def plot_learning_curve(self, model_name='svm'):
        """
        Строит кривые обучения для оценки влияния размера обучающей выборки на производительность модели.

        :param model_name: Название модели для построения кривой обучения ('svm').
        """
        model = self.models.get(model_name)  # Получаем модель из словаря
        if model is None:
            print(f'Ошибка: Модель {model_name} не обучена.')
            return

        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(num=f'Кривая обучения для модели {model_name}', figsize=FIGURE_SIZE)
        plt.plot(train_sizes, train_mean, label='Обучающая выборка', color='blue')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, color='blue', alpha=0.15)
        plt.plot(train_sizes, test_mean, label='Тестовая выборка', color='green')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='green', alpha=0.15)

        plt.title(f'Кривая обучения для модели {model_name}', pad=15)
        plt.xlabel('Размер обучающей выборки')
        plt.ylabel('Точность')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'{GRAPHICS_SAVE_PATH}learning_curve_{model_name}.png')
        plt.show()


    def run(self):
        """
        Запускает весь пайплайн: загрузка, предобработка, обучение и оценка моделей.
        """
        self.load_data()
        self.preprocess_data()

        # Обучаем и оцениваем SVM
        self.train_svm()
        self.evaluate_model('svm')
        self.tune_parameters_grid_search('svm')
        self.evaluate_model('svm')  # Оцениваем после настройки параметров
        self.plot_learning_curve('svm')

        # Обучаем и оцениваем логистическую регрессию
        self.train_logistic_regression()
        self.evaluate_model('logistic_regression')
        self.tune_parameters_grid_search('logistic_regression')
        self.evaluate_model('logistic_regression')  # Оцениваем после настройки параметров
        self.plot_learning_curve('logistic_regression')

        # Обучаем и оцениваем случайный лес
        self.train_random_forest()
        self.evaluate_model('random_forest')
        self.tune_parameters_grid_search('random_forest')
        self.evaluate_model('random_forest')  # Оцениваем после настройки параметров
        self.plot_learning_curve('random_forest')


if __name__ == '__main__':
    classifier = TitanicClassifier(data_path=TITANIC_DATASET_PATH)
    classifier.run()
import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def run_model_iterations(pipeline, X_train, y_train, X_test, y_test, n_iterations=100):
    """
    Запускает итерации обучения модели и возвращает массив ROC AUC.

    :param X_train: Признаки для обучения
    :param y_train: Целевые значения для обучения
    :param X_test: Признаки для тестирования
    :param y_test: Целевые значения для тестирования
    :param n_iterations: Количество итераций обучения модели
    :return: Массив ROC AUC
    """
    # imputer = SimpleImputer(strategy='most_frequent') 
    # model = LogisticRegression()
    # pipeline = Pipeline(steps=[('imputer', imputer), ('model', model)])
    
    roc_auc_scores = []

    for i in tqdm(range(n_iterations), desc="Training iterations"):
        # Устанавливаем случайное состояние
        random_state = i 
        pipeline.named_steps["model"].set_params(random_state=random_state)

        # обучаем модель и считаем метрику
        pipeline.fit(X_train, y_train)
        y_scores = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_scores)
        roc_auc_scores.append(roc_auc)

    return np.array(roc_auc_scores)

def compare_model_statistics(roc_auc_model1, roc_auc_model2):
    """
    Сравнивает статистики ROC AUC двух моделей с помощью t-теста.

    :param roc_auc_model1: Массив ROC AUC для первой модели
    :param roc_auc_model2: Массив ROC AUC для второй модели
    :return: t-статистика и p-значение
    """

    # Вычисление средних значений ROC AUC
    mean_model1 = np.mean(roc_auc_model1)
    mean_model2 = np.mean(roc_auc_model2)
    print("Среднее ROC AUC Model 1:", mean_model1)
    print("Среднее ROC AUC Model 2:", mean_model2)
    
    # Определение, какая модель имеет большую метрику
    if mean_model1 > mean_model2:
        print(f"Модель 1 имеет большую метрику ROC AUC на {mean_model1 - mean_model2}.")
    elif mean_model2 > mean_model1:
        print(f"Модель 2 имеет большую метрику ROC AUC на {mean_model2 - mean_model1}.")
    else:
        print("Обе модели имеют одинаковую метрику ROC AUC.")
    
    # Выполнение t-теста
    t_statistic, p_value = stats.ttest_ind(roc_auc_model1, roc_auc_model2)

    # Вывод t-статистики и p-значения
    print("t-статистика:", t_statistic)
    print("p-значение:", p_value)

    # Сравнение метрик
    if p_value < 0.05:
        print("Существует статистически значимая разница между моделями.")
    else:
        print("Нет статистически значимой разницы между моделями.")


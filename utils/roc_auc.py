import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(pipeline, X_test, y_test, need_plot=1):
    """
    Строит ROC-кривую и вычисляет AUC для заданной модели и тестовых данных.

    :param pipeline: Обученная модель (pipeline)
    :param X_test: Тестовые данные
    :param y_test: Истинные метки классов
    """
    # Предсказания вероятностей на тестовом наборе
    y_scores = pipeline.predict_proba(X_test)[:, 1]  # Вероятности для положительного класса

    # Вычисляем ROC-кривую и AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Построение ROC-кривой
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC-кривая (AUC = {:.6f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Линия случайного выбора
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ложноположительная ставка')
    plt.ylabel('Истинноположительная ставка')
    plt.title('ROC-кривая')
    plt.legend(loc='lower right')
    if need_plot:
        plt.show()
    
    return roc_auc
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import TomekLinks

class BalanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='smote', sampling_strategy='auto', **kwargs):
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.sampler = None
        print(self.method )
        # Инициализация соответствующего метода
        if self.method == 'smote':
            self.sampler = SMOTE(sampling_strategy=self.sampling_strategy, **kwargs)
        elif self.method == 'adasyn':
            self.sampler = ADASYN(sampling_strategy=self.sampling_strategy, **kwargs)
        elif self.method == 'tomek':
            self.sampler = TomekLinks()
        elif self.method == 'smote_enn':
            self.sampler = SMOTEENN(sampling_strategy=self.sampling_strategy, **kwargs)
        elif self.method == 'smote_tomek':
            self.sampler = SMOTETomek(sampling_strategy=self.sampling_strategy, **kwargs)
        else:
            raise ValueError("Метод должен быть одним из: 'smote', 'adasyn', 'tomek', 'smote_enn', 'smote_tomek'.")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Метод transform не будет изменять X, так как балансировка происходит в fit_resample
        return X

    def fit_resample(self, X, y):
        if y is None:
            raise ValueError("Целевая переменная 'y' должна быть передана в метод fit_resample.")
        
        if y.nunique() < 2:
            raise ValueError("Целевая переменная должна содержать как минимум два класса.")
        
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        return X_resampled, y_resampled

def plot_class_distribution(y, title):
    """Функция для визуализации распределения классов."""
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
    
    
if __name__ == "__main__":
    
    # Тестирование
    data = {
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'target': np.concatenate([np.zeros(950), np.ones(50)])  # Дисбаланс: 950 классов 0 и 50 классов 1
    }
    
    df = pd.DataFrame(data)
    plot_class_distribution(df['target'], title='Исходное распределение классов')
    X_resampled, y_resampled = handle_class_imbalance(df, target_column='target', method='smote')
    plot_class_distribution(y_resampled, title='Распределение классов после SMOTE')
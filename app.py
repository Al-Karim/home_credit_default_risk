import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Загрузка модели
model = joblib.load('models/calibrated_xgb_model.pkl')

# Определение признаков и их типов с пояснениями
features_info = {
    'DAYS_BIRTH': {
        'type': 'int',
        'description': '- количество дней с момента рождения до текущей даты.',
        'default': -10000  # Пример значения по умолчанию
    },
    'REGION_RATING_CLIENT': {
        'type': 'int',
        'description': 'Рейтинг региона, в котором проживает клиент',
        'default': 1  # Пример значения по умолчанию
    },
    'REGION_RATING_CLIENT_W_CITY': {
        'type': 'int',
        'description': 'Рейтинг региона, в котором проживает клиент, с учетом города',
        'default': 1  # Пример значения по умолчанию
    },
    'EXT_SOURCE_1': {
        'type': 'float',
        'description': 'Нормализованный скор из внешнего неизвестного источника',
        'default': 0.5  # Пример значения по умолчанию
    },
    'EXT_SOURCE_2': {
        'type': 'float',
        'description': 'Нормализованный скор из внешнего неизвестного источника',
        'default': 0.5  # Пример значения по умолчанию
    },
    'EXT_SOURCE_3': {
        'type': 'float',
        'description': 'Нормализованный скор из внешнего неизвестного источника',
        'default': 0.5  # Пример значения по умолчанию
    },
    'NAME_EDUCATION_TYPE_Higher education': {
        'type': 'bool',
        'description': 'Образования клиента - высшее?',
        'default': False  # Пример значения по умолчанию
    },
    'BUREAU_DAYS_CREDIT_MEAN': {
        'type': 'float',
        'description': 'Среднее количество дней кредита в бюро.',
        'default': 0.0  # Пример значения по умолчанию
    },
    'BUREAU_DAYS_CREDIT_MIN': {
        'type': 'float',
        'description': 'Минимальное количество дней кредита в бюро.',
        'default': 0.0  # Пример значения по умолчанию
    },
    'CREDIT_CARD_POSITIVE_STATUS': {
        'type': 'float',
        'description': 'Процент месяцев, в течение которых статус контракта по кредитной карте был "Completed" (завершен)',
        'default': 0.0  # Пример значения по умолчанию
    },
    'INSTALLMENTS_TOTAL_PAYMENT': {
        'type': 'float',
        'description': 'Общая сумма платежей по рассрочке.',
        'default': 0.0  # Пример значения по умолчанию
    }
}

# Заголовок приложения
st.title("Кредитный скоринг с использованием модели XGBoost на основе 11 самых важных признаков")

# Ввод данных пользователем
input_data = {}
for feature, info in features_info.items():
    # Добавляем пояснение для каждого признака
    st.markdown(f"**{feature}**: {info['description']}")
    
    if info['type'] == 'int':
        if feature == 'REGION_RATING_CLIENT_W_CITY' or feature == 'REGION_RATING_CLIENT':
            input_data[feature] = st.selectbox(
                f"Выберите значение для {feature}",
                options=[1, 2, 3],
                index=info.get('default', 0) - 1  # Установка значения по умолчанию
            )
        else:
            input_data[feature] = st.number_input(f"Введите значение для {feature}", value=info['default'], step=1, format="%d")
            if feature == 'DAYS_BIRTH' and input_data[feature] >= 0:
                st.error("Значение для DAYS_BIRTH должно быть отрицательным.")
    elif info['type'] == 'float':
        input_data[feature] = st.number_input(f"Введите значение для {feature}", value=info['default'], step=0.1, format="%.2f")
    elif info['type'] == 'bool':
        input_data[feature] = st.checkbox(f"{feature} (True/False)")
        
    # Добавляем разделитель между пунктами
    st.markdown("<hr>", unsafe_allow_html=True)

# Преобразование входных данных в DataFrame
input_df = pd.DataFrame([input_data])

# Кнопка для предсказания
if st.button("Сделать прогноз"):
    # Предсказание
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]  # Вероятность положительного класса

    # Отображение результата
    st.write(f"Откалиброванная вероятность: {prediction_proba[0]:.4f} <small>(1 - клиент с трудностями в оплате: он/она имел просрочку более чем на X дней хотя бы по одному из первых Y платежей по кредиту в нашей выборке, 0 - все остальные случаи)</small>", unsafe_allow_html=True)

    # Объяснение с помощью LIME
    explainer = LimeTabularExplainer(
        training_data=np.array(pd.read_csv('./data/processed_data/processed_application_train_with_features.csv')[list(features_info.keys())]),  # Замените на ваши данные
        feature_names=list(features_info.keys()),
        class_names=['Negative', 'Positive'],
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        input_df.values[0], 
        model.predict_proba, 
        num_features=11
    )

    # Визуализация объяснения
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)

# streamlit run app.py

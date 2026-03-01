import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Optional, Any, Dict


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розбиває дані на тренувальний та валідаційний набори.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return train_df, val_df

def get_feature_cols(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Автоматично визначає числові та категоріальні колонки, прибираючи Surname та ID.
    """
    # За рекомендацією завдання прибираємо 'Surname' та службові колонки 'id', 'CustomerId'
    cols_to_drop = [target_col, 'Surname', 'id', 'CustomerId']
    input_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    numeric_cols = input_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = input_df.select_dtypes(include='object').columns.tolist()
    
    return numeric_cols, categorical_cols

def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Масштабує числові ознаки.
    """
    scaler = StandardScaler().fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler

def encode_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Кодує категоріальні ознаки (One-Hot Encoding).
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    def transform_split(df):
        encoded = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        return pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    
    return transform_split(train_df), transform_split(val_df), encoder, encoded_cols

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, Any]:
    """
    Повний цикл обробки даних: вибір колонок, розбиття, кодування та масштабування.
    """
    target_col = 'Exited'
    
    # 1. Розбиття
    train_df, val_df = split_data(raw_df, target_col)
    
    # 2. Визначення колонок
    numeric_cols, categorical_cols = get_feature_cols(train_df, target_col)
    
    # 3. Кодування
    train_df, val_df, encoder, encoded_cols = encode_features(train_df, val_df, categorical_cols)
    
    # 4. Масштабування (опціонально)
    scaler = None
    if scaler_numeric:
        train_df, val_df, scaler = scale_features(train_df, val_df, numeric_cols)
    
    # Формування фінальних списків вхідних ознак
    input_cols = numeric_cols + encoded_cols
    
    return {
        'X_train': train_df[input_cols],
        'train_targets': train_df[target_col],
        'X_val': val_df[input_cols],
        'val_targets': val_df[target_col],
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str], scaler: Optional[StandardScaler], encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Обробляє нові дані (наприклад, test.csv) за допомогою вже навчених скейлера та енкодера.
    """
    # 1. Видалення непотрібних колонок
    df = new_df.copy()
    
    # 2. Кодування категоріальних ознак
    categorical_cols = encoder.feature_names_in_.tolist()
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    
    # 3. Масштабування (якщо передано скейлер)
    if scaler:
        numeric_cols = scaler.feature_names_in_.tolist()
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # 4. Відбір лише потрібних колонок (тих, що були в X_train)
    return df[input_cols]
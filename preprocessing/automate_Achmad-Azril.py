"""
automate_Achmad-Azril.py
Script untuk melakukan preprocessing data Exam Score Prediction secara otomatis.

Author: Achmad Azril
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse

def load_data(file_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Data loaded successfully. Shape: {df.shape}")
    return df

def drop_unnecessary_columns(df: pd.DataFrame, columns: list = ['Id']) -> pd.DataFrame:
    print(f"[INFO] Dropping columns: {columns}")
    existing_cols = [col for col in columns if col in df.columns]
    df = df.drop(columns=existing_cols, axis=1)
    print(f"[INFO] Shape after dropping: {df.shape}")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"[INFO] Removed {removed} duplicate rows. Remaining: {len(df)}")
    return df

def cap_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    print(f"[INFO] Capping outliers using IQR method...")
    df = df.copy()
    for column in columns:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    print(f"[INFO] Outliers capped successfully")
    return df

def split_features_target(df: pd.DataFrame, target_column: str = 'exam_score'):
    print(f"[INFO] Splitting features and target...")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def split_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    print(f"[INFO] Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, scaler=None):
    print(f"[INFO] Scaling features using StandardScaler...")
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    print(f"[INFO] Scaling completed. Mean: {X_train_scaled.mean().mean():.6f}, Std: {X_train_scaled.std().mean():.6f}")
    return X_train_scaled, X_test_scaled, scaler

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    train_data = X_train.copy()
    train_data['exam_score'] = y_train.values
    test_data = X_test.copy()
    test_data['exam_score'] = y_test.values
    train_path = os.path.join(output_dir, 'ExamScore_train.csv')
    test_path = os.path.join(output_dir, 'ExamScore_test.csv')
    all_path = os.path.join(output_dir, 'ExamScore_preprocessing.csv')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    all_data = pd.concat([train_data, test_data], axis=0)
    all_data.to_csv(all_path, index=False)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Files saved to {output_dir}:")
    print(f"  - {train_path} ({len(train_data)} rows)")
    print(f"  - {test_path} ({len(test_data)} rows)")
    print(f"  - {all_path} ({len(all_data)} rows)")
    print(f"  - {scaler_path}")
    return train_data, test_data, all_data

def preprocess(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    print("=" * 60)
    print("EXAM SCORE DATA PREPROCESSING")
    print("=" * 60)
    df = load_data(input_path)
    df = drop_unnecessary_columns(df, columns=['Id'])
    df = remove_duplicates(df)
    # Optional: tentukan kolom numerik untuk outlier capping jika perlu
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop('exam_score').tolist()
    df = cap_outliers_iqr(df, numeric_features)
    X, y = split_features_target(df, target_column='exam_score')
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    train_data, test_data, all_data = save_preprocessed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, output_dir
    )
    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'train_data': train_data,
        'test_data': test_data,
        'all_data': all_data
    }

def get_preprocessed_data(output_dir: str):
    train_path = os.path.join(output_dir, 'ExamScore_train.csv')
    test_path = os.path.join(output_dir, 'ExamScore_test.csv')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_train = train_data.drop('exam_score', axis=1)
    y_train = train_data['exam_score']
    X_test = test_data.drop('exam_score', axis=1)
    y_test = test_data['exam_score']
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exam Score Data Preprocessing')
    parser.add_argument('--input', type=str, default='../Exam_Score_Prediction_raw/Exam_Score_Prediction.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='./',
                        help='Output directory for preprocessed files')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    result = preprocess(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"\nFinal dataset shapes:")
    print(f"  Training: {result['X_train'].shape}")
    print(f"  Testing: {result['X_test'].shape}")
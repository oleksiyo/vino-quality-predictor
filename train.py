import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import xgboost as xgb


def load_data(path="./data/WineQT.csv"):
    df = pd.read_csv(path) 
    
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    return df


def feature_engineering(df):
    df = df.copy()
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
    df['good_acid_ratio'] = df['citric_acid'] / (df['volatile_acidity'] + 0.01)
    df['alcohol_sulphate'] = df['alcohol'] * df['sulphates']
    return df


def prepare_data(df):
    df = feature_engineering(df)
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")
    
    return rmse, r2


def save_artifacts(model, model_path="model.pkl"):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved as {model_path}")


def main():
    df = load_data()
    X_train, X_val, y_train, y_val = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate(model, X_val, y_val)
    save_artifacts(model, "wine_quality_model.pkl")


if __name__ == "__main__":
    main()
import pickle
import pandas as pd

MODEL_PATH = "wine_quality_model.pkl"

with open(MODEL_PATH, 'rb') as f_in:
    model = pickle.load(f_in)


def feature_engineering_single(sample: dict):
    df = pd.DataFrame([sample])
    
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
    df['good_acid_ratio'] = df['citric_acid'] / (df['volatile_acidity'] + 0.01)
    df['alcohol_sulphate'] = df['alcohol'] * df['sulphates']
    
    return df

def predict_single(wine):
    result = model.predict(wine)[0]
    return float(result)

def lambda_handler(event, context):
    wine = event['wine']
    wine = {k.lower().replace(' ', '_'): v for k, v in wine.items()}

    df_processed = feature_engineering_single(wine)
    
    wine_predict = predict_single(df_processed)
    
    return {
        "predicted_quality": round(float(wine_predict), 3)
    }
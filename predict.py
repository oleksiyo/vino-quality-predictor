import pickle
import pandas as pd

MODEL_PATH = "wine_quality_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f_in:
        model = pickle.load(f_in)
    return model


def feature_engineering_single(sample: dict):
    df = pd.DataFrame([sample])
    
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
    df['good_acid_ratio'] = df['citric_acid'] / (df['volatile_acidity'] + 0.01)
    df['alcohol_sulphate'] = df['alcohol'] * df['sulphates']
    
    return df


def predict(sample: dict):
    model = load_model()
    
    df_processed = feature_engineering_single(sample)
    
    prediction = model.predict(df_processed)[0]
    
    return round(float(prediction), 3)


if __name__ == "__main__":
    example_wine = {
        "fixed_acidity": 11.2,
        "volatile_acidity": 0.28,
        "citric_acid": 0.56,
        "residual_sugar": 1.9,
        "chlorides": 0.075,
        "free_sulfur_dioxide": 17.0,
        "total_sulfur_dioxide": 60.0,
        "density": 0.9980,
        "ph": 3.16,
        "sulphates": 0.58,
        "alcohol": 9.8
    }

    predicted_quality = predict(example_wine)
    print(f"Predicted wine quality: {predicted_quality}")
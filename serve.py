from flask import Flask, request, jsonify
import pickle
import pandas as pd

MODEL_PATH = "wine_quality_model.pkl"

app = Flask('wine-quality-predictor')


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def feature_engineering_single(sample: dict):
    df = pd.DataFrame([sample])
    
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
    df['good_acid_ratio'] = df['citric_acid'] / (df['volatile_acidity'] + 0.01)
    df['alcohol_sulphate'] = df['alcohol'] * df['sulphates']
    
    return df


@app.route('/', methods=['GET'])
def root():
    return {'status': 'OK', 'message': 'AI Sommelier: Wine Quality Prediction'}

@app.route('/health', methods=['GET'])
def health():
    return {"status": "OK"}

@app.route('/predict', methods=['POST'])
def predict():
    sample = request.get_json()

     # normalize keys
    sample = {k.lower().replace(' ', '_'): v for k, v in sample.items()}
    
    if not hasattr(predict, "model"):
        predict.model = load_model()
    
    df_processed = feature_engineering_single(sample)
    
    prediction = predict.model.predict(df_processed)[0]
    
    result = {
        "predicted_quality": round(float(prediction), 3)
    }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
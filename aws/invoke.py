
import boto3
import json

lambda_client = boto3.client('lambda')

wine_data = {
    "wine": {
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
}

response = lambda_client.invoke(
    FunctionName='vino-quality-lambda',
    InvocationType='RequestResponse',
    Payload=json.dumps(wine_data)
)

result = json.loads(response['Payload'].read())
print(json.dumps(result, indent=2))

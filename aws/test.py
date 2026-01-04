import requests
import json

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

wine =  {
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

result = requests.post(url, json= {"wine": wine}).json()
print(result)
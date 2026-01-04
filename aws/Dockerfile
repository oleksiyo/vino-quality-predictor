FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt .
COPY aws/lambda_function.py ./lambda_function.py
COPY wine_quality_model.pkl .

RUN pip install --no-cache-dir -r requirements.txt


CMD ["lambda_function.lambda_handler"]
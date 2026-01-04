#!/bin/bash

AWS_PROFILE=iamadmin-general
export AWS_PROFILE

IMAGE_NAME="vino-quality-lambda"
AWS_REGION="us-east-1"
FUNCTION_NAME="vino-quality-function"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity | jq -r ".Account")
LAMBDA_ROLE="lambda-promtail-loki-role-yxsxajcp"



ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${LAMBDA_ROLE}"

# Get latest commit SHA and current datetime
COMMIT_SHA=$(git rev-parse --short HEAD)
DATETIME=$(date +"%Y%m%d-%H%M%S")
IMAGE_TAG="${COMMIT_SHA}-${DATETIME}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${IMAGE_NAME}:${IMAGE_TAG}"

## run it only once
aws ecr create-repository \
  --repository-name ${IMAGE_NAME} \
  --region ${AWS_REGION}

aws ecr get-login-password \
  --region ${AWS_REGION} \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URI}


 echo  ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_URI}

docker build -f aws/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_URI}
docker push ${IMAGE_URI}

# if aws lambda get-function \
#   --function-name "$FUNCTION_NAME" \
#   --region "$AWS_REGION" > /dev/null 2>&1; then
    
#     echo "Lambda exists. Updating image..."
#     aws lambda update-function-code \
#       --function-name "$FUNCTION_NAME" \
#       --image-uri "$IMAGE_URI" \
#       --region "$AWS_REGION"

# else
#     echo "Lambda does not exist. Creating..."
#     echo ${ROLE_ARN}
#     aws lambda create-function \
#       --function-name ${FUNCTION_NAME} \
#       --package-type Image \
#       --code ImageUri=${IMAGE_URI} \
#       --role ${ROLE_ARN} \
#       --region ${AWS_REGION}
# fi

# AWS Deployment cost effective Workflow for Premium Prediction ML Pipeline using free tier resources


This document outlines the complete workflow for deploying the Premium Prediction ML pipeline to AWS.

## Optimized Deployment Strategy

1. Replace EKS with Lightweight Alternatives : EKS costs $0.10/hour (~$73/month) which is not covered by Free Tier. Instead:

# Option 1: Use AWS Elastic Beanstalk (Free Tier eligible)
```bash
aws elasticbeanstalk create-application --application-name premium-model-app
aws elasticbeanstalk create-environment \
  --application-name premium-model-app \
  --environment-name premium-model-dev \
  --solution-stack-name "64bit Amazon Linux 2 v3.4.9 running Python 3.8" \
  --option-settings file://eb-options.json

# Option 2: Use AWS Lambda + API Gateway for inference (Free Tier eligible)
aws lambda create-function \
  --function-name premium-model-inference \
  --runtime python3.8 \
  --handler app.lambda_handler \
  --zip-file fileb://deployment-package.zip \
  --role arn:aws:iam::$AWS_ACCOUNT_ID:role/lambda-execution-role
```

## Key Benefits of This Approach

1. **No EKS costs** - Using Lambda and Elastic Beanstalk instead of EKS saves ~$73/month
2. **Minimal ECR usage** - Lifecycle policies keep you under the 0.5GB free limit
3. **Efficient S3 usage:** - Batch operations reduce request count to stay within free tier
4. **Serverless architecture** - Pay only for actual usage, with generous free tiers
5. **CloudWatch optimization** - Consolidating metrics into fewer dashboards stays within free tier
6. **AWS CodePipeline** - For CI/CD automation

## Prerequisites

- AWS CLI installed and configured
- Docker installed

## Step 1: Optimize ECR Usage (0.5GB Free)


```bash
# Use multi-stage builds to minimize image size
cat > Dockerfile << EOF
FROM python:3.8-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.8-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY . .
CMD ["python", "app.py"]
EOF

# Clean up old images regularly
aws ecr list-images --repository-name premium-model-api --query 'imageIds[?type(imageTag)!=`string`].[imageDigest]' --output text | xargs -I {} aws ecr batch-delete-image --repository-name premium-model-api --image-ids imageDigest={}

```

### Optimize S3 Usage (Stay within request limits)

```bash
# Create buckets for model artifacts and MLflow data
# Batch operations to reduce number of requests
aws s3 sync local-directory s3://premium-model-artifacts/models --delete

# Use versioning instead of creating new files
aws s3api put-bucket-versioning --bucket premium-model-artifacts --versioning-configuration Status=Enabled

```

### Use CloudWatch Efficiently (3 free dashboards)

```bash
# Create a single comprehensive dashboard instead of multiple ones
cat > deploy/cloudwatch-dashboard.json << EOF
{
  "widgets": [
    {
      "type": "metric",
      "x": 0,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["PremiumPredictionAPI", "model_prediction_latency"],
          ["PremiumPredictionAPI", "prediction_count"],
          ["PremiumPredictionAPI", "feature_drift"],
          ["PremiumPredictionAPI", "model_error"]
        ],
        "period": 3600,
        "stat": "Average",
        "region": "$AWS_REGION",
        "title": "Model Performance Metrics"
      }
    }
  ]
}
EOF

```

## Serverless Training Pipeline


```bash
# Create a Lambda function for training
aws lambda create-function \
  --function-name premium-model-training \
  --runtime python3.8 \
  --timeout 900 \
  --memory-size 3008 \
  --handler train.lambda_handler \
  --zip-file fileb://training-package.zip \
  --role arn:aws:iam::$AWS_ACCOUNT_ID:role/lambda-training-role

# Set up scheduled training (once per week)
aws events put-rule \
  --name weekly-model-training \
  --schedule-expression "rate(7 days)"

aws events put-targets \
  --rule weekly-model-training \
  --targets "Id"="1","Arn"="arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:premium-model-training"

```

## Serverless Training Pipeline

```bash
#!/bin/bash
set -e

# Load environment variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=$(aws configure get region || echo "us-east-1")
export MODEL_ARTIFACTS_BUCKET=premium-model-artifacts
export MODEL_DATA_BUCKET=premium-model-data

echo "Setting up Free Tier deployment..."

# Step 1: Create S3 buckets if they don't exist
echo "Creating S3 buckets..."
aws s3api head-bucket --bucket $MODEL_ARTIFACTS_BUCKET 2>/dev/null || \
  aws s3 mb s3://$MODEL_ARTIFACTS_BUCKET
aws s3api head-bucket --bucket $MODEL_DATA_BUCKET 2>/dev/null || \
  aws s3 mb s3://$MODEL_DATA_BUCKET

# Step 2: Create ECR repository with lifecycle policy to stay under 0.5GB
echo "Creating ECR repository..."
aws ecr describe-repositories --repository-names premium-model-api 2>/dev/null || \
  aws ecr create-repository --repository-name premium-model-api

# Add lifecycle policy to keep only 3 latest images
aws ecr put-lifecycle-policy \
  --repository-name premium-model-api \
  --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"Keep only 3 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}'

# Step 3: Build and push a minimal Docker image
if command -v docker &> /dev/null; then
  echo "Building minimal Docker image..."
  # Create a minimal Dockerfile if it doesn't exist
  if [ ! -f Dockerfile ]; then
    cat > Dockerfile << EOF
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
EOF
  fi
  
  docker build -t premium-model-api:latest .
  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
  docker tag premium-model-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/premium-model-api:latest
  docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/premium-model-api:latest
else
  echo "Docker not installed, skipping image build"
fi

# Step 4: Create CloudWatch dashboard (Free Tier: 3 dashboards)
echo "Creating CloudWatch dashboard..."
cat > cloudwatch-dashboard.json << EOF
{
  "widgets": [
    {
      "type": "metric",
      "x": 0,
      "y": 0,
      "width": 24,
      "height": 6,
      "properties": {
        "metrics": [
          ["PremiumPredictionAPI", "model_prediction_latency"],
          ["PremiumPredictionAPI", "prediction_count"],
          ["PremiumPredictionAPI", "feature_drift"],
          ["PremiumPredictionAPI", "model_error"]
        ],
        "period": 3600,
        "stat": "Average",
        "region": "$AWS_REGION",
        "title": "Model Performance Metrics"
      }
    }
  ]
}
EOF

aws cloudwatch put-dashboard \
  --dashboard-name PremiumModelDashboard \
  --dashboard-body file://cloudwatch-dashboard.json

# Step 5: Create Lambda deployment package for inference
echo "Creating Lambda deployment package..."
mkdir -p lambda-package
cat > lambda-package/app.py << EOF
import json
import boto3
import os
import joblib

s3 = boto3.client('s3')
BUCKET = os.environ['MODEL_BUCKET']
MODEL_KEY = os.environ['MODEL_KEY']
model = None

def load_model():
    global model
    if model is None:
        # Download model from S3
        local_file = '/tmp/model.joblib'
        s3.download_file(BUCKET, MODEL_KEY, local_file)
        model = joblib.load(local_file)
    return model

def lambda_handler(event, context):
    # Load model if not already loaded
    model = load_model()
    
    # Parse input
    try:
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        features = body.get('features', [])
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
    
    # Make prediction
    try:
        prediction = model.predict([features])[0]
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': float(prediction)})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
EOF

# Create requirements.txt for Lambda
cat > lambda-package/requirements.txt << EOF
joblib==1.1.0
scikit-learn==1.0.2
numpy==1.22.3
EOF

echo "Free Tier deployment setup completed!"
echo ""
echo "Next steps:"
echo "1. Install dependencies in lambda-package: pip install -r requirements.txt -t lambda-package/"
echo "2. Create a zip file: cd lambda-package && zip -r ../lambda-deployment.zip ."
echo "3. Create Lambda function: aws lambda create-function --function-name premium-model-inference --runtime python3.8 --handler app.lambda_handler --zip-file fileb://lambda-deployment.zip --role <your-lambda-role-arn> --environment Variables={MODEL_BUCKET=$MODEL_ARTIFACTS_BUCKET,MODEL_KEY=models/latest/model.joblib}"
echo "4. Create API Gateway to expose your Lambda function"
echo ""
echo "For local testing, you can use AWS SAM CLI or run a local Flask server"


```

## Limitations

- Limited scalability: Free tier resources have performance constraints

- Manual deployment steps: Some automation is sacrificed for cost savings

- Cold starts: Lambda functions may experience latency on first invocation

- Limited monitoring: Reduced monitoring to stay within CloudWatch free tier

- This approach should allow us to deploy your ML workflow while staying largely within the AWS Free Tier limits. As your usage grows, you can gradually transition to more robust services as needed.
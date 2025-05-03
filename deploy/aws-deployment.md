# AWS Deployment Workflow for Premium Prediction ML Pipeline

This document outlines the complete workflow for deploying the Premium Prediction ML pipeline to AWS.

## Architecture Overview

The deployment architecture consists of:

1. **EKS Cluster** - For running containerized applications
2. **ECR Repositories** - For storing Docker images
3. **S3 Buckets** - For storing model artifacts and MLflow data
4. **RDS Database** - For MLflow backend (optional, for production)
5. **CloudWatch** - For monitoring and logging
6. **AWS CodePipeline** - For CI/CD automation

## Prerequisites

- AWS CLI installed and configured
- kubectl installed
- eksctl installed
- Docker installed

## Step 1: Set Up AWS Infrastructure

### Create ECR Repositories

```bash
# Set environment variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
export ECR_API_REPO=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/premium-model-api
export ECR_MLFLOW_REPO=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlflow-tracking

# Create repositories
aws ecr create-repository --repository-name premium-model-api
aws ecr create-repository --repository-name mlflow-tracking
```

### Create S3 Buckets

```bash
# Create buckets for model artifacts and MLflow data
aws s3 mb s3://premium-model-artifacts
aws s3 mb s3://premium-model-data

# Set bucket policies (optional)
aws s3api put-bucket-policy --bucket premium-model-artifacts --policy file://deploy/s3-bucket-policy.json
```

### Create EKS Cluster

```bash
# Create EKS cluster
eksctl create cluster \
  --name premium-prediction-cluster \
  --region $AWS_REGION \
  --version 1.27 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed
```

## Step 2: Configure AWS Services for ML Workflow

### Set Up IAM Roles

Create IAM roles for:
1. EKS cluster
2. MLflow to access S3
3. CodeBuild for CI/CD

```bash
# Create IAM policy for S3 access
aws iam create-policy \
  --policy-name MLflowS3Access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ],
        "Resource": [
          "arn:aws:s3:::premium-model-artifacts",
          "arn:aws:s3:::premium-model-artifacts/*",
          "arn:aws:s3:::premium-model-data",
          "arn:aws:s3:::premium-model-data/*"
        ]
      }
    ]
  }'

# Create service account for MLflow
eksctl create iamserviceaccount \
  --name mlflow-sa \
  --namespace mlops-premium \
  --cluster premium-prediction-cluster \
  --attach-policy-arn arn:aws:iam::$AWS_ACCOUNT_ID:policy/MLflowS3Access \
  --approve
```

## Step 3: Build and Push Docker Images

### Update Docker Images to Use AWS Services

```bash
# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push API image
docker build -t premium-model-api .
docker tag premium-model-api:latest $ECR_API_REPO:latest
docker push $ECR_API_REPO:latest

# Build and push MLflow image
docker pull ghcr.io/mlflow/mlflow:v2.5.0
docker tag ghcr.io/mlflow/mlflow:v2.5.0 $ECR_MLFLOW_REPO:latest
docker push $ECR_MLFLOW_REPO:latest
```

## Step 4: Update Kubernetes Configurations for AWS

### Create AWS-Specific ConfigMap

```bash
cat > kubernetes/aws-configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: premium-model-aws-config
  namespace: mlops-premium
data:
  MLFLOW_S3_ENDPOINT_URL: "https://s3.$AWS_REGION.amazonaws.com"
  MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
  ARTIFACT_ROOT: "s3://premium-model-artifacts"
  AWS_REGION: "$AWS_REGION"
EOF

kubectl apply -f kubernetes/aws-configmap.yaml
```

### Update Deployment Files

```bash
# Update image references in deployment files
sed -i "s|image: premium-model-api:latest|image: $ECR_API_REPO:latest|g" kubernetes/deployment.yaml
```

### Create MLflow Deployment for AWS

```bash
cat > kubernetes/mlflow-aws-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlops-premium
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      serviceAccountName: mlflow-sa
      containers:
      - name: mlflow
        image: $ECR_MLFLOW_REPO:latest
        ports:
        - containerPort: 5000
        command:
        - mlflow
        - server
        - --host=0.0.0.0
        - --port=5000
        - --backend-store-uri=sqlite:///mlruns.db
        - --default-artifact-root=s3://premium-model-artifacts
        env:
        - name: AWS_REGION
          valueFrom:
            configMapKeyRef:
              name: premium-model-aws-config
              key: AWS_REGION
        volumeMounts:
        - name: mlflow-data
          mountPath: /mlruns
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: mlops-premium
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
EOF

kubectl apply -f kubernetes/mlflow-aws-deployment.yaml
```

## Step 5: Deploy the Application to EKS

```bash
# Apply Kubernetes configurations
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/aws-configmap.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
kubectl apply -f kubernetes/monitoring.yaml
kubectl apply -f kubernetes/mlflow-aws-deployment.yaml
```

## Step 6: Set Up CI/CD Pipeline

### Create BuildSpec File for CodeBuild

```bash
cat > buildspec.yml << EOF
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.9
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
      - COMMIT_HASH=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c 1-7)
      - IMAGE_TAG=\${COMMIT_HASH:=latest}
      - echo Running tests...
      - pip install -r requirements.txt
      - pytest tests/
  build:
    commands:
      - echo Build started on `date`
      - echo Building the Docker image...
      - docker build -t $ECR_API_REPO:\$IMAGE_TAG .
      - docker tag $ECR_API_REPO:\$IMAGE_TAG $ECR_API_REPO:latest
  post_build:
    commands:
      - echo Build completed on `date`
      - echo Pushing the Docker image...
      - docker push $ECR_API_REPO:\$IMAGE_TAG
      - docker push $ECR_API_REPO:latest
      - echo Writing image definitions file...
      - aws eks update-kubeconfig --name premium-prediction-cluster --region $AWS_REGION
      - kubectl set image deployment/premium-model-api premium-model-api=$ECR_API_REPO:\$IMAGE_TAG -n mlops-premium

artifacts:
  files:
    - kubernetes/**/*
    - appspec.yml
    - buildspec.yml
EOF
```

### Create CodePipeline

```bash
# Create CodePipeline using AWS Console or CLI
aws codepipeline create-pipeline --cli-input-json file://deploy/codepipeline.json
```

## Step 7: Set Up Model Training Workflow

### Create Training Job Definition

```bash
cat > deploy/training-job.json << EOF
{
  "jobDefinitionName": "premium-model-training",
  "type": "container",
  "containerProperties": {
    "image": "$ECR_API_REPO:latest",
    "vcpus": 4,
    "memory": 8192,
    "command": ["python", "-m", "src.train"],
    "environment": [
      {"name": "LOG_LEVEL", "value": "info"},
      {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow-service:5000"},
      {"name": "MLFLOW_S3_ENDPOINT_URL", "value": "https://s3.$AWS_REGION.amazonaws.com"},
      {"name": "ARTIFACT_ROOT", "value": "s3://premium-model-artifacts"}
    ]
  }
}
EOF

aws batch register-job-definition --cli-input-json file://deploy/training-job.json
```

### Create Training Job Submission Script

```bash
cat > deploy/submit-training-job.sh << EOF
#!/bin/bash

# Submit training job to AWS Batch
aws batch submit-job \
  --job-name premium-training-job-\$(date +%Y%m%d-%H%M%S) \
  --job-queue premium-training-queue \
  --job-definition premium-model-training
EOF

chmod +x deploy/submit-training-job.sh
```

## Step 8: Set Up Monitoring and Alerting

### Create CloudWatch Dashboard

```bash
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
          ["PremiumPredictionAPI", "model_prediction_latency", "ModelVersion", "latest"]
        ],
        "period": 300,
        "stat": "Average",
        "region": "$AWS_REGION",
        "title": "Model Prediction Latency"
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["PremiumPredictionAPI", "prediction_count"]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "$AWS_REGION",
        "title": "Prediction Count"
      }
    },
    {
      "type": "metric",
      "x": 0,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["PremiumPredictionAPI", "feature_drift"]
        ],
        "period": 3600,
        "stat": "Maximum",
        "region": "$AWS_REGION",
        "title": "Feature Drift"
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["PremiumPredictionAPI", "model_error"]
        ],
        "period": 3600,
        "stat": "Average",
        "region": "$AWS_REGION",
        "title": "Model Error"
      }
    }
  ]
}
EOF

aws cloudwatch put-dashboard \
  --dashboard-name PremiumModelDashboard \
  --dashboard-body file://deploy/cloudwatch-dashboard.json
```

### Create CloudWatch Alarms

```bash
# Create alarm for high model error
aws cloudwatch put-metric-alarm \
  --alarm-name HighModelError \
  --alarm-description "Alarm when model error exceeds threshold" \
  --metric-name model_error \
  --namespace PremiumPredictionAPI \
  --statistic Average \
  --period 3600 \
  --threshold 0.15 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions arn:aws:sns:$AWS_REGION:$AWS_ACCOUNT_ID:model-alerts

# Create alarm for feature drift
aws cloudwatch put-metric-alarm \
  --alarm-name FeatureDrift \
  --alarm-description "Alarm when feature drift exceeds threshold" \
  --metric-name feature_drift \
  --namespace PremiumPredictionAPI \
  --statistic Maximum \
  --period 3600 \
  --threshold 0.1 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions arn:aws:sns:$AWS_REGION:$AWS_ACCOUNT_ID:model-alerts
```

## Step 9: Create AWS Lambda for Automated Model Retraining

```bash
cat > deploy/lambda_function.py << EOF
import boto3
import json
import os

def lambda_handler(event, context):
    # Parse CloudWatch alarm event
    alarm_name = event['detail']['alarmName']
    
    # If this is a model drift alarm, trigger retraining
    if alarm_name in ['FeatureDrift', 'HighModelError']:
        batch = boto3.client('batch')
        
        # Submit training job
        response = batch.submit_job(
            jobName='premium-retraining-' + context.aws_request_id,
            jobQueue='premium-training-queue',
            jobDefinition='premium-model-training'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(f"Triggered model retraining with job ID: {response['jobId']}")
        }
    
    return {
        'statusCode': 200,
        'body': json.dumps('Event processed but no action taken')
    }
EOF

# Create Lambda function (using AWS Console or CLI)
```

## Step 10: Create Deployment Scripts

### Create Main Deployment Script

```bash
cat > deploy/deploy-to-aws.sh << EOF
#!/bin/bash
set -e

# Load environment variables
source deploy/env.sh

# Step 1: Create ECR repositories if they don't exist
echo "Creating ECR repositories..."
aws ecr describe-repositories --repository-names premium-model-api || aws ecr create-repository --repository-name premium-model-api
aws ecr describe-repositories --repository-names mlflow-tracking || aws ecr create-repository --repository-name mlflow-tracking

# Step 2: Build and push Docker images
echo "Building and pushing Docker images..."
aws ecr get-login-password --region \$AWS_REGION | docker login --username AWS --password-stdin \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_REGION.amazonaws.com
docker build -t premium-model-api .
docker tag premium-model-api:latest \$ECR_API_REPO:latest
docker push \$ECR_API_REPO:latest

docker pull ghcr.io/mlflow/mlflow:v2.5.0
docker tag ghcr.io/mlflow/mlflow:v2.5.0 \$ECR_MLFLOW_REPO:latest
docker push \$ECR_MLFLOW_REPO:latest

# Step 3: Create S3 buckets if they don't exist
echo "Creating S3 buckets..."
aws s3api head-bucket --bucket premium-model-artifacts 2>/dev/null || aws s3 mb s3://premium-model-artifacts
aws s3api head-bucket --bucket premium-model-data 2>/dev/null || aws s3 mb s3://premium-model-data

# Step 4: Create or update EKS cluster
echo "Setting up EKS cluster..."
eksctl get cluster --name premium-prediction-cluster 2>/dev/null || \
eksctl create cluster \
  --name premium-prediction-cluster \
  --region \$AWS_REGION \
  --version 1.27 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed

# Step 5: Update Kubernetes configurations
echo "Updating Kubernetes configurations..."
sed -i "s|image: premium-model-api:latest|image: \$ECR_API_REPO:latest|g" kubernetes/deployment.yaml

# Step 6: Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/aws-configmap.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
kubectl apply -f kubernetes/monitoring.yaml
kubectl apply -f kubernetes/mlflow-aws-deployment.yaml

# Step 7: Set up CloudWatch dashboard and alarms
echo "Setting up CloudWatch monitoring..."
aws cloudwatch put-dashboard \
  --dashboard-name PremiumModelDashboard \
  --dashboard-body file://deploy/cloudwatch-dashboard.json

echo "Deployment completed successfully!"
EOF

chmod +x deploy/deploy-to-aws.sh
```

### Create Environment Variables File

```bash
cat > deploy/env.sh << EOF
#!/bin/bash

# AWS account and region
export AWS_ACCOUNT_ID=\$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# ECR repositories
export ECR_API_REPO=\$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_REGION.amazonaws.com/premium-model-api
export ECR_MLFLOW_REPO=\$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_REGION.amazonaws.com/mlflow-tracking

# EKS cluster name
export EKS_CLUSTER_NAME=premium-prediction-cluster

# S3 buckets
export MODEL_ARTIFACTS_BUCKET=premium-model-artifacts
export MODEL_DATA_BUCKET=premium-model-data
EOF

chmod +x deploy/env.sh
```

## Step 11: Create AWS CodePipeline Configuration

```bash
cat > deploy/codepipeline.json << EOF
{
  "pipeline": {
    "name": "premium-model-pipeline",
    "roleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/service-role/AWSCodePipelineServiceRole-premium-model",
    "artifactStore": {
      "type": "S3",
      "location": "premium-model-pipeline-artifacts"
    },
    "stages": [
      {
        "name": "Source",
        "actions": [
          {
            "name": "Source",
            "actionTypeId": {
              "category": "Source",
              "owner": "AWS",
              "provider": "CodeStarSourceConnection",
              "version": "1"
            },
            "configuration": {
              "ConnectionArn": "arn:aws:codestar-connections:$AWS_REGION:$AWS_ACCOUNT_ID:connection/your-connection-id",
              "FullRepositoryId": "your-github-username/E2EMLOPsregression",
              "BranchName": "main"
            },
            "outputArtifacts": [
              {
                "name": "SourceCode"
              }
            ]
          }
        ]
      },
      {
        "name": "Build",
        "actions": [
          {
            "name": "BuildAndTest",
            "actionTypeId": {
              "category": "Build",
              "owner": "AWS",
              "provider": "CodeBuild",
              "version": "1"
            },
            "configuration": {
              "ProjectName": "premium-model-build"
            },
            "inputArtifacts": [
              {
                "name": "SourceCode"
              }
            ],
            "outputArtifacts": [
              {
                "name": "BuildOutput"
              }
            ]
          }
        ]
      },
      {
        "name": "Deploy",
        "actions": [
          {
            "name": "DeployToEKS",
            "actionTypeId": {
              "category": "Deploy",
              "owner": "AWS",
              "provider": "ServiceCatalog",
              "version": "1"
            },
            "configuration": {
              "ClusterName": "premium-prediction-cluster",
              "ManifestPath": "BuildOutput::kubernetes",
              "Namespace": "mlops-premium"
            },
            "inputArtifacts": [
              {
                "name": "BuildOutput"
              }
            ]
          }
        ]
      }
    ]
  }
}
EOF
```

## Usage Instructions

1. **Initial Deployment**:
   ```bash
   # Set up environment variables
   source deploy/env.sh
   
   # Run the deployment script
   ./deploy/deploy-to-aws.sh
   ```

2. **Training a Model**:
   ```bash
   # Submit a training job
   ./deploy/submit-training-job.sh
   ```

3. **Accessing MLflow UI**:
   ```bash
   # Port forward MLflow service
   kubectl port-forward svc/mlflow-service 5000:5000 -n mlops-premium
   
   # Access MLflow UI at http://localhost:5000
   ```

4. **Accessing the API**:
   ```bash
   # Get the API endpoint
   kubectl get svc -n mlops-premium
   
   # For local testing, port forward
   kubectl port-forward svc/premium-model-api 8000:8000 -n mlops-premium
   ```

5. **Monitoring**:
   - Access CloudWatch dashboard in AWS Console
   - View logs in CloudWatch Logs
   - Check model metrics in MLflow

## Troubleshooting

- **Pod Startup Issues**: Check pod logs with `kubectl logs <pod-name> -n mlops-premium`
- **S3 Access Issues**: Verify IAM roles and policies
- **Model Training Failures**: Check CloudWatch logs for training jobs
- **API Errors**: Check API logs and ensure model files are correctly loaded from S3
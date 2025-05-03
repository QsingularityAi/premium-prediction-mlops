# AWS Deployment Guide for Premium Prediction ML Pipeline

This guide outlines the complete workflow for deploying the Premium Prediction ML pipeline to AWS, incorporating lessons learned from our deployment experiences.

## Architecture Overview

The deployment architecture consists of:

1. **EKS Cluster** - For running containerized applications
2. **ECR Repositories** - For storing Docker images
3. **S3 Buckets** - For storing model artifacts and MLflow data
4. **CloudWatch** - For monitoring and logging
5. **AWS CodePipeline/CodeBuild** - For CI/CD automation

## Prerequisites

- AWS CLI installed and configured
- kubectl installed
- eksctl installed
- Docker installed

## Step 1: Set Up Environment Variables

Create a central environment variables file to maintain consistency:

```bash
cat > deploy/env.sh << EOF
#!/bin/bash

# AWS account and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# ECR repositories
export ECR_API_REPO=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/premium-model-api
export ECR_MLFLOW_REPO=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlflow-tracking

# EKS cluster name
export EKS_CLUSTER_NAME=premium-prediction-cluster

# S3 buckets
export MODEL_ARTIFACTS_BUCKET=premium-model-artifacts
export MODEL_DATA_BUCKET=premium-model-data
EOF

chmod +x deploy/env.sh
source deploy/env.sh
```

## Step 2: Set Up AWS Infrastructure

### Create ECR Repositories

```bash
# Create repositories if they don't exist
aws ecr describe-repositories --repository-names premium-model-api 2>/dev/null || \
  aws ecr create-repository --repository-name premium-model-api

aws ecr describe-repositories --repository-names mlflow-tracking 2>/dev/null || \
  aws ecr create-repository --repository-name mlflow-tracking
```

### Create S3 Buckets

```bash
# Create buckets if they don't exist
aws s3api head-bucket --bucket $MODEL_ARTIFACTS_BUCKET 2>/dev/null || \
  aws s3 mb s3://$MODEL_ARTIFACTS_BUCKET

aws s3api head-bucket --bucket $MODEL_DATA_BUCKET 2>/dev/null || \
  aws s3 mb s3://$MODEL_DATA_BUCKET
```

## Step 3: Create EKS Cluster

```bash
# Create EKS cluster if it doesn't exist
eksctl get cluster --name $EKS_CLUSTER_NAME 2>/dev/null || \
eksctl create cluster \
  --name $EKS_CLUSTER_NAME \
  --region $AWS_REGION \
  --version 1.27 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed \
  --with-oidc  # Important for IAM roles for service accounts
```

## Step 4: Configure IAM and OIDC

```bash
# Associate IAM OIDC provider with the cluster
eksctl utils associate-iam-oidc-provider \
  --region=$AWS_REGION \
  --cluster=$EKS_CLUSTER_NAME \
  --approve

# Create IAM policy for S3 access
POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='MLflowS3Access'].Arn" --output text)

if [ -z "$POLICY_ARN" ]; then
  echo "Creating IAM policy for S3 access..."
  POLICY_ARN=$(aws iam create-policy \
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
            "arn:aws:s3:::'$MODEL_ARTIFACTS_BUCKET'",
            "arn:aws:s3:::'$MODEL_ARTIFACTS_BUCKET'/*",
            "arn:aws:s3:::'$MODEL_DATA_BUCKET'",
            "arn:aws:s3:::'$MODEL_DATA_BUCKET'/*"
          ]
        }
      ]
    }' --query 'Policy.Arn' --output text)
fi

# Create namespace
kubectl create namespace mlops-premium --dry-run=client -o yaml | kubectl apply -f -

# Create service account for MLflow
eksctl create iamserviceaccount \
  --name mlflow-sa \
  --namespace mlops-premium \
  --cluster $EKS_CLUSTER_NAME \
  --attach-policy-arn $POLICY_ARN \
  --approve \
  --override-existing-serviceaccounts
```

## Step 5: Build and Push Docker Images

```bash
# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running. Please start Docker and try again."
  echo "You can also run this script with --skip-docker to skip Docker steps."
  exit 1
fi

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build for the correct platform (if on M1/M2 Mac)
docker buildx build --platform linux/amd64 -t premium-model-api .
docker tag premium-model-api:latest $ECR_API_REPO:latest
docker push $ECR_API_REPO:latest

# Pull and push MLflow image
docker pull ghcr.io/mlflow/mlflow:v2.5.0
docker tag ghcr.io/mlflow/mlflow:v2.5.0 $ECR_MLFLOW_REPO:latest
docker push $ECR_MLFLOW_REPO:latest
```

## Step 6: Create Kubernetes Configurations

### Create ConfigMap

```bash
# Create AWS-specific ConfigMap
cat > kubernetes/aws-configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: premium-model-aws-config
  namespace: mlops-premium
data:
  MLFLOW_S3_ENDPOINT_URL: "https://s3.$AWS_REGION.amazonaws.com"
  MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
  ARTIFACT_ROOT: "s3://$MODEL_ARTIFACTS_BUCKET"
  AWS_REGION: "$AWS_REGION"
EOF

kubectl apply -f kubernetes/aws-configmap.yaml
```

### Create Secrets

```bash
# Create secrets for AWS credentials and Grafana
kubectl create secret generic premium-model-secrets -n mlops-premium \
  --from-literal=GRAFANA_ADMIN_PASSWORD=admin \
  --from-literal=AWS_ACCESS_KEY_ID=your-access-key \
  --from-literal=AWS_SECRET_ACCESS_KEY=your-secret-key \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Step 7: Deploy MLflow

```bash
# Create MLflow deployment with emptyDir instead of PVC
cat > kubernetes/mlflow-deployment.yaml << EOF
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
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.5.0
        ports:
        - containerPort: 5000
        command:
        - mlflow
        - server
        - --host=0.0.0.0
        - --port=5000
        - --backend-store-uri=sqlite:///tmp/mlruns.db
        env:
        - name: AWS_REGION
          valueFrom:
            configMapKeyRef:
              name: premium-model-aws-config
              key: AWS_REGION
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: premium-model-secrets
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: premium-model-secrets
              key: AWS_SECRET_ACCESS_KEY
        volumeMounts:
        - name: mlflow-data
          mountPath: /tmp
      volumes:
      - name: mlflow-data
        emptyDir: {}
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

kubectl apply -f kubernetes/mlflow-deployment.yaml
```

## Step 8: Deploy API Service

```bash
# Create API deployment with emptyDir instead of PVC
cat > kubernetes/api-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: premium-model-api
  namespace: mlops-premium
spec:
  replicas: 1
  selector:
    matchLabels:
      app: premium-model-api
  template:
    metadata:
      labels:
        app: premium-model-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: premium-model-api
        image: $ECR_API_REPO:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: premium-model-aws-config
              key: MLFLOW_TRACKING_URI
        - name: AWS_REGION
          valueFrom:
            configMapKeyRef:
              name: premium-model-aws-config
              key: AWS_REGION
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: premium-model-secrets
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: premium-model-secrets
              key: AWS_SECRET_ACCESS_KEY
        volumeMounts:
        - name: model-data
          mountPath: /app/models
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: model-data
        emptyDir: {}
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: premium-model-api-service
  namespace: mlops-premium
spec:
  selector:
    app: premium-model-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
EOF

kubectl apply -f kubernetes/api-deployment.yaml
```

## Step 9: Deploy Monitoring

```bash
# Create simplified monitoring deployment
cat > kubernetes/monitoring-simple.yaml << EOF
---
# Prometheus Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: mlops-premium
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'premium-model-api'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: (\\d+)
            replacement: $1
            target_label: __address__
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: kubernetes_pod_name
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: mlops-premium
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.41.0
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
        - name: prometheus-data
          mountPath: /prometheus
        args:
        - "--config.file=/etc/prometheus/prometheus.yml"
        - "--storage.tsdb.path=/prometheus"
        - "--web.console.libraries=/usr/share/prometheus/console_libraries"
        - "--web.console.templates=/usr/share/prometheus/consoles"
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
      - name: prometheus-data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: mlops-premium
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF

kubectl apply -f kubernetes/monitoring-simple.yaml
```

## Step 10: Set Up CI/CD with CodeBuild

### Create CodeBuild Role

```bash
# Create CodeBuild service role
ROLE_NAME="premium-model-codebuild-role"
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ]; then
  echo "Creating CodeBuild service role..."
  
  # Create trust policy
  cat > trust-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
POLICY

  # Create role
  ROLE_ARN=$(aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json --query 'Role.Arn' --output text)
  
  # Create policy document
  cat > codebuild-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetBucketAcl",
        "s3:GetBucketLocation"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "codebuild:CreateReportGroup",
        "codebuild:CreateReport",
        "codebuild:UpdateReport",
        "codebuild:BatchPutTestCases",
        "codebuild:BatchPutCodeCoverages"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "eks:DescribeCluster",
        "eks:ListClusters"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:PutParameter"
      ],
      "Resource": "*"
    }
  ]
}
POLICY

  # Attach policy
  aws iam put-role-policy --role-name $ROLE_NAME --policy-name codebuild-base-policy --policy-document file://codebuild-policy.json
  
  echo "Created role: $ROLE_ARN"
fi
```

### Create CodeBuild Project

```bash
# Store AWS account ID in Parameter Store
aws ssm put-parameter --name "/premium-model/aws-account-id" --value "$AWS_ACCOUNT_ID" --type String --overwrite

# Create buildspec file
cat > buildspec.yml << EOF
version: 0.2

env:
  variables:
    AWS_REGION: "$AWS_REGION"
    MLFLOW_TRACKING_URI: "http://localhost:5000"
  parameter-store:
    AWS_ACCOUNT_ID: "/premium-model/aws-account-id"

phases:
  install:
    runtime-versions:
      python: 3.9
    commands:
      - echo Installing dependencies...
      - pip install -r requirements.txt
      - pip install mlflow==2.5.0
      
  pre_build:
    commands:
      - echo Starting MLflow server...
      - mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db &
      - sleep 5  # Give MLflow time to start
      - echo Running tests...
      - pytest tests/ -v
      
  build:
    commands:
      - echo Build started on `date`
      - echo Training model...
      - bash deploy/train-model.sh
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
      - COMMIT_HASH=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c 1-7)
      - IMAGE_TAG=\${COMMIT_HASH:=latest}
      - ECR_API_REPO=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/premium-model-api
      - echo Building the Docker image...
      - docker build -t $ECR_API_REPO:$IMAGE_TAG .
      - docker tag $ECR_API_REPO:$IMAGE_TAG $ECR_API_REPO:latest
      
  post_build:
    commands:
      - echo Build completed on `date`
      - echo Pushing the Docker image...
      - docker push $ECR_API_REPO:$IMAGE_TAG
      - docker push $ECR_API_REPO:latest
      - echo Deploying to EKS...
      - aws eks update-kubeconfig --name premium-prediction-cluster --region $AWS_REGION
      - sed -i "s|image: premium-model-api:latest|image: $ECR_API_REPO:$IMAGE_TAG|g" kubernetes/deployment.yaml
      - kubectl apply -f kubernetes/deployment.yaml -n mlops-premium
      - echo "Updating model version tag in Kubernetes deployment..."
      - kubectl set image deployment/premium-model-api premium-model-api=$ECR_API_REPO:$IMAGE_TAG -n mlops-premium

artifacts:
  files:
    - kubernetes/**/*
    - appspec.yml
    - buildspec.yml
    - deploy/**/*
    - models/**/*
    - mlruns/**/*
  discard-paths: no

cache:
  paths:
    - '/root/.cache/pip/**/*'
EOF

# Create CodeBuild project
aws codebuild create-project \
  --name premium-model-build \
  --source "{\"type\": \"GITHUB\", \"location\": \"https://github.com/yourusername/E2EMLOPsregression.git\"}" \
  --artifacts "{\"type\": \"NO_ARTIFACTS\"}" \
  --environment "{\"type\": \"LINUX_CONTAINER\", \"image\": \"aws/codebuild/amazonlinux2-x86_64-standard:3.0\", \"computeType\": \"BUILD_GENERAL1_SMALL\", \"privilegedMode\": true}" \
  --service-role "$ROLE_ARN"
```

## Step 11: Set Up CloudWatch Monitoring

### Create CloudWatch Dashboard

```bash
# Create CloudWatch dashboard with explicit region
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

# Replace $AWS_REGION with actual value in the JSON file
sed -i '' "s|\$AWS_REGION|$AWS_REGION|g" deploy/cloudwatch-dashboard.json

aws cloudwatch put-dashboard \
  --dashboard-name PremiumModelDashboard \
  --dashboard-body file://deploy/cloudwatch-dashboard.json
```

### Create CloudWatch Alarms

```bash
# Create SNS topic for alerts
aws sns create-topic --name model-alerts

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

## Step 12: Create Automated Model Training

### Create Training Script

```bash
cat > deploy/train-model.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting model training..."

# Set environment variables
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
export MODEL_DIR=${MODEL_DIR:-models}
export DATA_DIR=${DATA_DIR:-data}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Create directories if they don't exist
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR/processed

# Check if data exists, if not download sample data
if [ ! -f "$DATA_DIR/raw/premium_data.csv" ]; then
  echo "Downloading sample data..."
  mkdir -p $DATA_DIR/raw
  # This is a placeholder - replace with your actual data download command
  # For example: aws s3 cp s3://your-bucket/premium_data.csv $DATA_DIR/raw/
  # For now, we'll create a simple CSV file for testing
  echo "id,age,gender,bmi,smoker,region,premium" > $DATA_DIR/raw/premium_data.csv
  echo "1,25,male,22.5,no,northeast,5000" >> $DATA_DIR/raw/premium_data.csv
  echo "2,42,female,28.1,yes,southwest,12500" >> $DATA_DIR/raw/premium_data.csv
  echo "3,38,male,24.3,no,southeast,7500" >> $DATA_DIR/raw/premium_data.csv
fi

# Run training script
echo "Running training script..."
python -m src.train --register-model --log-artifacts

echo "Model training completed successfully!"
EOF

chmod +x deploy/train-model.sh
```

## Step 13: Create Main Deployment Script

```bash
cat > deploy/deploy-to-aws.sh << 'EOF'
#!/bin/bash
set -e

# Load environment variables
source deploy/env.sh

# Parse command line arguments
SKIP_DOCKER=false
SKIP_EKS=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-docker) SKIP_DOCKER=true ;;
        --skip-eks) SKIP_EKS=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Step 1: Create ECR repositories if they don't exist
echo "Creating ECR repositories..."
aws ecr describe-repositories --repository-names premium-model-api 2>/dev/null || aws ecr create-repository --repository-name premium-model-api
aws ecr describe-repositories --repository-names mlflow-tracking 2>/dev/null || aws ecr create-repository --repository-name mlflow-tracking

# Step 2: Build and push Docker images
if [ "$SKIP_DOCKER" = false ]; then
    echo "Building and pushing Docker images..."
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running. Please start Docker and try again."
        echo "You can also run this script with --skip-docker to skip Docker steps."
        exit 1
    fi
    
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    docker build -t premium-model-api .
    docker tag premium-model-api:latest $ECR_API_REPO:latest
    docker push $ECR_API_REPO:latest

    docker pull ghcr.io/mlflow/mlflow:v2.5.0
    docker tag ghcr.io/mlflow/mlflow:v2.5.0 $ECR_MLFLOW_REPO:latest
    docker push $ECR_MLFLOW_REPO:latest
else
    echo "Skipping Docker build and push steps..."
fi

# Step 3: Create S3 buckets if they don't exist
echo "Creating S3 buckets..."
aws s3api head-bucket --bucket $MODEL_ARTIFACTS_BUCKET 2>/dev/null || aws s3 mb s3://$MODEL_ARTIFACTS_BUCKET
aws s3api head-bucket --bucket $MODEL_DATA_BUCKET 2>/dev/null || aws s3 mb s3://$MODEL_DATA_BUCKET

# Step 4: Create or update EKS cluster
if [ "$SKIP_EKS" = false ]; then
    echo "Setting up EKS cluster..."
    # Check if eksctl is installed
    if ! command -v eksctl &> /dev/null; then
        echo "Error: eksctl is not installed. Please install it and try again."
        echo "You can also run this script with --skip-eks to skip EKS steps."
        exit 1
    fi
    
    eksctl get cluster --name $EKS_CLUSTER_NAME 2>/dev/null || \
    eksctl create cluster \
      --name $EKS_CLUSTER_NAME \
      --region $AWS_REGION \
      --version 1.27 \
      --nodegroup-name standard-workers \
      --node-type t3.medium \
      --nodes 2 \
      --nodes-min 1 \
      --nodes-max 4 \
      --managed \
      --with-oidc

    # Step 5: Associate IAM OIDC provider with the cluster
    echo "Associating IAM OIDC provider with the cluster..."
    eksctl utils associate-iam-oidc-provider --region=$AWS_REGION --cluster=$EKS_CLUSTER_NAME --approve

    # Step 6: Create IAM policy and service account for MLflow
    echo "Setting up IAM roles for MLflow..."
    POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='MLflowS3Access'].Arn" --output text)

    if [ -z "$POLICY_ARN" ]; then
      echo "Creating IAM policy for S3 access..."
      POLICY_ARN=$(aws iam create-policy \
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
                "arn:aws:s3:::'$MODEL_ARTIFACTS_BUCKET'",
                "arn:aws:s3:::'$MODEL_ARTIFACTS_BUCKET'/*",
                "arn:aws:s3:::'$MODEL_DATA_BUCKET'",
                "arn:aws:s3:::'$MODEL_DATA_BUCKET'/*"
              ]
            }
          ]
        }' --query 'Policy.Arn' --output text)
    fi

    # Create namespace
    echo "Creating namespace..."
    kubectl create namespace mlops-premium --dry-run=client -o yaml | kubectl apply -f -

    echo "Creating service account for MLflow..."
    eksctl create iamserviceaccount \
      --name mlflow-sa \
      --namespace mlops-premium \
      --cluster $EKS_CLUSTER_NAME \
      --attach-policy-arn $POLICY_ARN \
      --approve \
      --override-existing-serviceaccounts

    # Step 7: Create AWS-specific ConfigMap
    echo "Creating AWS ConfigMap..."
    cat > kubernetes/aws-configmap.yaml << EOF2
apiVersion: v1
kind: ConfigMap
metadata:
  name: premium-model-aws-config
  namespace: mlops-premium
data:
  MLFLOW_S3_ENDPOINT_URL: "https://s3.$AWS_REGION.amazonaws.com"
  MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
  ARTIFACT_ROOT: "s3://$MODEL_ARTIFACTS_BUCKET"
  AWS_REGION: "$AWS_REGION"
EOF2

    # Step 8: Create MLflow deployment
    echo "Creating MLflow deployment..."
    cat > kubernetes/mlflow-deployment.yaml << EOF2
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
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.5.0
        ports:
        - containerPort: 5000
        command:
        - mlflow
        - server
        - --host=0.0.0.0
        - --port=5000
        - --backend-store-uri=sqlite:///tmp/mlruns.db
        env:
        - name: AWS_REGION
          valueFrom:
            configMapKeyRef:
              name: premium-model-aws-config
              key: AWS_REGION
        volumeMounts:
        - name: mlflow-data
          mountPath: /tmp
      volumes:
      - name: mlflow-data
        emptyDir: {}
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
EOF2

    # Step 9: Create API deployment
    echo "Creating API deployment..."
    cat > kubernetes/api-deployment.yaml << EOF2
apiVersion: apps/v1
kind: Deployment
metadata:
  name: premium-model-api
  namespace: mlops-premium
spec:
  replicas: 1
  selector:
    matchLabels:
      app: premium-model-api
  template:
    metadata:
      labels:
        app: premium-model-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: premium-model-api
        image: $ECR_API_REPO:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: premium-model-aws-config
              key: MLFLOW_TRACKING_URI
        - name: AWS_REGION
          valueFrom:
            configMapKeyRef:
              name: premium-model-aws-config
              key: AWS_REGION
        volumeMounts:
        - name: model-data
          mountPath: /app/models
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: model-data
        emptyDir: {}
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: premium-model-api-service
  namespace: mlops-premium
spec:
  selector:
    app: premium-model-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
EOF2

    # Step 10: Create secrets
    echo "Creating secrets..."
    kubectl create secret generic premium-model-secrets -n mlops-premium \
      --from-literal=GRAFANA_ADMIN_PASSWORD=admin \
      --from-literal=AWS_ACCESS_KEY_ID=your-access-key \
      --from-literal=AWS_SECRET_ACCESS_KEY=your-secret-key \
      --dry-run=client -o yaml | kubectl apply -f -

    # Step 11: Apply Kubernetes configurations
    echo "Applying Kubernetes configurations..."
    kubectl apply -f kubernetes/aws-configmap.yaml
    kubectl apply -f kubernetes/mlflow-deployment.yaml
    kubectl apply -f kubernetes/api-deployment.yaml
else
    echo "Skipping EKS setup steps..."
fi

# Step 12: Set up CloudWatch dashboard
echo "Setting up CloudWatch dashboard..."
cat > deploy/cloudwatch-dashboard.json << EOF2
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
EOF2

# Replace $AWS_REGION with actual value in the JSON file
sed -i '' "s|\$AWS_REGION|$AWS_REGION|g" deploy/cloudwatch-dashboard.json

aws cloudwatch put-dashboard \
  --dashboard-name PremiumModelDashboard \
  --dashboard-body file://deploy/cloudwatch-dashboard.json

echo "Deployment completed successfully!"
echo ""
echo "Next steps:"
echo "1. If you skipped Docker steps, make sure to build and push your Docker images"
echo "2. If you skipped EKS steps, set up your EKS cluster manually"
echo ""
echo "Access your MLflow UI by running: kubectl port-forward svc/mlflow-service 5000:5000 -n mlops-premium"
echo "Access your API by running: kubectl port-forward svc/premium-model-api-service 8000:80 -n mlops-premium"
EOF

chmod +x deploy/deploy-to-aws.sh
```

## Step 14: Create Helper Scripts

### Create Script to Trigger Builds

```bash
cat > deploy/trigger-build.sh << 'EOF'
#!/bin/bash
set -e

# Load environment variables
if [ -f deploy/env.sh ]; then
  source deploy/env.sh
else
  # Set default values if env.sh doesn't exist
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  AWS_REGION=$(aws configure get region)
fi

echo "Using AWS Account: $AWS_ACCOUNT_ID"
echo "Using AWS Region: $AWS_REGION"

# Check if the CodeBuild project exists
PROJECT_EXISTS=$(aws codebuild list-projects --query "projects[?contains(@, 'premium-model-build')]" --output text)

if [ -z "$PROJECT_EXISTS" ]; then
  echo "Error: CodeBuild project 'premium-model-build' does not exist."
  echo "Please run ./deploy/create-codebuild.sh first."
  exit 1
fi

# Start a CodeBuild build
echo "Triggering CodeBuild project..."
aws codebuild start-build \
  --project-name premium-model-build \
  --environment-variables-override name=TRAIN_MODEL,value=true,type=PLAINTEXT

echo "Build triggered successfully!"
echo "Check the build status in the AWS CodeBuild console."
EOF

chmod +x deploy/trigger-build.sh
```

### Create Script to Upload Source Code to S3

```bash
cat > deploy/upload-source.sh << 'EOF'
#!/bin/bash
set -e

# Load environment variables
if [ -f deploy/env.sh ]; then
  source deploy/env.sh
else
  # Set default values if env.sh doesn't exist
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  AWS_REGION=$(aws configure get region)
fi

BUCKET_NAME="premium-model-pipeline-artifacts-$AWS_ACCOUNT_ID"

# Check if bucket exists, create if not
aws s3api head-bucket --bucket $BUCKET_NAME 2>/dev/null || \
  aws s3 mb s3://$BUCKET_NAME

echo "Creating source.zip file..."
zip -r source.zip . -x "*.git*" -x "venv/*" -x "*.zip" -x "*.pyc" -x "__pycache__/*"

echo "Uploading to S3 bucket: $BUCKET_NAME"
aws s3 cp source.zip s3://$BUCKET_NAME/

echo "Source code uploaded successfully!"
echo "The pipeline should start automatically if configured."
EOF

chmod +x deploy/upload-source.sh
```

## Step 15: Usage Instructions

### Initial Deployment

```bash
# Set up environment variables
source deploy/env.sh

# Run the deployment script
./deploy/deploy-to-aws.sh
```

If Docker is not running or you want to skip certain steps:

```bash
# Skip Docker steps
./deploy/deploy-to-aws.sh --skip-docker

# Skip EKS steps
./deploy/deploy-to-aws.sh --skip-eks

# Skip both
./deploy/deploy-to-aws.sh --skip-docker --skip-eks
```

### Training a Model

```bash
# Submit a training job
./deploy/trigger-build.sh
```

### Accessing Services

```bash
# Access MLflow UI
kubectl port-forward svc/mlflow-service 5000:5000 -n mlops-premium
# Then open http://localhost:5000 in your browser

# Access API
kubectl port-forward svc/premium-model-api-service 8000:80 -n mlops-premium
# Then access http://localhost:8000 in your browser

# Access Grafana (if deployed)
kubectl port-forward svc/grafana-service 3000:3000 -n mlops-premium
# Then access http://localhost:3000 in your browser
```

## Step 16: Troubleshooting

### Common Issues and Solutions

1. **Docker Not Running**
   - Error: "Cannot connect to the Docker daemon"
   - Solution: Start Docker Desktop or use `--skip-docker` flag

2. **EKS Cluster Creation Fails**
   - Error: "Error creating EKS cluster"
   - Solution: Check AWS credentials and permissions

3. **PVC Issues**
   - Error: "PersistentVolumeClaim not found" or "pod has unbound immediate PersistentVolumeClaims"
   - Solution: Use emptyDir volumes instead of PVCs for testing

4. **IAM OIDC Provider Issues**
   - Error: "no IAM OIDC provider associated with cluster"
   - Solution: Run `eksctl utils associate-iam-oidc-provider --region=$AWS_REGION --cluster=$EKS_CLUSTER_NAME --approve`

5. **Image Pull Errors**
   - Error: "ImagePullBackOff" or "ErrImagePull"
   - Solution: Check ECR repository permissions and image tags

6. **CodeBuild Access Denied**
   - Error: "ACCESS_DENIED" when triggering builds
   - Solution: Update CodeBuild role permissions

### Debugging Commands

```bash
# Check pod status
kubectl get pods -n mlops-premium

# Check pod logs
kubectl logs -n mlops-premium <pod-name>

# Check pod details
kubectl describe pod -n mlops-premium <pod-name>

# Check service details
kubectl describe svc -n mlops-premium <service-name>

# Check persistent volume claims
kubectl get pvc -n mlops-premium

# Check AWS EKS cluster status
aws eks describe-cluster --name premium-prediction-cluster --region $AWS_REGION

# Check CodeBuild project status
aws codebuild batch-get-projects --names premium-model-build
```

## Conclusion

This guide provides a comprehensive approach to deploying the Premium Prediction ML pipeline to AWS. By following these steps, you can set up a robust, scalable, and maintainable ML infrastructure that leverages AWS services for container orchestration, CI/CD, monitoring, and automated training.

For production deployments, consider additional enhancements:
- Set up proper DNS and TLS with an ingress controller
- Configure MLflow to use a persistent database (RDS)
- Implement more sophisticated monitoring and alerting
- Set up automated backups for critical data
- Implement proper secrets management with AWS Secrets Manager

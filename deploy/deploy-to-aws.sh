#!/bin/bash
set -e

# Load environment variables
source deploy/env.sh

echo "Starting AWS deployment workflow..."

# Step 1: Create ECR repositories if they don't exist
echo "Creating ECR repositories..."
aws ecr describe-repositories --repository-names premium-model-api 2>/dev/null || aws ecr create-repository --repository-name premium-model-api
aws ecr describe-repositories --repository-names mlflow-tracking 2>/dev/null || aws ecr create-repository --repository-name mlflow-tracking

# Step 2: Build and push Docker images (if Docker is running)
if docker info > /dev/null 2>&1; then
  echo "Building and pushing Docker images..."
  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
  
  # Build and push API image
  echo "Building and pushing API image..."
  docker build -t premium-model-api .
  docker tag premium-model-api:latest $ECR_API_REPO:latest
  docker push $ECR_API_REPO:latest
  
  # Pull and push MLflow image
  echo "Pulling and pushing MLflow image..."
  docker pull ghcr.io/mlflow/mlflow:v2.5.0
  docker tag ghcr.io/mlflow/mlflow:v2.5.0 $ECR_MLFLOW_REPO:latest
  docker push $ECR_MLFLOW_REPO:latest
  
  USE_ECR_IMAGES=true
else
  echo "Docker is not running. Skipping Docker image build and push."
  echo "Will use public images for deployment."
  USE_ECR_IMAGES=false
fi

# Step 3: Create S3 buckets if they don't exist
echo "Creating S3 buckets..."
aws s3api head-bucket --bucket $MODEL_ARTIFACTS_BUCKET 2>/dev/null || aws s3 mb s3://$MODEL_ARTIFACTS_BUCKET
aws s3api head-bucket --bucket $MODEL_DATA_BUCKET 2>/dev/null || aws s3 mb s3://$MODEL_DATA_BUCKET

# Step 4: Create or update EKS cluster
echo "Setting up EKS cluster..."
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

# Step 5: Create IAM policy for S3 access
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

# Step 6: Associate IAM OIDC provider with the cluster
echo "Associating IAM OIDC provider with the cluster..."
eksctl utils associate-iam-oidc-provider --region=$AWS_REGION --cluster=$EKS_CLUSTER_NAME --approve

# Step 7: Create namespace and ConfigMap
echo "Creating namespace and ConfigMap..."
kubectl create namespace mlops-premium --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap premium-model-aws-config -n mlops-premium \
  --from-literal=MLFLOW_S3_ENDPOINT_URL=https://s3.$AWS_REGION.amazonaws.com \
  --from-literal=MLFLOW_TRACKING_URI=http://mlflow-service:5000 \
  --from-literal=ARTIFACT_ROOT=s3://$MODEL_ARTIFACTS_BUCKET \
  --from-literal=AWS_REGION=$AWS_REGION \
  --dry-run=client -o yaml | kubectl apply -f -

# Step 8: Create Secret
echo "Creating Secret..."
kubectl create secret generic premium-model-secrets -n mlops-premium \
  --from-literal=GRAFANA_ADMIN_PASSWORD=admin \
  --from-literal=AWS_ACCESS_KEY_ID=your-access-key \
  --from-literal=AWS_SECRET_ACCESS_KEY=your-secret-key \
  --dry-run=client -o yaml | kubectl apply -f -

# Step 9: Create service account for MLflow
echo "Creating service account for MLflow..."
eksctl create iamserviceaccount \
  --name mlflow-sa \
  --namespace mlops-premium \
  --cluster $EKS_CLUSTER_NAME \
  --attach-policy-arn $POLICY_ARN \
  --approve \
  --override-existing-serviceaccounts

# Step 10: Deploy MLflow
echo "Deploying MLflow..."
if [ "$USE_ECR_IMAGES" = true ]; then
  MLFLOW_IMAGE=$ECR_MLFLOW_REPO:latest
else
  MLFLOW_IMAGE=ghcr.io/mlflow/mlflow:v2.5.0
fi

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
        image: $MLFLOW_IMAGE
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

# Step 11: Deploy API
echo "Deploying API..."
if [ "$USE_ECR_IMAGES" = true ]; then
  API_IMAGE=$ECR_API_REPO:latest
else
  API_IMAGE=nginx:latest
fi

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
        prometheus.io/port: "80"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: premium-model-api
        image: $API_IMAGE
        ports:
        - containerPort: 80
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
    targetPort: 80
  type: ClusterIP
EOF

kubectl apply -f kubernetes/api-deployment.yaml

# Step 12: Deploy Monitoring
echo "Deploying Monitoring..."
cat > kubernetes/monitoring-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: mlops-premium
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:9.3.6
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: premium-model-secrets
              key: GRAFANA_ADMIN_PASSWORD
        - name: GF_INSTALL_PLUGINS
          value: "grafana-piechart-panel,grafana-worldmap-panel"
---
apiVersion: v1
kind: Service
metadata:
  name: grafana-service
  namespace: mlops-premium
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
EOF

kubectl apply -f kubernetes/monitoring-deployment.yaml

# Step 13: Set up CloudWatch dashboard
echo "Setting up CloudWatch dashboard..."
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

echo "Deployment completed successfully!"
echo "Access MLflow UI: kubectl port-forward svc/mlflow-service 5000:5000 -n mlops-premium"
echo "Access API: kubectl port-forward svc/premium-model-api-service 8000:80 -n mlops-premium"
echo "Access Grafana: kubectl port-forward svc/grafana-service 3000:3000 -n mlops-premium"

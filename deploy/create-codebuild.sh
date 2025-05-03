#!/bin/bash
set -e

# Load environment variables
if [ -f deploy/env.sh ]; then
  source deploy/env.sh
else
  # Set default values if env.sh doesn't exist
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  AWS_REGION=$(aws configure get region)
  EKS_CLUSTER_NAME="premium-prediction-cluster"
fi

echo "Using AWS Account: $AWS_ACCOUNT_ID"
echo "Using AWS Region: $AWS_REGION"

# Create CodeBuild service role if it doesn't exist
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
  
  # Attach policies with correct names
  aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonECR-FullAccess 2>/dev/null || aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonECRFullAccess
  aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy
  aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  
  # Create custom policy for kubectl
  cat > kubectl-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "eks:DescribeCluster",
        "eks:ListClusters"
      ],
      "Resource": "*"
    }
  ]
}
POLICY

  aws iam put-role-policy --role-name $ROLE_NAME --policy-name kubectl-access --policy-document file://kubectl-policy.json
  
  echo "Created role: $ROLE_ARN"
else
  echo "Using existing role: $ROLE_ARN"
fi

# Store AWS account ID in Parameter Store
echo "Storing AWS account ID in Parameter Store..."
aws ssm put-parameter --name "/premium-model/aws-account-id" --value "$AWS_ACCOUNT_ID" --type String --overwrite

# Create CodeBuild project
echo "Creating CodeBuild project..."
aws codebuild create-project \
  --name premium-model-build \
  --source "{\"type\": \"GITHUB\", \"location\": \"https://github.com/yourusername/E2EMLOPsregression.git\"}" \
  --artifacts "{\"type\": \"NO_ARTIFACTS\"}" \
  --environment "{\"type\": \"LINUX_CONTAINER\", \"image\": \"aws/codebuild/amazonlinux2-x86_64-standard:3.0\", \"computeType\": \"BUILD_GENERAL1_SMALL\", \"privilegedMode\": true}" \
  --service-role "$ROLE_ARN"

echo "CodeBuild project created successfully!"

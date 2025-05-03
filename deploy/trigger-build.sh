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

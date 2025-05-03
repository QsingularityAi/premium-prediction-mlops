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

# Create a simple buildspec file
cat > buildspec-simple.yml << 'BUILDSPEC'
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.9
  pre_build:
    commands:
      - echo "Starting pre-build phase"
      - python --version
  build:
    commands:
      - echo "Starting build phase"
      - echo "This is a test build"
  post_build:
    commands:
      - echo "Build completed successfully!"

artifacts:
  files:
    - README.md
BUILDSPEC

# Create a simple CodeBuild project
aws codebuild create-project \
  --name premium-model-test-build \
  --source "{\"type\": \"NO_SOURCE\", \"buildspec\": \"$(cat buildspec-simple.yml | sed 's/"/\\"/g' | tr -d '\n')\"}" \
  --artifacts "{\"type\": \"NO_ARTIFACTS\"}" \
  --environment "{\"type\": \"LINUX_CONTAINER\", \"image\": \"aws/codebuild/amazonlinux2-x86_64-standard:3.0\", \"computeType\": \"BUILD_GENERAL1_SMALL\"}" \
  --service-role "arn:aws:iam::$AWS_ACCOUNT_ID:role/premium-model-codebuild-role"

echo "Simple test build project created successfully!"
echo "You can trigger it with: aws codebuild start-build --project-name premium-model-test-build"

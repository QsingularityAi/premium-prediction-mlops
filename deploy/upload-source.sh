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

echo "Creating source.zip file..."
zip -r source.zip . -x "*.git*" -x "venv/*" -x "*.zip"

echo "Uploading to S3 bucket: $BUCKET_NAME"
aws s3 cp source.zip s3://$BUCKET_NAME/

echo "Source code uploaded successfully!"
echo "The pipeline should start automatically."

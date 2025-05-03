#!/bin/bash
set -e

# Load environment variables
source deploy/env.sh

# Create EventBridge rule for scheduled training
aws events put-rule \
  --name premium-model-scheduled-training \
  --schedule-expression "rate(1 day)" \
  --state ENABLED

# Create target for the rule
aws events put-targets \
  --rule premium-model-scheduled-training \
  --targets "Id"="1","Arn"="arn:aws:codebuild:$AWS_REGION:$AWS_ACCOUNT_ID:project/premium-model-build","RoleArn"="arn:aws:iam::$AWS_ACCOUNT_ID:role/service-role/Amazon_EventBridge_Invoke_Build_1234567890"

echo "Scheduled training job set up successfully!"

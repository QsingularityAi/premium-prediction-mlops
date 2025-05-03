#!/bin/bash

# Load environment variables
source deploy/env.sh

# Create AWS Batch compute environment if it doesn't exist
COMPUTE_ENV_NAME="premium-training-compute-env"
COMPUTE_ENV_EXISTS=$(aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --query "computeEnvironments[0].computeEnvironmentName" --output text 2>/dev/null || echo "")

if [ "$COMPUTE_ENV_EXISTS" != "$COMPUTE_ENV_NAME" ]; then
  echo "Creating AWS Batch compute environment..."
  aws batch create-compute-environment \
    --compute-environment-name $COMPUTE_ENV_NAME \
    --type MANAGED \
    --state ENABLED \
    --compute-resources type=EC2,minvCpus=0,maxvCpus=16,desiredvCpus=0,instanceTypes=c5.xlarge,subnets=$(aws ec2 describe-subnets --query "Subnets[0].SubnetId" --output text),securityGroupIds=$(aws ec2 describe-security-groups --query "SecurityGroups[0].GroupId" --output text)
  
  # Wait for compute environment to be ready
  echo "Waiting for compute environment to be ready..."
  aws batch wait compute-environment-available --compute-environments $COMPUTE_ENV_NAME
fi

# Create job queue if it doesn't exist
JOB_QUEUE_NAME="premium-training-queue"
JOB_QUEUE_EXISTS=$(aws batch describe-job-queues --job-queues $JOB_QUEUE_NAME --query "jobQueues[0].jobQueueName" --output text 2>/dev/null || echo "")

if [ "$JOB_QUEUE_EXISTS" != "$JOB_QUEUE_NAME" ]; then
  echo "Creating AWS Batch job queue..."
  aws batch create-job-queue \
    --job-queue-name $JOB_QUEUE_NAME \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order order=1,computeEnvironment=$COMPUTE_ENV_NAME
  
  # Wait for job queue to be ready
  echo "Waiting for job queue to be ready..."
  aws batch wait job-queue-available --job-queues $JOB_QUEUE_NAME
fi

# Create job definition if it doesn't exist
JOB_DEF_NAME="premium-model-training"
JOB_DEF_EXISTS=$(aws batch describe-job-definitions --job-definition-name $JOB_DEF_NAME --status ACTIVE --query "jobDefinitions[0].jobDefinitionName" --output text 2>/dev/null || echo "")

if [ "$JOB_DEF_EXISTS" != "$JOB_DEF_NAME" ]; then
  echo "Creating AWS Batch job definition..."
  aws batch register-job-definition \
    --job-definition-name $JOB_DEF_NAME \
    --type container \
    --container-properties '{
      "image": "'$ECR_API_REPO':latest",
      "vcpus": 4,
      "memory": 8192,
      "command": ["python", "-m", "src.train"],
      "environment": [
        {"name": "LOG_LEVEL", "value": "info"},
        {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow-service:5000"},
        {"name": "MLFLOW_S3_ENDPOINT_URL", "value": "https://s3.'$AWS_REGION'.amazonaws.com"},
        {"name": "ARTIFACT_ROOT", "value": "s3://'$MODEL_ARTIFACTS_BUCKET'"}
      ]
    }'
fi

# Submit training job
echo "Submitting training job to AWS Batch..."
JOB_NAME="premium-training-job-$(date +%Y%m%d-%H%M%S)"
aws batch submit-job \
  --job-name $JOB_NAME \
  --job-queue $JOB_QUEUE_NAME \
  --job-definition $JOB_DEF_NAME

echo "Training job $JOB_NAME submitted successfully!"
echo "Monitor job status with: aws batch describe-jobs --jobs \$(aws batch list-jobs --job-queue $JOB_QUEUE_NAME --job-status RUNNING --query 'jobSummaryList[0].jobId' --output text)"
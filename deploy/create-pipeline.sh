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

# Create S3 bucket for artifacts if it doesn't exist
BUCKET_NAME="premium-model-pipeline-artifacts-$AWS_ACCOUNT_ID"
aws s3api head-bucket --bucket $BUCKET_NAME 2>/dev/null || aws s3 mb s3://$BUCKET_NAME

# Create CodePipeline service role if it doesn't exist
ROLE_NAME="premium-model-pipeline-role"
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ]; then
  echo "Creating CodePipeline service role..."
  
  # Create trust policy
  cat > trust-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codepipeline.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
POLICY

  # Create role
  ROLE_ARN=$(aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json --query 'Role.Arn' --output text)
  
  # Create policy document
  cat > pipeline-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "codebuild:BatchGetBuilds",
        "codebuild:StartBuild"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:GetObjectVersion",
        "s3:GetBucketVersioning"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    }
  ]
}
POLICY

  # Attach policy
  aws iam put-role-policy --role-name $ROLE_NAME --policy-name pipeline-policy --policy-document file://pipeline-policy.json
  
  echo "Created role: $ROLE_ARN"
else
  echo "Using existing role: $ROLE_ARN"
fi

# Create pipeline definition file
cat > pipeline-definition.json << EOF2
{
  "pipeline": {
    "name": "premium-model-pipeline",
    "roleArn": "${ROLE_ARN}",
    "artifactStore": {
      "type": "S3",
      "location": "${BUCKET_NAME}"
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
              "provider": "S3",
              "version": "1"
            },
            "configuration": {
              "S3Bucket": "${BUCKET_NAME}",
              "S3ObjectKey": "source.zip"
            },
            "outputArtifacts": [
              {
                "name": "SourceCode"
              }
            ],
            "runOrder": 1
          }
        ]
      },
      {
        "name": "Build",
        "actions": [
          {
            "name": "BuildAndDeploy",
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
            "runOrder": 1
          }
        ]
      }
    ]
  }
}
EOF2

# Create CodePipeline
echo "Creating CodePipeline..."
aws codepipeline create-pipeline --cli-input-json file://pipeline-definition.json

echo "CodePipeline created successfully!"
echo "To trigger the pipeline, upload your source code to S3:"
echo "zip -r source.zip . && aws s3 cp source.zip s3://$BUCKET_NAME/"

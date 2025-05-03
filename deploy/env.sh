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
#!/bin/bash
set -e

# Load environment variables
source deploy/env.sh

# Start a CodeBuild build
aws codebuild start-build \
  --project-name premium-model-build \
  --environment-variables-override name=TRAIN_MODEL,value=true,type=PLAINTEXT

echo "Training job triggered successfully!"

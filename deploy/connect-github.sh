#!/bin/bash
set -e

echo "This script will help you connect CodeBuild to your GitHub repository."
echo "You'll need to provide your GitHub personal access token."
echo ""

# Ask for GitHub details
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter your GitHub repository name: " GITHUB_REPO
read -s -p "Enter your GitHub personal access token: " GITHUB_TOKEN
echo ""

# Update the CodeBuild project
aws codebuild update-project \
  --name premium-model-build \
  --source "{\"type\": \"GITHUB\", \"location\": \"https://github.com/$GITHUB_USERNAME/$GITHUB_REPO.git\", \"auth\": {\"type\": \"OAUTH\", \"resource\": \"$GITHUB_TOKEN\"}}"

echo "GitHub connection updated successfully!"
echo "Now let's create a webhook to trigger builds automatically."

# Create webhook
aws codebuild create-webhook \
  --project-name premium-model-build

echo "Webhook created successfully!"

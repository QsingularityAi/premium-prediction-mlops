import boto3
import json
import os
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    AWS Lambda function to automatically trigger model retraining
    when CloudWatch alarms for model drift or performance degradation are triggered.
    
    Args:
        event: CloudWatch event that triggered the Lambda
        context: Lambda execution context
    
    Returns:
        Response dictionary with status code and message
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Parse CloudWatch alarm event
        if 'detail' in event and 'alarmName' in event['detail']:
            alarm_name = event['detail']['alarmName']
            logger.info(f"Processing alarm: {alarm_name}")
            
            # If this is a model drift or performance alarm, trigger retraining
            if alarm_name in ['FeatureDrift', 'HighModelError', 'ModelPerformanceDegradation']:
                batch = boto3.client('batch')
                job_queue = os.environ.get('JOB_QUEUE', 'premium-training-queue')
                job_definition = os.environ.get('JOB_DEFINITION', 'premium-model-training')
                
                # Submit training job
                response = batch.submit_job(
                    jobName=f'auto-retraining-{context.aws_request_id[:8]}',
                    jobQueue=job_queue,
                    jobDefinition=job_definition,
                    parameters={
                        'TRIGGER_SOURCE': 'lambda',
                        'ALARM_NAME': alarm_name
                    }
                )
                
                job_id = response['jobId']
                logger.info(f"Triggered model retraining with job ID: {job_id}")
                
                # Send notification about retraining
                sns = boto3.client('sns')
                topic_arn = os.environ.get('SNS_TOPIC_ARN')
                
                if topic_arn:
                    sns.publish(
                        TopicArn=topic_arn,
                        Subject=f"Automatic model retraining triggered by {alarm_name}",
                        Message=f"Model retraining has been automatically triggered due to {alarm_name} alarm.\n\n"
                                f"Job ID: {job_id}\n"
                                f"Triggered by: CloudWatch Alarm\n"
                                f"Timestamp: {event['time'] if 'time' in event else 'N/A'}"
                    )
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': f"Triggered model retraining with job ID: {job_id}",
                        'jobId': job_id
                    })
                }
            else:
                logger.info(f"Alarm {alarm_name} does not require retraining")
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': f"Alarm {alarm_name} processed but no action taken"
                    })
                }
        else:
            logger.warning("Event format not recognized as CloudWatch alarm")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'message': "Event format not recognized as CloudWatch alarm"
                })
            }
    
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f"Error processing event: {str(e)}"
            })
        }
from sagemaker.tensorflow import TensorFlow
import sagemaker
import boto3

session = boto3.Session(region_name='eu-central-1')
sagemaker_session = sagemaker.Session(boto_session=session)
role = 'arn:aws:iam::564083281396:role/AmazonSageMakerExecutionRoleForTraining'

estimator = TensorFlow(
    entry_point='train.py',
    source_dir='code',
    role=role,
    instance_type='ml.m5.large',
    instance_count=1,
    framework_version='2.13',
    py_version='py310',
    hyperparameters={
        'EPOCHS': 10,
        'BATCH_SIZE': 32
    },
    output_path='s3://skin-lesion-ham10000-euc1/output',
    base_job_name='cnn-ham',
)

estimator.fit({
    'training': 's3://skin-lesion-ham10000-euc1/HAM10000_sorted/train/'
})

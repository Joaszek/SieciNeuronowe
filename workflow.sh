aws sagemaker create-training-job \
  --training-job-name cnn-ham-$(date +%s) \
  --algorithm-specification TrainingImage=763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-training:2.13-cpu-py310,TrainingInputMode=File \
  --role-arn arn:aws:iam::564083281396:role/AmazonSageMakerExecutionRoleForTraining \
  --input-data-config '[
    {
      "ChannelName":"train",
      "DataSource":{
        "S3DataSource":{
          "S3DataType":"S3Prefix",
          "S3Uri":"s3://skin-lesion-ham10000-euc1/HAM10000_sorted/train/",
          "S3DataDistributionType":"FullyReplicated"
        }
      }
    },
    {
      "ChannelName":"val",
      "DataSource":{
        "S3DataSource":{
          "S3DataType":"S3Prefix",
          "S3Uri":"s3://skin-lesion-ham10000-euc1/HAM10000_sorted/val/",
          "S3DataDistributionType":"FullyReplicated"
        }
      }
    },
    {
      "ChannelName":"code",
      "DataSource":{
        "S3DataSource":{
          "S3DataType":"S3Prefix",
          "S3Uri":"s3://skin-lesion-ham10000-euc1/code/latest/",
          "S3DataDistributionType":"FullyReplicated"
        }
      }
    }
  ]' \
  --output-data-config '{"S3OutputPath":"s3://skin-lesion-ham10000-euc1/output"}' \
  --resource-config '{"InstanceType":"ml.m5.large","InstanceCount":1,"VolumeSizeInGB":20}' \
  --stopping-condition '{"MaxRuntimeInSeconds":1800,"MaxWaitTimeInSeconds":3600}' \
  --enable-managed-spot-training \
  --region eu-central-1

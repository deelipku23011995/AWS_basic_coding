
import boto3
import json
import sys
import time
import urllib.parse
import boto3
import re
import uuid
import pandas as pd
from botocore.client import Config
import hashlib
from botocore.errorfactory import ClientError
import time
import os
import warnings
warnings.filterwarnings('ignore')

import io
import pandas as pd
import boto3
import argparse


s3 = boto3.client('s3')
config = Config(retries = dict(max_attempts = 20),region_name='eu-west-2') # Amazon Textract client 



class ProcessType:
    DETECTION = 1
    ANALYSIS = 2


class DocumentProcessor:
    jobId = ''
    textract = boto3.client('textract', config=config)
    sqs = boto3.client('sqs',config=config)
    sns = boto3.client('sns',config=config)

    roleArn = ''
    bucket = ''
    document = ''

    sqsQueueUrl = ''
    snsTopicArn = ''
    processType = ''
    s3_dir_key = ''
    dest_bucket = ''
    
    def main(self, bucketName, documentName, key_id, dest_bucket, s3_dir_key, process_type="DETECTION"):
        self.roleArn = 'arn:aws:iam::661082688832:role/service-role/AmazonSageMaker-ExecutionRole-20210921T210509'

        self.bucket = bucketName
        self.document = documentName
        self.key_id = key_id
        self.s3_dir_key = s3_dir_key
        self.dest_bucket = dest_bucket
        # self.file_name = file_name

        #self.CreateTopicandQueue()
        if process_type=="DETECTION":
            self.ProcessDocument(ProcessType.DETECTION)
        elif process_type=="ANALYSIS":
            self.ProcessDocument(ProcessType.ANALYSIS)
        else:
            raise Exception(f"process_type can be DETECTION/ANALYSIS, but \"{process_type}\" was passed.")
        #self.DeleteTopicandQueue()

    def ProcessDocument(self, type):
        jobFound = False

        self.processType = type
        validType = False

        # Determine which type of processing to perform
        if self.processType == ProcessType.DETECTION:
            response = self.textract.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket': self.bucket, 'Name': self.document}})
            print('Processing type: Detection')
            validType = True

        if self.processType == ProcessType.ANALYSIS:
            response = self.textract.start_document_analysis(DocumentLocation={'S3Object': {'Bucket': self.bucket, 'Name': self.document}},
                                                             FeatureTypes=[
                                                                 "FORMS","LAYOUT"])
            print('Processing type: Analysis')
            validType = True

        if validType == False:
            raise Exception(f"process_type can be DETECTION/ANALYSIS, but \"{process_type}\" was passed.")

        print('Start Job Id: ' + response['JobId'])
        
        
        if self.processType == ProcessType.DETECTION:
            while(self.textract.get_document_text_detection(JobId=str(response['JobId']))["JobStatus"]!="SUCCEEDED"):
                pass
        
        if self.processType == ProcessType.ANALYSIS:
            while(self.textract.get_document_analysis(JobId=str(response['JobId']))["JobStatus"]!="SUCCEEDED"):
                pass
        
        print('Matching Job Found:' + response['JobId'])
        print("storing in S3")
        jobFound = True
        results = self.GetResults(response['JobId'])
        self.StoreInS3(results)


        print('Done!')

    # Store the result in a S3 bucket
    def StoreInS3(self, response):
        print('registering in s3 bucket...')
        outputInJsonText = str(response)
        filename = str(self.key_id).split('/')[-1]
        pdfTextExtractionS3ObjectName = os.path.join(self.s3_dir_key, str(filename) + ".json") 
        pdfTextExtractionS3Bucket = self.dest_bucket

        s3 = boto3.client('s3')

        s3.put_object(Body=outputInJsonText,
                      Bucket= pdfTextExtractionS3Bucket, Key=pdfTextExtractionS3ObjectName)
        print('file ' + pdfTextExtractionS3ObjectName + ' registered successfully!')

    def CreateTopicandQueue(self):

        millis = str(int(round(time.time() * 1000)))

        # Create SNS topic
        snsTopicName = "AmazonTextractTopic" + millis

        topicResponse = self.sns.create_topic(Name=snsTopicName)
        self.snsTopicArn = topicResponse['TopicArn']

        # create SQS queue
        sqsQueueName = "AmazonTextractQueue" + millis
        self.sqs.create_queue(QueueName=sqsQueueName)
        self.sqsQueueUrl = self.sqs.get_queue_url(
            QueueName=sqsQueueName)['QueueUrl']

        attribs = self.sqs.get_queue_attributes(QueueUrl=self.sqsQueueUrl,
                                                AttributeNames=['QueueArn'])['Attributes']

        sqsQueueArn = attribs['QueueArn']

        # Subscribe SQS queue to SNS topic
        self.sns.subscribe(
            TopicArn=self.snsTopicArn,
            Protocol='sqs',
            Endpoint=sqsQueueArn)

        # Authorize SNS to write SQS queue
        policy = """{{
  "Version":"2012-10-17",
  "Statement":[
    {{
      "Sid":"MyPolicy",
      "Effect":"Allow",
      "Principal" : {{"AWS" : "*"}},
      "Action":"SQS:SendMessage",
      "Resource": "{}",
      "Condition":{{
        "ArnEquals":{{
          "aws:SourceArn": "{}"
        }}
      }}
    }}
  ]
}}""".format(sqsQueueArn, self.snsTopicArn)

        response = self.sqs.set_queue_attributes(
            QueueUrl=self.sqsQueueUrl,
            Attributes={
                'Policy': policy
            })

    def DeleteTopicandQueue(self):
        self.sqs.delete_queue(QueueUrl=self.sqsQueueUrl)
        self.sns.delete_topic(TopicArn=self.snsTopicArn)

    def GetResults(self, jobId):
        maxResults = 1000
        paginationToken = None
        finished = False
        pages = []

        while finished == False:

            response = None

            if self.processType == ProcessType.DETECTION:
                if paginationToken == None:
                    response = self.textract.get_document_text_detection(JobId=jobId,
                                                                         MaxResults=maxResults)
                else:
                    response = self.textract.get_document_text_detection(JobId=jobId,
                                                                         MaxResults=maxResults,
                                                                         NextToken=paginationToken)
                    
                    
            if self.processType == ProcessType.ANALYSIS:
                if paginationToken == None:
                    response = self.textract.get_document_analysis(JobId=jobId,
                                                                         MaxResults=maxResults)
                else:
                    response = self.textract.get_document_analysis(JobId=jobId,
                                                                         MaxResults=maxResults,
                                                                         NextToken=paginationToken)


            # Put response on pages list
            pages.append(response)

            if 'NextToken' in response:
                paginationToken = response['NextToken']
            else:
                finished = True

        # convert pages as JSON pattern
        line_list=[]
        word_counter = 0
        line_counter = 0
        n_pages = (pages[0]["DocumentMetadata"]["Pages"])
  
        for item in pages[0]["Blocks"]:
            if item["BlockType"] == "LINE":
                line_counter = line_counter + 1
                line_list.append(item["Text"])
            if item["BlockType"] == "WORD":
                word_counter = word_counter + 1
                
        rawtext=' '.join(line_list)
        cutoff = min(500,len(rawtext))
        language = "EN"
        # response = clientt.detect_dominant_language(Text=str(rawtext)[:cutoff])
        # language = response["Languages"][0]["LanguageCode"]

        pages = json.dumps(pages)
        return pages
    
    
    
def process_text_analysis(bucket, document):
    #Get the document from S3
    s3_connection = boto3.resource("s3")
    
    client = boto3.client('s3')
    
    result = client.get_object(Bucket=bucket, Key=document)
    text = result['Body'].read().decode('utf-8')
    res = json.loads(text)
    
    left_cor = []
    top_cor = []
    width_cor = []
    height_cor = []
    page = []

    line_text = []

    for response in res:
        blocks=response["Blocks"]
        for block in blocks:
            if (block["BlockType"] == "WORD"):
                left_cor.append(float("{:.2f}".format(block["Geometry"]["BoundingBox"]["Left"])))
                top_cor.append(float("{:.2f}".format(block["Geometry"]["BoundingBox"]["Top"])))
                width_cor.append(float("{:.2f}".format(block["Geometry"]["BoundingBox"]["Width"])))
                height_cor.append(float("{:.2f}".format(block["Geometry"]["BoundingBox"]["Height"])))
                line_text.append((block["Text"]))
                page.append(block["Page"])
    
    df = pd.DataFrame(list(zip(left_cor,top_cor,width_cor,height_cor,line_text,page)),columns = ["xmin","ymin","width_cor","height_cor","line_text","page"])    
    df["xmax"] = (df["xmin"] + df["width_cor"])
    df["ymax"] = (df["ymin"] + df["height_cor"])
    
    pages = df.page.unique().tolist()
    text_dict = {}
    for p in pages:
        dfp = df[df.page==p]
        txt_list = dfp.line_text.tolist()
        txt = " ".join(txt_list)
        text_dict[p] = txt

    return df,text_dict,res


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-path", type=str)
    
    args, _ = parser.parse_known_args()
    
    print("Received arguments {}".format(args))
    print("Key Path:---",args.key_path)
    

    key_path = args.key_path
    key = key_path
    s3 = boto3.client('s3')
    bucket ='lossadjustmentdataset'
    dest_bucket = 'deeliptutorial'
    s3_dir_key = 'XAAS_Practice/XAAS_processing_job_testing/preprocessing_outptut'

    analyzer = DocumentProcessor()
    digest = key.replace(".pdf","")
    analyzer.main(bucket,key_path,digest, dest_bucket, s3_dir_key, process_type="ANALYSIS")


    #  Extracting Daata path from s3


    bucket_name='deeliptutorial'
    prefix = 'XAAS_Practice/XAAS_processing_job_testing/preprocessing_outptut'

    import boto3
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name,Prefix=prefix)
    ct = 0
    f_list = []
    for page in pages:
        for obj in page['Contents']:
            if ".json" in obj['Key'].lower():
                ct = ct + 1
                f_list.append(obj['Key'])
                
                
    key_path = f_list[0]
    bucket_name='deeliptutorial'
    df,text_dict,res = process_text_analysis(bucket_name, key_path)
    
    ## Saving the file in op/ml/processing/train
    
    train_features_output_path = os.path.join("/opt/ml/processing/output", "output_sklearn_built_in.csv")
    print("Saving training features to {}".format(train_features_output_path))
    df.to_csv(train_features_output_path, header=False, index=False)
    
    
    excel_path = key_path.replace("json","xlsx")
    
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer)
        data = output.getvalue()

    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).put_object(Key=excel_path, Body=data)
    
    
    
    



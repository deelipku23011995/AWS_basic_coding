import sys
import numpy
def handler(event, context):
    print("event:-----",event)
#     return 'Hello from AWS Lambda using Python' + sys.version + '!  array is :--'+ str(numpy.random.rand(2))
    return event
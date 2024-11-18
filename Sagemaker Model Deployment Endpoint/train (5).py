
import torch
import torchvision
import transformers

print(f"BMBM: Versions >> {torch.__version__} ,{torchvision.__version__} , {transformers.__version__}")



import boto3
import sys
import argparse
import pandas as pd
import numpy as np
import ast
import json
import io
import os
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import LayoutLMv2Processor, AdamW
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from transformers import LayoutLMv2ForSequenceClassification, AdamW
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, recall_score

from PIL import Image
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_model(model, optimizer, dataloader):
    global train_losses, train_accuracies, train_recalls
    losses=[]
    true_labels=[]
    pred_labels=[]
    model.train()
    for batch in tqdm(dataloader, desc="Training"):
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, labels=labels) 
        loss = outputs.loss
        losses.append(loss.item())
        
        predictions = outputs.logits.argmax(-1)
        
        true_labels.extend(labels.tolist())
        pred_labels.extend(predictions.tolist())
        
        loss.backward()
        optimizer.step()
    train_loss = np.mean(losses)
    accuracy = 100 * accuracy_score(true_labels, pred_labels)
    recall = 100* recall_score(true_labels, pred_labels, average='macro', pos_label=1)
    train_losses.append(train_loss)
    train_accuracies.append(accuracy)
    train_recalls.append(recall)
    print(f'train loss---> {train_loss}, train accuracy---> {accuracy}, train recall---> {recall}')


def eval_model(model, dataloader):
    global eval_loss, eval_recall
    global eval_losses, eval_accuracies, eval_recalls
    losses=[]
    true_labels=[]
    pred_labels=[]
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                            token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
            predictions = outputs.logits.argmax(-1)
            
            true_labels.extend(labels.tolist())
            pred_labels.extend(predictions.tolist())
    
    eval_loss = np.mean(losses)
    accuracy = 100 * accuracy_score(true_labels, pred_labels)
    recall = 100* recall_score(true_labels, pred_labels, average='macro', pos_label=1)
    eval_recall = recall
    eval_losses.append(eval_loss)
    eval_accuracies.append(accuracy)
    eval_recalls.append(recall)
    print(f'eval loss---> {eval_loss}, eval accuracy---> {accuracy}, eval recall---> {eval_recall}')
    
def preprocess_data(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    words = examples['words']
    boxes = examples['bbox']
    label = examples['label']

    encoded_inputs = processor(images, words, boxes=boxes, max_length=512,
                             padding="max_length", truncation=True)
    encoded_inputs['labels'] = label
    return encoded_inputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    
    dff = pd.read_csv('all_data.csv',index_col= 0)
    dff = dff.dropna()
    print(dff.shape)
    
    dff['words'].astype('str')
    dff['words'] = dff['words'].apply(lambda x: ast.literal_eval(x))
    dff['bbox'].astype('str')
    dff['bbox'] = dff['bbox'].apply(lambda x: ast.literal_eval(x))
    
    
    # For bordereau
    t_labels = [i.split('/')[-2] for i in dff['image_path']]

    df_labels = []
    for l in t_labels:
        if 'bordereau' in l :
            df_labels.append(1)
        else:
            df_labels.append(0)
    print(len(df_labels)  )
    
    dff['label']=df_labels
    dff = dff.dropna()
    print(dff.shape)
    
    print(dff['label'].value_counts())
    
    data_train, data_eval = train_test_split(dff, test_size=0.20, stratify=dff['label'], shuffle=True)
    print("\n Shape **********************************")
    print(data_train.shape, data_eval.shape)
    
    
#     a = data_train.iloc[[1095,1318,1289,1095,0,10]]
#     b = data_eval.iloc[[0,10]]

#     train_dataset = Dataset(pa.Table.from_pandas(a))
#     eval_dataset = Dataset(pa.Table.from_pandas(b))
    

    labels  = ['others', 'bordereau']
    
    ##6 convert data_df to Huggingface dataset
    train_dataset = Dataset(pa.Table.from_pandas(data_train))
    eval_dataset = Dataset(pa.Table.from_pandas(data_eval))
    
    
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    #we need to define custom features
    features = Features({
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': ClassLabel(names=labels),
    })
    
    eval_dataset =eval_dataset.map(preprocess_data, batched=True, remove_columns=eval_dataset.column_names,
                                features=features)
    
    train_dataset =train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names,
                                features=features)    
    
    ## setting format to torch
    BATCH_SIZE = 1

    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")
    #test_dataset.set_format(type="torch")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    batch = next(iter(train_dataloader))
    for k,v in batch.items():
        if type(v)==list:
            print(k, len(v))
        else:
            print(k, v.shape)
    
    print(len(train_dataloader))
    print(len(list(eval_dataloader)))
    
    
    ########################################
    # Model creation and training

    #assert 1==1 #comment this line to start training, be careful with model paths, it will overwrite existing model

    MODEL_PATH_BEST = "/opt/ml/model/model_page_classification_best.pt"
    MODEL_PATH_LAST = "/opt/ml/model/model_page_classification_last.pt"


    #loading model from pretrained
    model = LayoutLMv2ForSequenceClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                              num_labels=len(labels))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=4e-5)

    num_train_epochs = 10
    t_total = len(train_dataloader) * num_train_epochs # total number of training steps
    print("total number of training steps--->",t_total)

    global train_losses, eval_losses, train_accuracies, eval_accuracies, train_recalls, eval_recalls
    train_losses,eval_losses,train_accuracies,eval_accuracies,train_recalls, eval_recalls=[],[],[],[],[],[]

    global eval_loss
    global eval_recall
    eval_recall=0
    best_recall=0
    eval_loss=0
    best_loss=0
    best_epoch=0
    early_stopping_round=0
    EARLY_STOPPING_PATIENCE = 30

    t1=datetime.now()

    for epoch in range(num_train_epochs):
        print('*********************************************\n')
        print("Epoch:", epoch)
        print("early_stopping_round:", early_stopping_round)

        if early_stopping_round >= EARLY_STOPPING_PATIENCE:
            print("Aborting due to early stopping rounds.")
            break

        train_model(model, optimizer, train_dataloader)
        eval_model(model, eval_dataloader)

        if (eval_recall > best_recall) or (best_recall==0):
            best_recall = eval_recall
            best_epoch = epoch
            print(f"Model saved--->TRUE, best recall--->{best_recall}, best epoch--->{best_epoch}")
            torch.save(model.state_dict(), MODEL_PATH_BEST)
            early_stopping_round = 0
        else:
            print(f"Model saved--->FALSE, best recall--->{best_recall}, best epoch--->{best_epoch}")
            early_stopping_round+=1

    t2=datetime.now()


    torch.save(model.state_dict(), MODEL_PATH_LAST)

    training_history = pd.DataFrame({'train_loss':train_losses, 'eval_loss':eval_losses, 'train_accuracy':train_accuracies, 
                                     'eval_accuracy':eval_accuracies, 'train_recall':train_recalls, 'eval_recall':eval_recalls})
    training_history.to_csv('/opt/ml/model/training_history.csv',index=False)

            
    
    
    
    
    
    
    
    
    

    

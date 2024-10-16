# factual-accuracy: validating numerical values in finacial summaries

## The FactCheck model prediction results are available under "results" folder.  

## Prerequisite: Python 3.6 or higher versions
## clone the git repo and download the finetuned model checkpoint-22588 from the google drive [link](https://drive.google.com/drive/folders/1VAH5KL5v10CTpaMapj65aO267EMJlKkL?usp=sharing)
## create python environemnt
python3 -m venv .env
## enable environment
source .env/bin/activate

## Install all packages in requirement.txt
pip3 install -r requirement.txt   <br>
python -m spacy download en_core_web_sm  

## RESULT
### run the following test files. The testdata folder has 8 test files for 4 test dataset with original gpt4 summary and manual corrected ones.
python infer.py 'testdata/bbc-business-news-summary - extractive.csv'  <br>
python infer.py 'testdata/bbc-business-news-summary - gpt4 - corrected.csv' <br>
python infer.py 'testdata/bbc-business-news-summary - gpt4 - original.csv'  <br>
python infer.py 'testdata/findsum_subset - gpt4 - corrected.csv'  <br>
python infer.py 'testdata/findsum_subset - gpt4 - original.csv'  <br>
python infer.py 'testdata/indian_financial_news - T5 - corrected.csv'  <br>
python infer.py 'testdata/indian_financial_news - T5 - original.csv'  <br>
python infer.py 'testdata/us_financial_news - gpt4 - corrected.csv'   <br>
python infer.py 'testdata/us_financial_news - gpt4 - original.csv'   <br>


### the corresponding result files are stored in the "results" folder  

### original findsum_subset and india_financial_news test dataset files are in "orginal" folder.

# To Train the model
### the train dataset are avialble under "train" folder
### run the follwoing command to finetune the T5-base model
python train.py
### The trained model is saved in checkpoint-22588 

## Description of the other python scripts
### 1. rag_pipeline.py : this file used to select best passage for each sentence in the summary after splitting the source document.
### 2. doc_chunking.py: this script is used to split the training dataset in smaller passages
### 3. semantic_chunking.py : This script split source document in smaller passages based on similarity score of the consecuative sentecnes.  this script is internally called from  rag_pipeline.py & doc_chunking.py
### 4. dataProcessing.py: this script mask the numerical values and create final input data to teh model.

### Apache License Version 2.0


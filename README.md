# factual-accuracy: validating numerical values in finacial summaries

## Prerequisite: Python 3.6 or higher versions
## clone the git repo and unzip the model checkpoint-22588.zip
## create python environemnt
python3 -m venv .env
## enable environment
source .env/bin/activate

## Install all packages in requirement.txt
pip3 install -r requirement.txt
python -m spacy download en_core_web_sm

## RESULT
### run the following test files. The testdata folder has 8 test files for 4 test dataset with original gpt4 summary and manual corrected ones.
python infer.py 'testdata/bbc-business-news-summary - extractive.csv'  
python infer.py 'testdata/bbc-business-news-summary - gpt4 - corrected.csv' 
python infer.py 'testdata/bbc-business-news-summary - gpt4 - original.csv'
python infer.py 'testdata/findsum_subset - gpt4 - corrected.csv'
python infer.py 'testdata/findsum_subset - gpt4 - original.csv'
python infer.py 'testdata/indian_financial_news - T5 - corrected.csv'
python infer.py 'testdata/indian_financial_news - T5 - original.csv'
python infer.py 'testdata/us_financial_news - gpt4 - corrected.csv'
python infer.py 'testdata/us_financial_news - gpt4 - original.csv'


### the corresponding result files are stored in the "results" folder  

### original findsum_subset and india_financial_news dataset files are in "orginal" folder.

# To Train the model
### the train dataset are avialble under "train" folder
### run the follwoing command
python train.py
### The trained model is saved in checkpoint-22588 

## Description of the other python scripts
### 1. rag_pipeline.py : this file used to select best passage for each sentence in the summary after splitting the source document.
### 2. doc_chunking.py: ths script is used to split the training dataset in smaller passages
### 3. semantic_chunking.py : Thsi script split source document in smaller passages based on similarity score of the consecuative sentecnes.  this script is internally called from  rag_pipeline.py & doc_chunking.py
### 4. dataProcessing.py: this script mask the numerical values and create final input data to teh model.

### Apache License Version 2.0


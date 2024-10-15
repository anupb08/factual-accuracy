import nltk
import evaluate
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset,Dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
model.to('cuda')


#df = pd.read_excel("train28k.xlsx")
df1 = pd.read_csv("traindata/financial-report-sec.csv")
df2 = pd.read_csv("traindata/FINDSum.csv")
df = df1.append(df2)
print(df.head())

data=Dataset.from_pandas(df)
data=data.train_test_split(test_size=0.2)


# We prefix our tasks with "answer the question"
prefix = "Using the above 'context', predict [NUM] as numerical value in following text:"

# Define the preprocessing function

def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = ["context: " + doc1 + "\n" + prefix + " " + doc2 for doc1, doc2 in zip(examples["source"], examples["masked"])]
   model_inputs = tokenizer(inputs, max_length=1024, truncation=True,padding=True)
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target=examples["num"],
                       max_length=8,
                      truncation=True,
                      padding=True,)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = data.map(preprocess_function, batched=True)


nltk.download("punkt", quiet=True)
#metric = evaluate.load("rouge")
metric = evaluate.load("exact_match")

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


   result = metric.compute(predictions=decoded_preds, references=decoded_labels)

   return result



# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="./results",
   evaluation_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)


trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

trainer.train()

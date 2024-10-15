from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import evaluate
import sys
import os

last_checkpoint = "checkpoint-22588/"

tokenizer = T5Tokenizer.from_pretrained(last_checkpoint, device="cuda")#).to("cuda")
finetuned_model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
finetuned_model.to('cuda')
#tokenizer.to("cuda")

folder = "processed"
file = "test_bbc-business-news-summary.csv"
file = "test_bbcSummary.csv"
#file = "testFileLiqSum2.csv"
#file = "test_liquiditySummary_dynamic_rag.csv" 
#file = "test_kdave-Indian_Financial_News1_rag.csv"
#file = "test_kdave-Indian_Financial_News_10000.csv"
#file = "test_test_ashraq_sum.csv"
#file = "test_ashraq_edited_corrected.csv"
#file = "test_ashraq_masked.csv"
#file = "test_Liquidity_summary_dynamics_rag.csv"
#file = "test_Liquidity_summary_dynamics_error_corrected_rag.csv"
#file = "test_kdave-Indian_Financial_News1_rag3.csv"
#file = "test_kdave-Indian_Financial_News1_rag_mukaj.csv"

def inference(df):
    b = 0
    inputs = []
    preds = []
    labels = []
    for index, row in df.iterrows():
        doc1 =  row['source']
        print(row['summ_no'])
        doc_prev = row['prev']
        doc2 = row['masked']
        num = row['target']
        #if len(doc2) < 50:
        #    df.drop(index, inplace=True)
        #    continue
        #print(doc_prev)
        p = "context: " + doc_prev+ " " + doc1 + "\n" + "Using the above 'context', predict [NUM] as numerical value in following text:\n " + doc2
        inputs.append(p)
        labels.append(num)
        b = b+1
        if b > 2:
            b = 0
            inputs = tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")
            outputs = finetuned_model.generate(**inputs)
            answer = tokenizer.batch_decode(outputs.to("cpu"), skip_special_tokens=True)
            preds.extend(answer)
            inputs = []

    if len(inputs) > 0:
        inputs = tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")
        outputs = finetuned_model.generate(**inputs)
        answer = tokenizer.batch_decode(outputs.to("cpu"), skip_special_tokens=True)
        preds.extend(answer)
        inputs = []

    return preds


file = sys.argv[1]
folder, filename = os.path.split(file)
print(file)
#file = "test_bbcSummary.csv"
df = pd.read_csv(file, header = 0, dtype=str, keep_default_na=False)
preds = inference(df)
    #print(preds)
    #print(labels)
df["preds"] = preds
    #df["compare"] = [str(a) == str(b) for a, b in zip(preds, labels)]
    #df["corrected"] = labels

df.to_csv('results/'+filename, index=False)
    #metric = evaluate.load("exact_match")
    #result = metric.compute(predictions=preds, references=labels)
    #print(result)

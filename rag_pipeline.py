### Retrieve best chunk for each sentence in summary text.


import torch.nn.functional as F
import nltk
import pandas as pd
import csv
import numpy as np

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from semantic_chunking import get_splitter


tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model = AutoModel.from_pretrained('intfloat/e5-large-v2')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def retrieve_best_chunk(document, summary, doc_no):

    input_texts, semantic_chunks, queries = chunks_doc_and_summary(document, summary)
  
    ## following code is to retrieve best chuck for a query (here for each sentence from summary text)

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    l = len(queries)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:l] @ embeddings[l:].T) * 100
    indx = np.argmax(scores.tolist(), axis=1)
    psg_chunks = []
    for e, i in enumerate(indx):
        print(queries[e])
        #print(semantic_chunks[i])
        if i > 0:
            psg_chunks.append({"doc_no": doc_no, "summ_no": i, "prev_chunk": semantic_chunks[i-1], "doc_chunk": semantic_chunks[i],  "sentence": queries[e]})
        else:
            psg_chunks.append({"doc_no": doc_no, "summ_no": i, "prev_chunk": "", "doc_chunk": semantic_chunks[i], "sentence": queries[e]})

    return psg_chunks





def chunks_doc_and_summary(document, summary):
    ## input:
    ##      document: source document 
    ##      summary: summary of thesource document

    ## output: queries is a list of sentences from the summary text, semantic_chunks is a list of 
    ##      chunks from source document, and input_text is diction contain both queries and document chunks 
    ##      which will be used for retrieval purpose

    semantic_chunks = get_splitter(document)
    input_texts = []
    queries = nltk.sent_tokenize(summary)
    for q in queries:
        input_texts.append('query: ' + q)

    for chunk in semantic_chunks:
        input_texts.append('passage: ' + chunk)


    return input_texts, semantic_chunks, queries



def load_input_file(input_file):
    doc_chunks = []
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        doc_no = -1
        for row in reader:
            doc_no = doc_no + 1
            document = row['document']
            summary = row['summary']
            print(len(document))
            if len(document) > 46000:
                continue
            psg_chunks = retrieve_best_chunk(document, summary, doc_no)
            doc_chunks.extend(psg_chunks)
            #print(row)
    return doc_chunks


def write_output(out_file, doc_chunks):
   with open(out_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["doc_no","summ_no", "prev_chunk", "doc_chunk", "sentence"])
        writer.writeheader()
        writer.writerows(doc_chunks)

file = sys.argv[1]
folder, input_file = os.path.split(file)
print(file)

doc_chunks = load_input_file(input_file)
out_file = "rag_"+ input_file
write_output(out_file, doc_chunks)

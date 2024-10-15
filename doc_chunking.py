from typing import List

from sentence_transformers import SentenceTransformer, util

import spacy

import csv
import pandas as pd

_model_ = 'intfloat/e5-large-v2'
#_model_ = 'all-MiniLM-L6-v2'

class SentenceTransformersSimilarity:

    def __init__(self, model=_model_, similarity_threshold=0.2):

        self.model = SentenceTransformer(model)

        self.similarity_threshold = similarity_threshold

    def similarities(self, sentences: List[str]):

        # Encode all sentences

        embeddings = self.model.encode(sentences)

        # Calculate cosine similarities for neighboring sentences

        similarities = []

        for i in range(1, len(embeddings)):

            sim = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()

            similarities.append(sim)

        return similarities


class SpacySentenceSplitter():

    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")

    def split(self, text: str) -> List[str]:

        doc = self.nlp(text)

        return [str(sent).strip() for sent in doc.sents]


class SimilarSentenceSplitter():

    def __init__(self, similarity_model, sentence_splitter):

        self.model = similarity_model

        self.sentence_splitter = sentence_splitter

    def split_text(self, text: str, group_max_sentences=10) -> List[str]:

        """

        group_max_sentences: The maximum number of sentences in a group.

        """

        sentences = self.sentence_splitter.split(text)

        if len(sentences) == 0:

            return []

        similarities = self.model.similarities(sentences)

        # The first sentence is always in the first group.

        groups = [[sentences[0]]]

        # Using the group min/max sentences contraints,

        # group together the rest of the sentences.

        for i in range(1, len(sentences)):

            if len(groups[-1]) >= group_max_sentences:

                groups.append([sentences[i]])

            elif similarities[i - 1] >= self.model.similarity_threshold:

                groups[-1].append(sentences[i])

            else:

                groups.append([sentences[i]])

        return [" ".join(g) for g in groups]



#text = open("source.txt").read()
#text = open("tt").read()
#print(text)

def get_splitter(passage):
    model = SentenceTransformersSimilarity() # emb model

    sentence_splitter = SpacySentenceSplitter() # sentence tokenizer

    splitter = SimilarSentenceSplitter(model, sentence_splitter)

    #text = open(file).read()

    output = splitter.split_text(passage)
    #print(output)
    return output

def read_input(input_file):
    doc_chunks = []
    with open(input_file, newline='') as csvfile:
        df = pd.read_csv(csvfile, header=0)
        for index, row in df.iterrows():
            document = row['document']
            if len(document.split(' '))< 700:
                doc_chunks.append({"document": document})
                continue
            chunks = get_splitter(document)
            for ch in chunks:
                if(len(ch.split(' '))) < 70:
                    continue
                print(len(ch.split(' ')))
                doc_chunks.append({"document":ch})

    return doc_chunks


def write_output(out_file, doc_chunks):
   with open(out_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["document"])
        writer.writeheader()
        writer.writerows(doc_chunks)



file = sys.argv[1]
folder, filename = os.path.split(file)
print(file)
 
doc_chunks = read_input(filename)
out_file = "test_" + filename
write_output(out_file, doc_chunks)


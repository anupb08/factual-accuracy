import pandas as pd
import re
import sys
import os

#csv_file = "fin_report_small_doc_falconsai.csv"
#csv_file = "fin_report_small_doc_facebook_cnn.csv"
csv_file = "bbc-business-news-summary.csv"
csv_file = "liquidity_segment_0_input_2_1000_subset.csv"
csv_file = "t_rag.csv"
csv_file = "billsum_rag.csv"
csv_file = "bbcSummary.csv"
csv_file = "liquiditySummary_rag.csv"
csv_file = "trainLiquiditySummary.csv"
csv_file = "liquiditySummary_dynamic_rag.csv"
csv_file = "kdave-Indian_Financial_News1_rag.csv"
#csv_file = "Liquidity_summary_dynamics_rag.csv"
#csv_file = "Liquidity_summary_dynamics_error_corrected_rag.csv"
#csv_file = "kdave-Indian_Financial_News1_rag_mukaj.csv"

# #to rename a dataframe header
# df2 = pd.read_csv("somepath.csv",usecols=[3,9],header=0)
# df2.rename(columns={'Original_title_for_context_here':'context','Original_title_for_summary_here':'summary'},inplace=True)


# +
def mask_numericals(statement):
    #Define regex pattern
    pattern = r'[+-]?\d{1,9}(?:,\d{3})*(?:\.\d+)?|\.\d+'
    
    # List to store intermediate sentences 
    masked_sentences = []
    matches = []
    
    #Function to replace one match at a time
    def replace_one_match(match, current_statement):
        #Replace the current match with [NUM]
        start, end = match.span()
        return current_statement[:start] + '[NUM]' + current_statement[end:]
    
    original_statement = statement
    for match in re.finditer(pattern, original_statement):
        statement = replace_one_match(match, original_statement)
        #print(statement)
        if len(statement) < 50:
            continue
        matches.append(match.group())
        masked_sentences.append(statement)
        
    def extract_sentence_with_num(paragraph):
        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        # Find the sentence that contains "[NUM]"
        for sentence in sentences:
            if "[NUM]" in sentence:
                return sentence
        return None
    
    for i in range(len(masked_sentences)):
        masked_sentences[i] = extract_sentence_with_num(masked_sentences[i])
        
    return masked_sentences, matches

db = pd.DataFrame(columns=['doc_no', 'summ_no','prev','source','masked','target'])
#db = pd.DataFrame(columns=['source','masked','num'])


file = sys.argv[1]
folder, filename = os.path.split(file)
print(file)


df = pd.read_csv(file, header=0, keep_default_na=False)
for index, row in df.iterrows():
    #value = row['document']
    #summary = row['summary']
    doc_no = row["doc_no"]
    sum_no = row["summ_no"]
    value = row['doc_chunk']
    summary = row['sentence']
    prev_chunk = row['prev_chunk']
    masked_list,num_list = mask_numericals(summary)
    for i in range(len(masked_list)):
        db.loc[len(db)] = [doc_no, sum_no, str(prev_chunk), value, masked_list[i], str(num_list[i])]
        #db.loc[len(db)] = [value,masked_list[i],str(num_list[i])]
        

db.to_csv('testdata/'+filename, index=True)

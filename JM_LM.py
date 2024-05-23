from helper_tasks import *
import math
import os

# Task 02: KM_LM (Jelinek-Mercer based Language Model)
# calculate jm_lm probability for a given query term
def jm_lm(coll, query_term_freq, df):
    jm_lm_scores = {}
    lambda_param=0.4

    #data_cx = collection_length is the total number of word occurrences in collection
    data_cx = sum(rcv1_doc.getDocLen() for rcv1_doc in coll.values())

    for doc_id, rcv1_doc in coll.items():
        doc_len = rcv1_doc.getDocLen() # get Document length
        score = 0.0

        # Get the product value for each query_term in query_term_freq
        for term, qfi in query_term_freq.items():
            fqi = rcv1_doc.terms.get(term, 0) # number of time query term occurs in the document
            cqi = df.get(term, 0) #number of times query term occurs in document collection
            if doc_len > 0 and data_cx > 0:
                score += math.log10((1 - lambda_param)* (fqi / doc_len)) + (lambda_param * (cqi / data_cx))
        
        jm_lm_scores[doc_id] = score

    return jm_lm_scores

#Task 04: save jm_lm ranked files per query 
def save_jmlm_ranking_to_file(jm_lm_scores, query_num):
    output_file = f"RankingOutputs_JM_LM/JM_LM_{query_num}Ranking.dat"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for doc_id, score in sorted(jm_lm_scores.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{doc_id} {score}\n")

def print_jmlm_top_documents(query_num):
    output_file = f"RankingOutputs_JM_LM/JM_LM_{query_num}Ranking.dat"
    with open(output_file, 'r') as f:
        lines = f.readlines()
        sorted_docs = sorted(lines, key=lambda x: float(x.split(' ')[1]), reverse=True)
        top_documents = sorted_docs[:15]

        print(f"Top 15 Documents for {query_num} (DocID Weight):")
        for doc_score in top_documents:
            doc_id, score = doc_score.strip().split(' ')
            print(f"{doc_id} {score}\n")

if __name__ == '__main__':
    query_file_path = 'the50Queries.txt'
    input_path, stopwordList = get_input_path_and_stopword_list() #get stopword file as a list and input filepath for docs
    queries_dict = parse_queries(query_file_path, stopwordList, 'JM_LM') #get queries and their frequency for JM_LM method 

    # for each query title in the50Queries.txt, calculate the jm_lm score
    for query_num, query_terms in queries_dict.items():
        collection_num = ''.join(c for c in query_num if c.isdigit()) #get corresponding collection for the query number
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}") # create a collection of Rcv1Doc object for the corresponding coll
        df = my_df(coll) # get a dictionary of {term: doc_fre} : how many documents contains the term 
        jm_lm_scores = jm_lm(coll, query_terms, df) # calculate jm_lm probabaility for given query term
        save_jmlm_ranking_to_file(jm_lm_scores, query_num.split(':')[-1].strip()) #save the ranked file per query term
        print_jmlm_top_documents(query_num.split(':')[-1].strip())

    # Test for only one query R107
    query_num = 'Number: R107'
    if query_num in queries_dict:
        query_terms = queries_dict[query_num]
        collection_num = ''.join(c for c in query_num if c.isdigit())  # Get corresponding collection for the query number
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}")  # Create a collection of Rcv1Doc objects for the corresponding coll
        df = my_df(coll)  # Get a dictionary of {term: doc_fre} : how many documents contain the term
        jm_lm_scores = jm_lm(coll, query_terms, df)  # Calculate JM_LM probability for given query term
        save_jmlm_ranking_to_file(jm_lm_scores, query_num.split(':')[-1].strip())  # Save the ranked file per query term
        print_jmlm_top_documents(query_num.split(':')[-1].strip())
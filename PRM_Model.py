from helper_tasks import *
import math
import os
from BM25 import *

# Task 3: My_PRM (Pseudo-Relevance Model)
# first, use BM25 to rank the documents based on the initial query
# Then, use Rocchio's algorithm to adjust the query vector using the top k(10) documents from the initial BM25 ranking
# Then, re-rank the documents again using the updated query vector and BM25 scores.
def my_prm(coll, query_terms, df):
    # Initially, calculating bm25 scores for long queries(with title, desc & narratives)
    bm25_scores = my_bm25(coll, query_terms, df) 
        
    # sort the bm25 scores in descending order to get the relevant and nonrelevant docs 
    sorted_scores_desc = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True) 

    # with pseudo-relevance feedback technique, we assumed that the 10 top-ranked documents are relevant and last 10 are non-relevant
    relevant_documents = {docid: vector_normalize(coll[docid]) for docid, score in sorted_scores_desc[:10]}
    nonrelevant_documents = {docid: vector_normalize(coll[docid]) for docid, score in sorted_scores_desc[-10:]}

    # Using Rocchio's algorithm and the top-ranked documents to refine the query vector
    updated_query_vector = rocchios_algorithm(vector_normalize(query_terms), relevant_documents, nonrelevant_documents)

    # Re-rank documents again using the refined query vector.
    prm_scores = my_bm25(coll, updated_query_vector, df)

    return prm_scores


# This is the Rocchio's Algorithm developed based on pseudo-relevance 
# Rocchio's Algorithm: Maximizes the difference between the average vector representing 
# the relevant documents and the average vector representing the non-relevant documents.
def rocchios_algorithm(vactorized_query, relevant_docs, nonrelevant_docs):
    alpha = 1
    beta = 4
    gamma = 0.1

    centroid_for_relevant_docs = {term: 0 for term in vactorized_query}
    centroid_for_nonrelevant_docs = {term: 0 for term in vactorized_query}

    for doc in relevant_docs.values():
        for term, freq in doc.items():
            if term in centroid_for_relevant_docs:
                centroid_for_relevant_docs[term] += freq

    for doc in nonrelevant_docs.values():
        for term, freq in doc.items():
            if term in centroid_for_nonrelevant_docs:
                centroid_for_nonrelevant_docs[term] += freq

    for term in centroid_for_relevant_docs:
        centroid_for_relevant_docs[term] /= len(relevant_docs)

    for term in centroid_for_nonrelevant_docs:
        centroid_for_nonrelevant_docs[term] /= len(nonrelevant_docs)

    centroid_for_relevant_docs = vector_normalize(centroid_for_relevant_docs)
    centroid_for_nonrelevant_docs = vector_normalize(centroid_for_nonrelevant_docs)

    updated_query = {}
    for term in vactorized_query:
        updated_query[term] = alpha * vactorized_query[term] + beta * centroid_for_relevant_docs[term] - gamma * centroid_for_nonrelevant_docs[term]
    return updated_query

# Use to tune the Rocchio model
def vector_normalize(vector):
    len = sum([v*v for v in vector.values()])
    if len == 0:
        return vector
    len = math.sqrt(len)
    return {key: val/len for key, val in vector.items()}

#Task 04: save my_prm ranked scores per query 
def save_prm_ranking_to_file(jm_lm_scores, query_num):
    output_file = f"RankingOutputs_MY_PRM/MY_PRM_{query_num}Ranking.dat"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for doc_id, score in sorted(jm_lm_scores.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{doc_id} {score}\n")

def print_prm_top_documents(query_num):
    output_file = f"RankingOutputs_MY_PRM/MY_PRM_{query_num}Ranking.dat"
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
    # Get input_path for .xml files and get stopwordList by calling method defined in Question01
    input_path, stopwordList = get_input_path_and_stopword_list()

    queries_dict = parse_queries(query_file_path, stopwordList, 'PRM') #get long-queries(title, desc & narratives) for PRM model

    for query_num, query_terms in queries_dict.items():
        collection_num = ''.join(c for c in query_num if c.isdigit())  # Extract the collection number from the query number
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}")
        df = my_df(coll) # Get a dictionary of {term: doc_fre} : how many documents contain the term

        prm_model_scores = my_prm(coll, query_terms, df)
        save_prm_ranking_to_file(prm_model_scores, query_num)
        print_prm_top_documents(query_num)

    # Test for only one query R107
    query_num = 'Number: R107'
    if query_num in queries_dict:
        query_terms = queries_dict[query_num]
        collection_num = ''.join(c for c in query_num if c.isdigit())  # Get corresponding collection for the query number
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}")  # Create a collection of Rcv1Doc objects for the corresponding coll
        df = my_df(coll)  # Get a dictionary of {term: doc_fre} : how many documents contain the term
        jm_lm_scores = my_prm(coll, query_terms, df)  # Calculate JM_LM probability for given query term
        save_prm_ranking_to_file(jm_lm_scores, query_num.split(':')[-1].strip())  # Save the ranked file per query term
        print_prm_top_documents(query_num.split(':')[-1].strip())
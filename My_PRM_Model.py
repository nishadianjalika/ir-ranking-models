from helper_tasks import *
import math
import os
from BM25_Model import *

# Task 3: My_PRM (Pseudo-Relevance Model)
# first, use BM25 to rank the documents based on the initial query
# Then, use Rocchio's algorithm to adjust the query vector using the top k(10) documents from the initial BM25 ranking
# Then, re-rank the documents again using the updated query vector and BM25 scores.
def my_prm(coll, query_terms, df, alpha, beta, gamma):
    # Initially, calculating bm25 scores for long queries(with title, desc & narratives)
    bm25_scores = my_bm25(coll, query_terms, df) 
        
    # sort the bm25 scores in descending order to get the relevant and nonrelevant docs 
    sorted_scores_desc = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True) 

    # with pseudo-relevance feedback technique, we assumed that the 10 top-ranked documents are relevant and last 10 are non-relevant
    relevant_documents = {docid: vector_normalize(coll[docid].terms) for docid, score in sorted_scores_desc[:15]}
    nonrelevant_documents = {docid: vector_normalize(coll[docid].terms) for docid, score in sorted_scores_desc[-15:]}

    # Using Rocchio's algorithm and the top-ranked documents to refine the query vector
    updated_query_vector = rocchios_algorithm(vector_normalize(query_terms), relevant_documents, nonrelevant_documents, alpha, beta, gamma)

    # Re-rank documents again using the refined query vector.
    prm_scores = my_bm25(coll, updated_query_vector, df)

    return prm_scores


# This is the Rocchio's Algorithm developed based on pseudo-relevance 
# Rocchio's Algorithm: Maximizes the difference between the average vector representing 
# the relevant documents and the average vector representing the non-relevant documents.
def rocchios_algorithm(vactorized_query, relevant_docs, nonrelevant_docs, alpha, beta, gamma):
    # alpha = 1 #8 #1 #Controls the weight of the original query vector.
    # beta = 0.75 #16 #4 #Controls the weight of the centroid of relevant documents
    # gamma = 0.15 #4 #0.1 #Controls the weight of the centroid of non-relevant documents

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
# def vector_normalize(vector):
#     len = sum([v*v for v in vector.values()])
#     if len == 0:
#         return vector
#     len = math.sqrt(len)
#     return {key: val/len for key, val in vector.items()}

def vector_normalize(vector):
    length = math.sqrt(sum([v*v for v in vector.values()]))
    if length == 0:
        return vector
    return {term: freq / length for term, freq in vector.items()}

#Task 04: save my_prm ranked scores per query 
def save_prm_ranking_to_file(jm_lm_scores, query_num):
    output_file = f"RankingOutputs_MY_PRM/MY_PRM_{query_num}Ranking.dat"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for doc_id, score in sorted(jm_lm_scores.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{doc_id} {score}\n")

def print_prm_top_documents(query_num, log_file):
    output_file = f"RankingOutputs_MY_PRM/MY_PRM_{query_num}Ranking.dat"
    with open(output_file, 'r') as f:
        lines = f.readlines()
        sorted_docs = sorted(lines, key=lambda x: float(x.split(' ')[1]), reverse=True)
        top_documents = sorted_docs[:15]

        output = []
        output.append(f"Top 15 Documents for {query_num} (DocID Weight):")
        for doc_score in top_documents:
            doc_id, score = doc_score.strip().split(' ')
            output.append(f"{doc_id} {score}\n")
        
        print("\n".join(output))

        # Write to the log file
        with open(log_file, 'a') as log_f:
            log_f.write("\n".join(output))
            log_f.write("\n\n")

if __name__ == '__main__':
    query_file_path = 'the50Queries.txt'
    # Get input_path for .xml files and get stopwordList by calling method defined in Question01
    input_path, stopwordList = get_input_path_and_stopword_list()

    queries_dict = parse_queries(query_file_path, stopwordList, 'PRM') #get long-queries(title, desc & narratives) for PRM model

    for query_num, query_terms in queries_dict.items():
        collection_num = ''.join(c for c in query_num if c.isdigit())  # Extract the collection number from the query number
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}")
        df = my_df(coll) # Get a dictionary of {term: doc_fre} : how many documents contain the term

        alpha = 1 #8 #1 #Controls the weight of the original query vector.
        beta = 4 #16 #4 #Controls the weight of the centroid of relevant documents
        gamma = 0.01 #4 #0.1 #Controls the weight of the centroid of non-relevant documents

        prm_model_scores = my_prm(coll, query_terms, df, alpha, beta, gamma)

        log_file = "output_file_MyPRM_Ranking_Docs_top15"
        save_prm_ranking_to_file(prm_model_scores, query_num.split(':')[-1].strip())
        print_prm_top_documents(query_num.split(':')[-1].strip(), log_file)

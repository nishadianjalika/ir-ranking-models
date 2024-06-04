from helper_tasks import *
import math
import os
from BM25_Model import *
from typing import Counter

# Task 3: My_PRM (Pseudo-Relevance Model)
# first, use BM25 to rank the documents based on the initial query
# Then, use Rocchio's algorithm to adjust the query vector using the top k(10) documents from the initial BM25 ranking
# Then, re-rank the documents again using the updated query vector and BM25 scores.
def my_prm(coll, query_terms, df, alpha, beta, gamma):
    prm_scores_dict = {}
    # Initially, calculating bm25 scores for long queries(with title, desc & narratives)
    bm25_scores = my_bm25(coll, query_terms, df) 
        
    # sort the bm25 scores in descending order to get the relevant and nonrelevant docs 
    sorted_scores_desc = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True) 

    # with pseudo-relevance feedback technique, we assumed that the 10 top-ranked documents are relevant and last 10 are non-relevant
    relevant_documents = {docid: vector_normalize(coll[docid].terms) for docid, score in sorted_scores_desc[:30]}
    nonrelevant_documents = {docid: vector_normalize(coll[docid].terms) for docid, score in sorted_scores_desc[-5:]}

    # Using Rocchio's algorithm and the top-ranked documents to refine the query vector
    updated_query_vector = rocchios_algorithm(vector_normalize(query_terms), coll, relevant_documents, nonrelevant_documents, alpha, beta, gamma)
    
    sorted_query = sorted(updated_query_vector.items(), key=lambda x: x[1], reverse=True)
    top_terms = dict(sorted_query[:3])  # Keep only top 10 terms

    expanded_query_terms = {}
    for term, fre in top_terms.items():
        query_terms[term] = query_terms.get(term, 0) + 1

    # Re-rank documents again using the refined query vector.
    # prm_scores_dict = my_bm25(coll, query_terms, df)

    for doc_id in bm25_scores:
        prm_scores = 0
        for term in query_terms:
            if term in coll[doc_id].terms:
                prm_scores += bm25_scores.get(doc_id, 0)
        prm_scores_dict[doc_id] = prm_scores

    return prm_scores_dict


# This is the Rocchio's Algorithm developed based on pseudo-relevance 
# Rocchio's Algorithm: Maximizes the difference between the average vector representing 
# the relevant documents and the average vector representing the non-relevant documents.
def rocchios_algorithm_old(vactorized_query, relevant_docs, nonrelevant_docs, alpha, beta, gamma):
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

def rocchios_algorithm(query_terms, coll, relevant_docs, nonrelevant_docs, alpha, beta, gamma):
    rel_doc_vectors = [vector_normalize(coll[doc_id].terms) for doc_id in relevant_docs]
    nonrel_doc_vectors = [vector_normalize(coll[doc_id].terms) for doc_id in nonrelevant_docs]

    # Compute centroid of relevant and non-relevant document vectors
    centroid_rel = compute_centroid(rel_doc_vectors)
    centroid_nonrel = compute_centroid(nonrel_doc_vectors)

    # Adjust the query vector
    updated_query = {}
    all_terms = set(query_terms.keys()).union(set(centroid_rel.keys()), set(centroid_nonrel.keys()))
    for term in all_terms:
        query_val = query_terms.get(term, 0)
        rel_val = centroid_rel.get(term, 0)
        nonrel_val = centroid_nonrel.get(term, 0)
        updated_query[term] = alpha * query_val + beta * rel_val - gamma * nonrel_val

    # Optionally, you could normalize the updated query here
    updated_query = vector_normalize(updated_query)
    return updated_query

def compute_centroid(doc_vectors):
    centroid = {}
    for vector in doc_vectors:
        for term, value in vector.items():
            centroid[term] = centroid.get(term, 0) + value / len(doc_vectors)
    return centroid

def vector_normalize(vector):
    norm = math.sqrt(sum(v**2 for v in vector.values()))
    return {term: (v / norm if norm else 0) for term, v in vector.items()}

def vector_normalize(vector):
    length = math.sqrt(sum([v*v for v in vector.values()]))
    if length == 0:
        return vector
    return {term: freq / length for term, freq in vector.items()}

#Task 04: save my_prm ranked scores per query 
def save_prm_ranking_to_file(prm_scores, query_num):
    output_file = f"RankingOutputs_MY_PRM/MY_PRM_{query_num}Ranking.dat"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for doc_id, score in sorted(prm_scores.items(), key=lambda item: item[1], reverse=True):
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

    queries_dict = parse_queries(query_file_path, stopwordList, 'BM25') #get long-queries(title, desc & narratives) for PRM model

    for query_num, query_terms in queries_dict.items():
        collection_num = ''.join(c for c in query_num if c.isdigit())  # Extract the collection number from the query number
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}")
        df = my_df(coll) # Get a dictionary of {term: doc_fre} : how many documents contain the term

        alpha = 1 #8 #1 #Controls the weight of the original query vector.
        beta = 0.75 #16 #4 #Controls the weight of the centroid of relevant documents
        gamma = 0.1 #4 #0.1 #Controls the weight of the centroid of non-relevant documents

        prm_model_scores = my_prm(coll, query_terms, df, alpha, beta, gamma)

        log_file = "output_file_MyPRM_Ranking_Docs_top15"
        save_prm_ranking_to_file(prm_model_scores, query_num.split(':')[-1].strip())
        print_prm_top_documents(query_num.split(':')[-1].strip(), log_file)

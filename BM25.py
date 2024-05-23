from helper_tasks import *
import math
import os

# This is the function to calculate BM25 score for a given document and query
# df is the document frequency dictionary
# it returns a dictionary with key docid and BM25 score for each document a value
def my_bm25(coll, query_term_freq, df):

    k1 = 1.2
    k2 = 500
    b = 0.75
    r = R = 0
    bm25_scores = {} # initialize an empty dictionary to store scores
    N = len(coll)  # Total number of documents
    avdl = avg_length(coll)  # Average document length calculated using avg_length() function

    # Parse the query to get term frequency using parse_query() method in Question01
    # query_term_freq = parse_query(q, stopwordList)

    # Calculate BM25 score for each rcv1_doc document in coll collection
    for doc_id, rcv1_doc in coll.items():
        dl = rcv1_doc.getDocLen()  # Document length
        K = k1 * ((1 - b) + ( b * (dl / avdl)))  # Calculate K for the document K = k1*((1-b) + b*dl /avdl)
        bm25_doc_score = 0

        # Get the sum for each query term 'term' and qfi is term frequency of the query
        for term, qfi in query_term_freq.items():
            ni = df.get(term, 0)  # Document frequency of term in the collection of documents
            fi = rcv1_doc.terms.get(term, 0)  # Term frequency in the document

            # # Calculate score for probabilistic argument component
            probabilistic_arg_numerator = (r + 0.5) / (R - r + 0.5)
            probabilistic_arg_denominator = (ni - r + 0.5) / (N - ni - R + r + 0.5)
            probabilistic_arg = probabilistic_arg_numerator / probabilistic_arg_denominator

            # Calculate score for experimental argument components
            experimental_validation_01 = ((k1 + 1) * fi) / (K + fi)
            experimental_validation_02 = ((k2 + 1) * qfi) / (k2 + qfi)

            # Calculate final BM25 score by multiplying three components and take log for base 10
            # and add it to bm25_doc_score
            bm25_doc_score += (1 + math.log10(probabilistic_arg)) * experimental_validation_01 * experimental_validation_02

        # Store the BM25 score against each documentId
        # bm25_scores[doc_id] = bm25_doc_score
        bm25_scores[doc_id] = max(bm25_doc_score, 0)

    return bm25_scores

def save_bm25_ranking_to_file(bm25_scores, query_num):
    output_file = f"RankingOutputs_BM25/BM25_{query_num}Ranking.dat"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for doc_id, score in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{doc_id} {score}\n")

def print_bm25_top_documents(query_num, log_file):
    output_file = f"RankingOutputs_BM25/BM25_{query_num}Ranking.dat"
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
    # Query file
    query_file_path = 'the50Queries.txt'
    # Get input_path for .xml files and get stopwordList by calling method defined in Question01
    input_path, stopwordList = get_input_path_and_stopword_list()
    log_file = "output_file_BM25_Ranking_Docs"
    queries_dict = parse_queries(query_file_path, stopwordList, 'BM25')

    for query_num, query_terms in queries_dict.items():
        collection_num = ''.join(c for c in query_num if c.isdigit())  # Extract the collection number from the query number
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}")
        df = my_df(coll)
        bm25_scores = my_bm25(coll, query_terms, df)
        save_bm25_ranking_to_file(bm25_scores, query_num.split(':')[-1].strip())
        print_bm25_top_documents(query_num.split(':')[-1].strip(), log_file)



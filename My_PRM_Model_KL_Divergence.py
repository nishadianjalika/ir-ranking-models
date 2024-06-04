import os
import math
from helper_tasks import *
from BM25_Model import *

def prm_kl(coll, initial_query, df, top_k=5, top_m_terms=5):
    # Perform initial BM25 retrieval
    initial_bm25_scores = my_bm25(coll, initial_query, df)
    # Get top_k documents
    top_docs = sorted(initial_bm25_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Collect term distributions from top_k documents
    term_freqs = {}
    total_terms = 0
    for doc_id, score in top_docs:
        doc = coll.get(doc_id)
        term_list = doc.get_term_list()
        for term_freq in term_list:
            term, freq = term_freq.split(':')
            freq = int(freq)
            total_terms += freq
            if term in term_freqs:
                term_freqs[term] += freq
            else:
                term_freqs[term] = freq
    
    # Convert term frequencies to probabilities
    term_probs = {term: freq / total_terms for term, freq in term_freqs.items()}
    
    # Calculate the background distribution
    background_freqs = my_df(coll)
    background_total_terms = sum(background_freqs.values())
    background_probs = {term: freq / background_total_terms for term, freq in background_freqs.items()}
    
    # Calculate KL Divergence for each term
    term_kl_scores = {term: kl_divergence([term_probs.get(term, 1e-10)], [background_probs.get(term, 1e-10)])
                      for term in term_probs.keys()}
    
    # Select top_m_terms based on KL Divergence scores
    expanded_terms = sorted(term_kl_scores.items(), key=lambda x: x[1], reverse=True)[:top_m_terms]
    expanded_query_terms = [term for term, score in expanded_terms]
    
    # Expand the initial query
    expanded_query = initial_query + ' ' + ' '.join(expanded_query_terms)
    # # Expand the initial query by updating frequencies of terms in the initial query
    # for term in expanded_query_terms:
    #     initial_query[term] = initial_query.get(term, 0) + 1
    return expanded_query

def kl_divergence(p, q):
    return sum(p[i] * math.log(p[i] / q[i]) for i in range(len(p)))

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

def main_kl_divergence():
    query_file_path = 'the50Queries.txt'
    input_path, stopwordList = get_input_path_and_stopword_list()
    queries_dict = parse_queries(query_file_path, stopwordList, 'PRM')

    with open(query_file_path, 'r') as f:
        query_text = f.read()
        titles = re.findall(r'<title>(.*?)<desc>', query_text, re.DOTALL)
        descriptions = re.findall(r'<desc>(.*?)<narr>', query_text, re.DOTALL)
        narratives = re.findall(r'<narr>(.*?)</top>', query_text, re.DOTALL)
        queries = [title.strip() for title in titles] 

    for i, query in enumerate(queries):
        collection_path = os.path.join("Data_Collection", f"Data_C{101 + i}")
        coll = parse_rcv1v2(stopwordList, collection_path)
        df = my_df(coll)

        # query_terms = query.split()
        # qfs = {}
        # for t in query_terms:
        #     term = stem(t.lower())
        #     try:
        #         qfs[term] += 1
        #     except KeyError:
        #         qfs[term] = 1

        kl_rankings_q = prm_kl(coll, query, df)
        kl_rankings = my_bm25(coll, kl_rankings_q, df)

        # Save rankings to file
        query_num = f"R{101 + i}"
        save_prm_ranking_to_file(kl_rankings, query_num)
        print_prm_top_documents(query_num, "output_file_KL_Divergence")

if __name__ == '__main__':
    input_path, stopwordList = get_input_path_and_stopword_list()
    main_kl_divergence()

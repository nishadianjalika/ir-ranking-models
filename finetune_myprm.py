import itertools
import pandas as pd
from helper_tasks import *
from My_PRM_Model import *
from BM25_Model import *
from Model_Evaluations import *

def rocchios_algorithm(vectorized_query, relevant_docs, nonrelevant_docs, alpha, beta, gamma):
    centroid_for_relevant_docs = {term: 0 for term in vectorized_query}
    centroid_for_nonrelevant_docs = {term: 0 for term in vectorized_query}

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
    for term in vectorized_query:
        updated_query[term] = alpha * vectorized_query[term] + beta * centroid_for_relevant_docs[term] - gamma * centroid_for_nonrelevant_docs[term]
    return updated_query

def compare_precision_prm( prm_precision):
    warnings.filterwarnings('ignore')

    # Concatenate the dataframes horizontally
    # precision_table = pd.concat([ prm_precision], axis=1)

    # Reset the index of the concatenated dataframe
    prm_precision = prm_precision.reset_index(drop=True)

    # Select columns 0, 1, 3, and 5
    # full_precision_table = precision_table.iloc[:, [0,1,3,5]]

    # Display the resulting dataframe
    print("----- Table 1: The performance of 3 models on average precision -----")
    # print(full_precision_table.head(151))

    average_values_precision = prm_precision.mean() 
    # print("average_values_precision(MAP)")
    # print(average_values_precision)

    # Create a new row with the average values
    average_row = pd.DataFrame([average_values_precision], columns=prm_precision.columns)

    # Concatenate the average row to the original DataFrame
    df_with_average = pd.concat([prm_precision, average_row], ignore_index=True)

    # Print the DataFrame with the average row
    df_with_average.iloc[:, 0] = df_with_average.iloc[:, 0].fillna("Average")
    # print(df_with_average)
    return df_with_average, average_values_precision

def evaluate_my_prm(alpha, beta, gamma):
    # Load stop words and input path
    input_path, stopwordList = get_input_path_and_stopword_list()

    # Parse the queries
    queries_dict = parse_queries('the50Queries.txt', stopwordList, 'PRM')

    data = []
    for query_num, query_terms in queries_dict.items():
        collection_num = ''.join(c for c in query_num if c.isdigit())
        coll = parse_rcv1v2(stopwordList, f"{input_path}Data_C{collection_num}")
        df = my_df(coll)
        prm_model_scores = my_prm(coll, query_terms, df, alpha, beta, gamma)
        # Save the ranking results for evaluation
        save_prm_ranking_to_file(prm_model_scores, query_num.split(':')[-1].strip())
    
    # Evaluate the model
    prm_precision, prm_precision_10, prm_DCG10 = model_evaluation("RankingOutputs_MY_PRM", "MY_PRM")
    _, avg_map = compare_precision_prm(prm_precision)
    # _, avg_prec_10 = compare_precision10(None, None, prm_precision_10)
    # _, avg_dcg_10 = compare_DCG(None, None, prm_DCG10)
    
    return avg_map['MY_PRM']

# Grid Search for Best Parameters
best_map = 0
best_alpha = 1
best_beta = 0.5
best_gamma = 0.1

alpha_values = [1,3,6,0.1]
beta_values = [4, 0.1]
gamma_values = [ 0.001, 0.005]

for alpha, beta, gamma in itertools.product(alpha_values, beta_values, gamma_values):
    avg_map = evaluate_my_prm(alpha, beta, gamma)
    if avg_map > best_map:
        best_map = avg_map
        best_alpha = alpha
        best_beta = beta
        best_gamma = gamma
    print(f"Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}, MAP: {avg_map}")

print(f"Best Parameters - Alpha: {best_alpha}, Beta: {best_beta}, Gamma: {best_gamma}")

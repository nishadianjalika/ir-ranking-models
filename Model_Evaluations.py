import pandas as pd
import re
import os
import math
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

### Task 5. Using 3 effectiveness measures to evaluate the three models
# Average precision (and MAP),
# Precision@10 (and their average)
# Discounted cumulative gain at rank position 10 (p = 10), DCG10 (and their average)

# Calculate and sort the precision in calculate_precision and calculate_precision_N functions
def calculate_metrics(dataset_id, dataset_ranking, dataset_feedback):
    ri = 0
    map1 = 0.0
    R = len([id for (id, v) in dataset_feedback[dataset_id].items() if v > 0]) #Used just for Recall
    for (n, id_) in sorted(dataset_ranking[dataset_id].items(), key=lambda x: int(x[0])):
        if dataset_feedback[dataset_id].get(id_, 0) > 0:
            ri += 1
            pi = round(float(ri) / float(int(n)), 4)            
            map1 += pi

    if ri > 0:
        map1 = map1 / float(ri)

    return f"R{dataset_id}", map1


# Process feedback file
def process_feedback():
    feedback_folder = "Feedback"
    dataset_feedback = {}

    for filename in os.listdir(feedback_folder):
        if filename.endswith(".txt"):
            dataset_id = re.search(r"\d+", filename).group()
            file_path = os.path.join(feedback_folder, filename)
            with open(file_path, "r") as feedback_file:
                lines = feedback_file.readlines()
                dataset_feedback[dataset_id] = {}
                for line in lines:
                    line = line.strip()
                    lineList = line.split() #['R101', '46547', '1']
                    dataset_feedback[dataset_id][lineList[1]] = float(lineList[2])

    return dataset_feedback #The structure is: {dataset ID, document ID, and relevance score}


# Calculate the total precision
def calculate_precision(folder_result, model):
    dataset_feedback = process_feedback()
    result_folder = folder_result
    dataset_ranking = {}
    data = []

    for dataset_folder in os.listdir(result_folder):
        dataset_path = os.path.join(result_folder, dataset_folder)
        if os.path.isdir(dataset_path):
            dataset_id = dataset_folder.split("Dataset")[1]
            dataset_ranking[dataset_id] = {}
            i = 1
            for filename in os.listdir(dataset_path):
                if filename.endswith(".dat"):
                    file_path = os.path.join(dataset_path, filename)
                    with open(file_path, "r") as result_file:
                        lines = result_file.readlines()
                        for line in lines:
                            line = line.strip()
                            line1 = line.split(":") #['46547', ' 4.859532662539241']
                            dataset_ranking[dataset_id][str(i)] = line1[0]
                            i += 1 # assigns the current position to each dataset_id in the ranking

            label, map1_score = calculate_metrics(dataset_id, dataset_ranking, dataset_feedback)
            data.append([label, map1_score])

    df = pd.DataFrame(data, columns=['Topic', model])
    return df


# Calculate the precision@12
def calculate_precision_N(folder_result, model):
    dataset_feedback = process_feedback()
    result_folder = folder_result
    dataset_ranking = {}
    data = []

    for dataset_folder in os.listdir(result_folder):
        dataset_path = os.path.join(result_folder, dataset_folder)
        if os.path.isdir(dataset_path):
            dataset_id = dataset_folder.split("Dataset")[1]
            dataset_ranking[dataset_id] = {}
            data_rank = {}
            for filename in os.listdir(dataset_path):
                if filename.endswith(".dat"):
                    file_path = os.path.join(dataset_path, filename)
                    with open(file_path, "r") as result_file:
                        lines = result_file.readlines()
                        for line in lines:
                            line = line.strip()
                            line1 = line.split(":") #['46547', ' 4.859532662539241']
                            data_rank[line1[0]] = float(line1[1])

            i = 1
            for (k, v) in sorted(data_rank.items(), key=lambda x: x[1], reverse=True): #score in descending
                dataset_ranking[dataset_id][str(i)] = k
                i += 1
                if i > 12:
                    break
            
            label, map1_score = calculate_metrics(dataset_id, dataset_ranking, dataset_feedback)
            data.append([label, map1_score])
    
    df = pd.DataFrame(data, columns=['Topic', model])
    return df


# Calculate the Discounted Community Gain 12
def calculate_DCGp(rel_scores, p):
    DCGp = rel_scores[0]  # Cumulative gain of the first document
    for i in range(2, min(p+1, len(rel_scores)+1)):
        reli = rel_scores[i-1]  # Relevance score of the document at position i
        DCGp += reli / (math.log2(i))
    return DCGp


def calculate_DCG(folder_result, model):
    dataset_feedback = process_feedback()
    dataset_ranking = {}
    result_folder = folder_result
    
    
    # calculates a list of relevance scores for the documents in the ranking based on the feedback
    def calculate_metrics_DCG(dataset_id, dataset_ranking, dataset_feedback):
        rel_scores = [1 if dataset_feedback[dataset_id].get(id_, 0) > 0 else 0 for id_ in dataset_ranking[dataset_id].values()]
        DCG12 = calculate_DCGp(rel_scores, 12)
        return round(DCG12, 4)

    DCG12_values = []
    for dataset_folder in os.listdir(result_folder):
        dataset_path = os.path.join(result_folder, dataset_folder)
        if os.path.isdir(dataset_path):
            dataset_id = dataset_folder.split("Dataset")[1]
            dataset_ranking[dataset_id] = {}
            data_rank = {}
            for filename in os.listdir(dataset_path):
                if filename.endswith(".dat"):
                    file_path = os.path.join(dataset_path, filename)
                    with open(file_path, "r") as result_file:
                        lines = result_file.readlines()
                        for line in lines:
                            line = line.strip()
                            line1 = line.split(":")
                            data_rank[line1[0]] = float(line1[1])

            i = 1
            for (k, v) in sorted(data_rank.items(), key=lambda x: x[1], reverse=True):
                dataset_ranking[dataset_id][str(i)] = k
                i += 1
                if i > 12:
                    break

            DCG12 = calculate_metrics_DCG(dataset_id, dataset_ranking, dataset_feedback)
            DCG12_values.append([f"R{dataset_id}", DCG12])

    df = pd.DataFrame(DCG12_values, columns=['Topic', model])
    return df

"""#### model_evaluation using three methods """
def model_evaluation(folder_result, model):
    precision = calculate_precision(folder_result, model)
    precision_10 = calculate_precision_N(folder_result, model)
    DCG10 = calculate_DCG(folder_result, model)
    return precision, precision_10, DCG10

"""#### 01. Precision"""
def compare_precision(df1_precision, df2_precision, df3_precision):
    warnings.filterwarnings('ignore')

    # Concatenate the dataframes horizontally
    concatenated_precision = pd.concat([df1_precision, df2_precision, df3_precision], axis=1)

    # Reset the index of the concatenated dataframe
    concatenated_precision = concatenated_precision.reset_index(drop=True)

    # Select columns 2 and 5
    df_precision = concatenated_precision.iloc[:, [0,1,3,5]]

    # Display the resulting dataframe
    print("----- Table 1: The performance of 3 models on average precision -----")
    #df_precision.head(151)

    average_values_precision = df_precision.mean() 

    # Create a new row with the average values
    average_row = pd.DataFrame([average_values_precision], columns=df_precision.columns)

    # Concatenate the average row to the original DataFrame
    df_with_average = pd.concat([df_precision, average_row], ignore_index=True)

    # Print the DataFrame with the average row
    df_with_average.iloc[:, 0] = df_with_average.iloc[:, 0].fillna("Average")
    return df_with_average, average_values_precision


"""#### 02. Precision@12"""
def compare_precision10(df1_precision_10, df2_precision_10, df3_precision_10):
    # 2. Precision12
    # Concatenate the dataframes horizontally
    concatenated_precision12 = pd.concat([df1_precision_10, df2_precision_10, df3_precision_10], axis=1)

    # Reset the index of the concatenated dataframe
    concatenated_precision12 = concatenated_precision12.reset_index(drop=True)

    # Select columns 2 and 5
    df_precision12 = concatenated_precision12.iloc[:, [0, 1,3,5]]

    # Display the resulting dataframe
    print("----- Table 2. The performance of 3 models on precision@12 -----")
    #df_precision12.head(151)

    average_values_precision12 = df_precision12.mean()

    # Create a new row with the average values
    average_row = pd.DataFrame([average_values_precision12], columns=df_precision12.columns)

    # Concatenate the average row to the original DataFrame
    df_with_average = pd.concat([df_precision12, average_row], ignore_index=True)

    # Print the DataFrame with the average row
    df_with_average.iloc[:, 0] = df_with_average.iloc[:, 0].fillna("Average")
    return df_with_average, average_values_precision12

def plot_average_val_to_compare(average_values_precision, average_values_precision12, average_values_DCG12):
    # Create a dictionary from the variables
    data = {
        'MAP': average_values_precision,
        'Average Precision 12': average_values_precision12,
        'Average DCG 12': average_values_DCG12
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    df

    # Plot Average Effectiveness Measures 
    fig = px.line(df)
    fig.update_traces(mode='lines+markers', marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(
        title='Effectiveness Measures',
        xaxis_title='Models',
        yaxis_title='Average Score',
        legend=dict(
            title='Metrics',
            orientation='v'
        )
    )
    fig.show()


"""# 03. DCG12"""
def compare_DCG(df1_DCG12, df2_DCG12, df3_DCG12):
    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Concatenate the dataframes horizontally
    concatenated_DCG12 = pd.concat([df1_DCG12, df2_DCG12, df3_DCG12], axis=1)

    # Reset the index of the concatenated dataframe
    concatenated_DCG12 = concatenated_DCG12.reset_index(drop=True)

    # Select columns 2 and 5
    df_DCG12 = concatenated_DCG12.iloc[:, [0, 1,3,5]]

    # Display the resulting dataframe
    print("----- Table 3: The performance of 3 models on DCG12 -----")
    #df_DCG12.head(151)

    average_values_DCG12 = df_DCG12.mean() 

    # Create a new row with the average values
    average_row = pd.DataFrame([average_values_DCG12], columns=df_DCG12.columns)

    # Concatenate the average row to the original DataFrame
    df_with_average = pd.concat([df_DCG12, average_row], ignore_index=True)

    # Print the DataFrame with the average row
    df_with_average.iloc[:, 0] = df_with_average.iloc[:, 0].fillna("Average")
    return df_with_average, average_values_DCG12


if __name__ == '__main__':

    # Call functions
    folder_result_BM25 = "RankingOutputs_BM25" #<============== INPUT YOUR FILE NAME
    folder_result_JMLM = "RankingOutputs_JM_LM" #<============== INPUT YOUR FILE NAME
    folder_result_MYPRM = "RankingOutputs_MY_PRM" #<============== INPUT YOUR FILE NAME
    
    bm25_model = "BM25"
    jmlm_model = "JM_LM"
    myprm_model = "MY_PRM"

    #Evaluate BM25 model results
    df1_precision, df1_precision_10, df1_DCG10 = model_evaluation(folder_result_BM25, bm25_model)
    #Evaluate JM_LM model results
    df2_precision, df2_precision_10, df2_DCG10 = model_evaluation(folder_result_JMLM, jmlm_model)
    #Evaluate MY_PRM model results
    df3_precision, df3_precision_10, df3_DCG10 = model_evaluation(folder_result_MYPRM, myprm_model)

    # compare_precision
    df_with_average, average_values_precision = compare_precision(df1_precision, df2_precision, df3_precision)
    # compare_precision10
    df_with_average_10, average_values_precision12 = compare_precision10(df1_precision_10, df2_precision_10, df3_precision_10)
    # compare_DCG
    df_with_average_DCG, average_values_DCG12 = compare_DCG(df1_DCG10, df2_DCG10, df3_DCG10)

    plot_average_val_to_compare(average_values_precision, average_values_precision12, average_values_DCG12)



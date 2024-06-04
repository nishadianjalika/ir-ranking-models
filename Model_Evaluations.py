import pandas as pd
import re
import os
import math
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import seaborn as sns

### Task 5. Using 3 effectiveness measures to evaluate the three models
# Average precision (and MAP),
# Precision@10 (and their average)
# Discounted cumulative gain at rank position 10 (p = 10), DCG10 (and their average)

# Calculate and sort the precision in calculate_precision and calculate_precision_N functions
def calculate_metrics(dataset_id, dataset_ranking, benchmark_dataset):
    ri = 0
    map1 = 0.0
    R = len([id for (id, v) in benchmark_dataset[dataset_id].items() if v > 0]) #Used just for Recall
    for (n, id_) in sorted(dataset_ranking[dataset_id].items(), key=lambda x: int(x[0])):
        if benchmark_dataset[dataset_id].get(id_, 0) > 0:
            ri += 1
            pi = round(float(ri) / float(int(n)), 4)            
            map1 += pi

    if ri > 0:
        map1 = map1 / float(ri)

    return f"R{dataset_id}", map1


# Process evaluation benchmark dataset
def process_benckmark_files():
    benchmark_folder = "EvaluationBenchmark"
    benchmark_dataset = {}

    for benchmark_file in os.listdir(benchmark_folder):
        if benchmark_file.endswith(".txt"):
            dataset_id = re.search(r"\d+", benchmark_file).group()
            file_path = os.path.join(benchmark_folder, benchmark_file)
            with open(file_path, "r") as benchmark_f:
                b_lines = benchmark_f.readlines()
                benchmark_dataset[dataset_id] = {}
                for line in b_lines:
                    line = line.strip()
                    lineList = line.split()
                    benchmark_dataset[dataset_id][lineList[1]] = float(lineList[2])
    
    #Returns a dictionary with the structure of {datasetID, docId, and relevanceScore}
    return benchmark_dataset 

# calculate precision values for the given folder where results are stored for then given model
# compare the generated results with benchmark given results
# returns a precision scores to all documents within the folder 
def calculate_precision_values(folder_of_model_result, modelname):
    benchmark_vals = process_benckmark_files()
    dataset_ranking = {}
    data = []

    # Iterarte through all the documents within the result folder by searching files with .dat
    for filename in os.listdir(folder_of_model_result):
        if filename.endswith(".dat"):
            dataset_id = re.search(r"R(\d+)", filename).group(1) #extract the dataset id
            dataset_ranking[dataset_id] = {}
            i = 1
            file_path = os.path.join(folder_of_model_result, filename)
            with open(file_path, "r") as result_file:
                lines = result_file.readlines()
                for line in lines:
                    line = line.strip()
                    doc_id, score = line.split()
                    dataset_ranking[dataset_id][str(i)] = doc_id
                    i += 1 # assigns the current position to each dataset_id in the ranking

            label, map1_score = calculate_metrics(dataset_id, dataset_ranking, benchmark_vals)
            data.append([label, map1_score])

    df = pd.DataFrame(data, columns=['Topic', modelname])
    return df

# Calculate the precision@10
def calculate_precision_of_N(folder_of_model_result, modelname, N=10):
    benchmark_vals = process_benckmark_files()
    dataset_ranking = {}
    data = []

    for filename in os.listdir(folder_of_model_result):
        if filename.endswith(".dat"):
            dataset_id = re.search(r"R(\d+)", filename).group(1)
            dataset_ranking[dataset_id] = {}
            data_rank = {}
            file_path = os.path.join(folder_of_model_result, filename)
            with open(file_path, "r") as result_file:
                lines = result_file.readlines()
                for line in lines:
                    line = line.strip()
                    doc_id, score = line.split()
                    data_rank[doc_id] = float(score)

            i = 1
            for (k, v) in sorted(data_rank.items(), key=lambda x: x[1], reverse=True): #score in descending
                dataset_ranking[dataset_id][str(i)] = k
                i += 1
                if i > N:
                    break
            
            label, map1_score = calculate_metrics(dataset_id, dataset_ranking, benchmark_vals)
            data.append([label, map1_score])
    
    df = pd.DataFrame(data, columns=['Topic', modelname])
    return df

# Calculate the Discounted Community Gain 10
def calculate_DCGp(rel_scores, p):
    DCGp = rel_scores[0]  # Cumulative gain of the first document
    for i in range(2, min(p+1, len(rel_scores)+1)):
        reli = rel_scores[i-1]  # Relevance score of the document at position i
        DCGp += reli / (math.log2(i))
    return DCGp

# calculates a list of relevance scores for the documents in the ranking based on the feedback
def calculate_metrics_DCG(dataset_id, dataset_ranking, benchmark_dataset):
    rel_scores = [1 if benchmark_dataset[dataset_id].get(id_, 0) > 0 else 0 for id_ in dataset_ranking[dataset_id].values()]
    DCG10 = calculate_DCGp(rel_scores, 10)
    return round(DCG10, 4)

def calculate_DCG(folder_result, model):
    dataset_feedback = process_benckmark_files()
    dataset_ranking = {}
    result_folder = folder_result

    DCG10_values = []
    for filename in os.listdir(result_folder):
        if filename.endswith(".dat"):
            dataset_id = re.search(r"R(\d+)", filename).group(1)
            dataset_ranking[dataset_id] = {}
            data_rank = {}
            file_path = os.path.join(result_folder, filename)
            with open(file_path, "r") as result_file:
                lines = result_file.readlines()
                for line in lines:
                    line = line.strip()
                    doc_id, score = line.split()
                    data_rank[doc_id] = float(score)

            i = 1
            for (k, v) in sorted(data_rank.items(), key=lambda x: x[1], reverse=True):
                dataset_ranking[dataset_id][str(i)] = k
                i += 1
                if i > 10:
                    break

            DCG10 = calculate_metrics_DCG(dataset_id, dataset_ranking, dataset_feedback)
            DCG10_values.append([f"R{dataset_id}", DCG10])

    df = pd.DataFrame(DCG10_values, columns=['Topic', model])
    return df

# Evaluate each given model using three matrics(precision, precision@10 and DCG10)
#returns a dataframe with calculated values per document 
def model_evaluation(folder_result, model):
    precision = calculate_precision_values(folder_result, model)
    precision_10 = calculate_precision_of_N(folder_result, model)
    DCG10 = calculate_DCG(folder_result, model)
    return precision, precision_10, DCG10

#### Combine Precision values and get the average value for three models
def compare_precision(bm25_precision, jmlm_precision, prm_precision):
    warnings.filterwarnings('ignore')

    # Concatenate the dataframes horizontally
    precision_table = pd.concat([bm25_precision, jmlm_precision, prm_precision], axis=1)

    # Reset the index of the concatenated dataframe
    precision_table = precision_table.reset_index(drop=True)

    # Select columns 0, 1, 3, and 5
    full_precision_table = precision_table.iloc[:, [0,1,3,5]]

    # Display the resulting dataframe
    print("----- Table 1: The performance of 3 models on average precision -----")
    # print(full_precision_table.head(151))

    average_values_precision = full_precision_table.mean() 
    # print("average_values_precision(MAP)")
    # print(average_values_precision)

    # Create a new row with the average values
    average_row = pd.DataFrame([average_values_precision], columns=full_precision_table.columns)

    # Concatenate the average row to the original DataFrame
    df_with_average = pd.concat([full_precision_table, average_row], ignore_index=True)

    # Print the DataFrame with the average row
    df_with_average.iloc[:, 0] = df_with_average.iloc[:, 0].fillna("Average")
    print(df_with_average)
    return df_with_average, average_values_precision


#### Combine Precision@10 values and get the average value for three models
def combine_precision10_for_models(bm25_precision_10, jmlm_precision_10, myprm_precision_10):
    # Concatenate the dataframes horizontally
    concatenated_precision10 = pd.concat([bm25_precision_10, jmlm_precision_10, myprm_precision_10], axis=1)

    # Reset the index of the concatenated dataframe
    concatenated_precision10 = concatenated_precision10.reset_index(drop=True)

    # Select columns 2 and 5
    df_precision10 = concatenated_precision10.iloc[:, [0, 1,3,5]]

    # Display the resulting dataframe
    print("----- Table 2. The performance of 3 models on precision@10 -----")
    # print(df_precision10.head(151))

    average_values_precision10 = df_precision10.mean()
    # print("Average: precision10")
    # print(average_values_precision10)

    # Create a new row with the average values
    average_row = pd.DataFrame([average_values_precision10], columns=df_precision10.columns)

    # Concatenate the average row to the original DataFrame
    df_with_average = pd.concat([df_precision10, average_row], ignore_index=True)

    # Print the DataFrame with the average row
    df_with_average.iloc[:, 0] = df_with_average.iloc[:, 0].fillna("Average")
    print(df_with_average)
    return df_with_average, average_values_precision10

def plot_average_val_to_compare(average_values_precision, average_values_precision10, average_values_DCG10):
    # Create a dictionary from the variables
    data = {
        'MAP': average_values_precision,
        'Average Precision 10': average_values_precision10,
        'Average DCG 10': average_values_DCG10
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

## Plotting bar graph to compare the results visually for three models with three matrices used
def plot_bar_chart_comparison(average_values_precision, average_values_precision10, average_values_DCG10):
    # Create a DataFrame from the average values
    df = pd.DataFrame({
        'Metric': ['MAP', 'Average Precision 10', 'Average DCG 10'],
        'BM25': [average_values_precision['BM25'], average_values_precision10['BM25'], average_values_DCG10['BM25']],
        'JM_LM': [average_values_precision['JM_LM'], average_values_precision10['JM_LM'], average_values_DCG10['JM_LM']],
        'MY_PRM': [average_values_precision['MY_PRM'], average_values_precision10['MY_PRM'], average_values_DCG10['MY_PRM']]
    })

    # Melt the DataFrame to long format
    df_melted = df.melt(id_vars='Metric', var_name='Model', value_name='Score')

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model')
    plt.title('Comparison of Models on Different Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Average Score')
    plt.legend(title='Model')
    plt.show()

### Combine DCG10 values and get the average value for three models
def combine_DCG_for_models(bm25_DCG10, jmlm_DCG10, myprm_DCG10):
    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Concatenate the dataframes horizontally
    concatenated_DCG10 = pd.concat([bm25_DCG10, jmlm_DCG10, myprm_DCG10], axis=1)

    # Reset the index of the concatenated dataframe
    concatenated_DCG10 = concatenated_DCG10.reset_index(drop=True)

    # Select columns 2 and 5
    df_DCG10 = concatenated_DCG10.iloc[:, [0, 1,3,5]]

    # Display the resulting dataframe
    print("----- Table 3: The performance of 3 models on DCG10 -----")
    # print(df_DCG10.head(151))

    average_values_DCG10 = df_DCG10.mean() 
    # print("Average: DCG10")
    # print(average_values_DCG10)

    # Create a new row with the average values
    average_row = pd.DataFrame([average_values_DCG10], columns=df_DCG10.columns)

    # Concatenate the average row to the original DataFrame
    df_with_average = pd.concat([df_DCG10, average_row], ignore_index=True)

    # Print the DataFrame with the average row
    df_with_average.iloc[:, 0] = df_with_average.iloc[:, 0].fillna("Average")
    print(df_with_average)
    return df_with_average, average_values_DCG10




if __name__ == '__main__':
    #Evaluate BM25 model results
    bm25_precision, bm25_precision_10, bm25_DCG10 = model_evaluation("RankingOutputs_BM25", "BM25")
    
    #Evaluate JM_LM model results
    jmlm_precision, jmlm_precision_10, jmlm_DCG10 = model_evaluation("RankingOutputs_JM_LM" , "JM_LM")
    
    #Evaluate MY_PRM model results
    prm_precision, prm_precision_10, prm_DCG10 = model_evaluation("RankingOutputs_MY_PRM", "MY_PRM")

    # print(f"{bm25_precision} , { jmlm_precision}, {prm_precision}")
    
    # compare_precision
    df_with_average, average_values_precision = compare_precision(bm25_precision, jmlm_precision, prm_precision)
    
    # compare_precision10
    df_with_average_10, average_values_precision10 = combine_precision10_for_models(bm25_precision_10, jmlm_precision_10, prm_precision_10)
    
    # compare_DCG
    df_with_average_DCG, average_values_DCG10 = combine_DCG_for_models(bm25_DCG10, jmlm_DCG10, prm_DCG10)

    # plot_average_val_to_compare(average_values_precision, average_values_precision10, average_values_DCG10)
    # Plot the comparison using bar chart
    plot_bar_chart_comparison(average_values_precision, average_values_precision10, average_values_DCG10)



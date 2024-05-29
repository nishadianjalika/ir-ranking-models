from Model_Evaluations import *
import scipy.stats as stats


def perform_t_test(BM25_Model_eval, JM_LM_Model_eval, My_PRM_Model_eval ):
    
    jmlmVsBM25_Tstat, jmlmVsBM25_Pval = stats.ttest_ind(JM_LM_Model_eval, BM25_Model_eval, alternative='less')

    prmVsBM25_Tstat, prmVsBM25_Pval = stats.ttest_ind(My_PRM_Model_eval, BM25_Model_eval, alternative='less')

    jmlmVsPrm_Tstat, jmlmVsPrm_Pval = stats.ttest_ind(JM_LM_Model_eval, My_PRM_Model_eval, alternative='less')

    t_test_data = {'Model': ["JM_LM vs BM25", "MY_PRM vs BM25", "JM_LM vs MY_PRM"],
        'T-Statistic': [jmlmVsBM25_Tstat, prmVsBM25_Tstat, jmlmVsPrm_Tstat],
        'P-Value': [jmlmVsBM25_Pval, prmVsBM25_Pval, jmlmVsPrm_Pval ]
        
    }
    # Create a DataFrame from the dictionary
    t_test_results = pd.DataFrame(t_test_data)
    print (t_test_results)


if __name__ == '__main__':
    #Evaluate BM25 model results
    bm25_precision, bm25_precision_10, bm25_DCG10 = model_evaluation("RankingOutputs_BM25", "BM25")
    
    #Evaluate JM_LM model results
    jmlm_precision, jmlm_precision_10, jmlm_DCG10 = model_evaluation("RankingOutputs_JM_LM" , "JM_LM")
    
    #Evaluate MY_PRM model results
    prm_precision, prm_precision_10, prm_DCG10 = model_evaluation("RankingOutputs_MY_PRM", "MY_PRM")

    BM25_Model_prec = bm25_precision['BM25'].tolist()
    JM_LM_Model_prec = jmlm_precision['JM_LM'].tolist()
    My_PRM_Model_prec = prm_precision['MY_PRM'].tolist()       

    BM25_precision_10 = bm25_precision_10['BM25'].tolist()
    JMLM_precision10 = jmlm_precision_10['JM_LM'].tolist()
    MY_PRM_precision10 = prm_precision_10['MY_PRM'].tolist()

    BM25_DCG10 = bm25_DCG10['BM25'].tolist()
    JMLM_DCG10 = jmlm_DCG10['JM_LM'].tolist()
    MY_PRM_DCG10 = prm_DCG10['MY_PRM'].tolist()

    print("BM25, JMLM, MYPRM models t-test results for : Precision")
    perform_t_test(BM25_Model_prec, JM_LM_Model_prec, My_PRM_Model_prec)

    print("BM25, JMLM, MYPRM models t-test results for : Precision@10")
    perform_t_test(BM25_precision_10, JMLM_precision10, MY_PRM_precision10)

    print("BM25, JMLM, MYPRM models t-test results for : DCG10")
    perform_t_test(BM25_DCG10, JMLM_DCG10, MY_PRM_DCG10)
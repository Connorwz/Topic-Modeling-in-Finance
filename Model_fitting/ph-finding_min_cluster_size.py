import os
import pandas as pd 
from sentence_transformers import SentenceTransformer
from cuml.decomposition import PCA
from cuml.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import gc
from scipy.linalg import block_diag
print("libs are read")

# Set up part
def ph(min_cluster_size):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = PCA(n_components = 10)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size,metric = "euclidean", cluster_selection_method="eom",
                            gen_min_span_tree = True, prediction_data = False, min_samples = 40, verbose = True)
    vectorizer_model = CountVectorizer()
    Topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
                        calculate_probabilities = False,verbose = True)
    return Topic_model

def find_min_cluster_size(min_cluster_size,min_cluster_size_list,headlines,embeddings,target_num_cluster):
    Topic_model = ph(min_cluster_size)
    Topic_model.fit(headlines,embeddings)
    num_cluster = Topic_model.get_topic_info().shape[0]-1
    if num_cluster > (target_num_cluster + round(5*target_num_cluster/120)) or num_cluster < (target_num_cluster - round(5*target_num_cluster/120)):
        no_tuning = 0
        tried_min_cluster_size_list = []
    while (num_cluster > (target_num_cluster + round(5*target_num_cluster/120)) or num_cluster < (target_num_cluster - round(5*target_num_cluster/120))) and (min_cluster_size not in tried_min_cluster_size_list):
        no_tuning +=1
        tried_min_cluster_size_list.append(min_cluster_size)
        if num_cluster > (target_num_cluster + round(5*target_num_cluster/120)):
            if np.array(min_cluster_size_list)[np.array(min_cluster_size_list) > min_cluster_size].size >0:
                min_cluster_size = np.array(min_cluster_size_list)[np.array(min_cluster_size_list) > min_cluster_size][0]
                Topic_model = ph(min_cluster_size)
                Topic_model.fit(headlines,embeddings)
                num_cluster = Topic_model.get_topic_info().shape[0]-1
            else:
                print(f"after {no_tuning-1}th tuning, the final min_cluster_size is {min_cluster_size}")
                return min_cluster_size
        else:
            if np.array(min_cluster_size_list)[np.array(min_cluster_size_list) < min_cluster_size].size > 0:
                min_cluster_size = np.array(min_cluster_size_list)[np.array(min_cluster_size_list) < min_cluster_size][-1]
                Topic_model = ph(min_cluster_size)
                Topic_model.fit(headlines,embeddings)
                num_cluster = Topic_model.get_topic_info().shape[0]-1
            else:
                print(f"after {no_tuning-1}th tuning, the final min_cluster_size is {min_cluster_size}")
                return min_cluster_size
        print(f"after {no_tuning}th tuning, the min_cluster_size is {min_cluster_size}")
    return min_cluster_size

min_cluster_size_list = range(20,200,5)

def tr_te_split(headlines,df,embeddings,i=1):
    indices = np.arange(len(headlines))
    tr_ind, te_ind = train_test_split(indices, test_size=0.2, shuffle= True, random_state=i)
    tr_df = df.iloc[tr_ind,:]
    te_df = df.iloc[te_ind,:]
    tr_headlines = [headlines[ind] for ind in tr_ind]
    te_headlines = [headlines[ind] for ind in te_ind]
    tr_embeddings = embeddings[tr_ind,:]
    return tr_df,te_df,tr_headlines,te_headlines,tr_embeddings

print("set up finished")

df_folder = "/shared/share_tm-finance/Processed_df_Sentiment/One_year_window"
embeddings_folder = "/shared/share_tm-finance/Embeddings_with_Sentiment/One_year_window"
# saved_model_folder = "/shared/share_tm-finance/Stored_model/pcakmeans"
year_list = range(2021,2022)
num_clusters = 120
pos_min_cluster_size_list = []
neg_min_cluster_size_list = []
neu_min_cluster_size_list = []
pos_cluster_num_list = []
neg_cluster_num_list = []
neu_cluster_num_list = []
cluster_num_list = []
print("pre-computations finished")

for year in year_list:
    print(f"computation for {year} starts")
    torch.cuda.empty_cache()
    gc.collect()
    df = pd.read_csv(df_folder+f"/contem_{year}_senti.csv")
    headlines = df.vocab_con_headline.tolist()
    embeddings = np.load(embeddings_folder+f"/contem_{year}_senti_embeddings.npy")

    print(f"The df and embeddings in {year} finished")

    pos_indices = df[df['css'] > 0].index
    neg_indices = df[df['css'] < 0].index
    neu_indices = df[df['css'] == 0].index
    pos_df = df.iloc[pos_indices,:]
    neg_df = df.iloc[neg_indices,:]
    neu_df = df.iloc[neu_indices,:]
    pos_headlines = [headlines[ind] for ind in pos_indices]
    neg_headlines = [headlines[ind] for ind in neg_indices]
    neu_headlines = [headlines[ind] for ind in neu_indices]
    pos_embeddings = embeddings[pos_indices,:]
    neg_embeddings = embeddings[neg_indices,:]
    neu_embeddings = embeddings[neu_indices,:]

    #set pos_cluster_num, neg_cluster_num, neu_cluster_num based on the number of embeddings
    pos_cluster_num = int(num_clusters * len(pos_embeddings) / len(embeddings))
    neg_cluster_num = int(num_clusters * len(neg_embeddings) / len(embeddings))
    neu_cluster_num = int(num_clusters * len(neu_embeddings) / len(embeddings))
    diff = num_clusters - (pos_cluster_num + neg_cluster_num + neu_cluster_num)
    if pos_cluster_num < neg_cluster_num and pos_cluster_num < neu_cluster_num:
        pos_cluster_num += diff
    elif neg_cluster_num < pos_cluster_num and neg_cluster_num < neu_cluster_num:
        neg_cluster_num += diff
    else:
        neu_cluster_num += diff

    print(f"target pos_cluster_num in {year} is:", pos_cluster_num)
    print(f"target neg_cluster_num in {year} is:", neg_cluster_num)
    print(f"target neu_cluster_num in {year} is:", neu_cluster_num)

    pos_tr_df_tune, pos_te_df_tune, pos_tr_headlines_tune,pos_te_headlines_tune,pos_tr_embeddings_tune = tr_te_split(pos_headlines,pos_df,pos_embeddings)
    neg_tr_df_tune, neg_te_df_tune, neg_tr_headlines_tune,neg_te_headlines_tune,neg_tr_embeddings_tune = tr_te_split(neg_headlines,neg_df,neg_embeddings)
    neu_tr_df_tune, neu_te_df_tune, neu_tr_headlines_tune,neu_te_headlines_tune,neu_tr_embeddings_tune = tr_te_split(neu_headlines,neu_df,neu_embeddings)
    
    min_cluster_size_tune = 100
    pos_min_cluster_size = find_min_cluster_size(min_cluster_size_tune,min_cluster_size_list,pos_tr_headlines_tune,pos_tr_embeddings_tune,pos_cluster_num)
    neg_min_cluster_size = find_min_cluster_size(min_cluster_size_tune,min_cluster_size_list,neg_tr_headlines_tune,neg_tr_embeddings_tune,neg_cluster_num)
    neu_min_cluster_size = find_min_cluster_size(min_cluster_size_tune,min_cluster_size_list,neu_tr_headlines_tune,neu_tr_embeddings_tune,neu_cluster_num)
    print(f"The pos_min_cluster_size in {year} is {pos_min_cluster_size}")
    print(f"The neg_min_cluster_size in {year} is {neg_min_cluster_size}")
    print(f"The neu_min_cluster_size in {year} is {neu_min_cluster_size}")
    pos_min_cluster_size_list.append(pos_min_cluster_size) 
    neg_min_cluster_size_list.append(neg_min_cluster_size) 
    neu_min_cluster_size_list.append(neu_min_cluster_size) 
    pos_topic_model = ph(pos_min_cluster_size)
    pos_topic_model.fit(pos_tr_headlines_tune,pos_tr_embeddings_tune)
    pos_num_cluster = pos_topic_model.get_topic_info().shape[0]-1
    neg_topic_model = ph(neg_min_cluster_size)
    neg_topic_model.fit(neg_tr_headlines_tune,neg_tr_embeddings_tune)
    neg_num_cluster = neg_topic_model.get_topic_info().shape[0]-1
    neu_topic_model = ph(neu_min_cluster_size)
    neu_topic_model.fit(neu_tr_headlines_tune,neu_tr_embeddings_tune)
    neu_num_cluster = neu_topic_model.get_topic_info().shape[0]-1
    print(f"The real pos_num_cluster in {year} is {pos_num_cluster}")
    print(f"The real neg_num_cluster in {year} is {neg_num_cluster}")
    print(f"The real neu_num_cluster in {year} is {neu_num_cluster}")
    print(f"The real num_cluster in {year} is {pos_num_cluster+neg_num_cluster+neu_num_cluster}")
    pos_cluster_num_list.append(pos_num_cluster)
    neg_cluster_num_list.append(neg_num_cluster)
    neu_cluster_num_list.append(neu_num_cluster)
    cluster_num_list.append(pos_num_cluster+neg_num_cluster+neu_num_cluster)


print(f"The pos_min_cluster_size_list is {pos_min_cluster_size_list}")
print(f"The neg_min_cluster_size_list is {neg_min_cluster_size_list}")
print(f"The neu_min_cluster_size_list is {neu_min_cluster_size_list}")
print(f"The pos_cluster_num_list is {pos_cluster_num_list}")
print(f"The neg_cluster_num_list is {neg_cluster_num_list}")
print(f"The neu_cluster_num_list is {neu_cluster_num_list}")
print(f"The cluster_num_list is {cluster_num_list}")
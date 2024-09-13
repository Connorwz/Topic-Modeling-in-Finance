import os
import pandas as pd 
from sentence_transformers import SentenceTransformer
from cuml.decomposition import PCA
from cuml.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import gc
from scipy.linalg import block_diag
print("libs are read")

# Set up part
def pk(num_cluster):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = PCA(n_components = 10)
    hdbscan_model = KMeans(n_clusters = num_cluster)
    vectorizer_model = CountVectorizer()
    Topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
                        calculate_probabilities = False,verbose = True)
    return Topic_model

def tr_te_split(headlines,df,embeddings):
    indices = np.arange(len(headlines))
    tr_ind, te_ind = train_test_split(indices, test_size=0.2, shuffle= True, random_state=i)
    tr_df = df.iloc[tr_ind,:]
    te_df = df.iloc[te_ind,:]
    tr_headlines = [headlines[ind] for ind in tr_ind]
    te_headlines = [headlines[ind] for ind in te_ind]
    tr_embeddings = embeddings[tr_ind,:]
    return tr_df,te_df,tr_headlines,te_headlines,tr_embeddings

def assign_prob(topic_exposure_dist, topic_assignment):
    for i in range(topic_exposure_dist.shape[0]):
        if np.sum(topic_exposure_dist[i,:]) == 0:
            topic_exposure_dist[i,topic_assignment[i]] = 1
    return topic_exposure_dist

def linear_regression(tr_topic_dist,te_topic_dist,tr_df,te_df):
    tr_ret_topic_dist = pd.concat([tr_df.drop(columns = ["rp_entity_id","headline","vocab_con_headline","css"]),pd.DataFrame(tr_topic_dist)],axis = 1)
    tr_grouped = tr_ret_topic_dist.groupby(['date',"comnam","ret"])
    tr_grouped_sum = tr_grouped.sum()
    tr_X = np.array(tr_grouped_sum)
    tr_ret = [ind[2] for ind in list(tr_grouped_sum.index)]
    tr_Y = np.array(tr_ret).reshape(-1,1)
    tr_Y_mean = np.mean(tr_Y)
    regression = LinearRegression(fit_intercept=True)
    regression.fit(tr_X,tr_Y)
    tr_r2 = regression.score(tr_X,tr_Y)

    te_ret_topic_dist = pd.concat([te_df.drop(columns = ["rp_entity_id","headline","vocab_con_headline","css"]),pd.DataFrame(te_topic_dist)],axis = 1)
    te_grouped = te_ret_topic_dist.groupby(['date',"comnam","ret"])
    te_grouped_sum = te_grouped.sum()
    te_X = np.array(te_grouped_sum)
    te_ret = [ind[2] for ind in list(te_grouped_sum.index)]
    te_Y = np.array(te_ret).reshape(-1,1)
    te_Y_pred = regression.predict(te_X)
    te_sst = np.sum((te_Y-tr_Y_mean)**2)
    te_sse = np.sum((te_Y-te_Y_pred)**2)
    te_r2 = 1-te_sse/te_sst
    return tr_r2,te_r2
    
print("set up finished")

df_folder = "/shared/share_tm-finance/Processed_df_Sentiment/One_year_window"
embeddings_folder = "/shared/share_tm-finance/Embeddings_with_Sentiment/One_year_window"
# pos_saved_model_folder = "/shared/share_tm-finance/Stored_model/three_model/pcakmeans/pos"
# neg_saved_model_folder = "/shared/share_tm-finance/Stored_model/three_model/pcakmeans/neg"
# neu_saved_model_folder = "/shared/share_tm-finance/Stored_model/three_model/pcakmeans/neu"
year_list = range(2014,2024)
num_clusters = 120

com_tr_r2_dict = dict()
com_te_r2_dict = dict()
mean_com_tr_r2_dict = dict()
mean_com_te_r2_dict = dict()
sep_tr_r2_dict = dict()
sep_te_r2_dict = dict()
mean_sep_tr_r2_dict = dict()
mean_sep_te_r2_dict = dict()

print("pre-computations finished")

for year in year_list:
    print(f"computation for {year} starts")
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

    print(f"pos_cluster_num in {year} is:", pos_cluster_num)
    print(f"neg_cluster_num in {year} is:", neg_cluster_num)
    print(f"neu_cluster_num in {year} is:", neu_cluster_num)

    com_tr_r2_list = []
    com_te_r2_list = []
    sep_tr_r2_list = []
    sep_te_r2_list = []
    for i in range(1,6):
        print(f"The {i}th computation in {year} starts")
        torch.cuda.empty_cache()
        gc.collect()

        pos_tr_df, pos_te_df, pos_tr_headlines,pos_te_headlines,pos_tr_embeddings = tr_te_split(pos_headlines,pos_df,pos_embeddings)
        neg_tr_df, neg_te_df, neg_tr_headlines,neg_te_headlines,neg_tr_embeddings = tr_te_split(neg_headlines,neg_df,neg_embeddings)
        neu_tr_df, neu_te_df, neu_tr_headlines,neu_te_headlines,neu_tr_embeddings = tr_te_split(neu_headlines,neu_df,neu_embeddings)
        pos_tr_df.reset_index(drop=True,inplace=True)
        pos_te_df.reset_index(drop=True,inplace=True)
        neg_tr_df.reset_index(drop=True,inplace=True)
        neg_te_df.reset_index(drop=True,inplace=True)
        neu_tr_df.reset_index(drop=True,inplace=True)
        neu_te_df.reset_index(drop=True,inplace=True)
        pos_topic_model = pk(pos_cluster_num)
        neg_topic_model = pk(neg_cluster_num)
        neu_topic_model = pk(neu_cluster_num)
        pos_tr_topics, _ = pos_topic_model.fit_transform(pos_tr_headlines,pos_tr_embeddings)
        neg_tr_topics, _ = neg_topic_model.fit_transform(neg_tr_headlines,neg_tr_embeddings)
        neu_tr_topics, _ = neu_topic_model.fit_transform(neu_tr_headlines,neu_tr_embeddings)
        pos_te_topics, _ = pos_topic_model.transform(pos_te_headlines)
        neg_te_topics, _ = neg_topic_model.transform(neg_te_headlines)
        neu_te_topics, _ = neu_topic_model.transform(neu_te_headlines)
        # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # pos_topic_model.save(pos_saved_model_folder+f"/{year}_{pos_cluster_num}_{i}",
        #                      serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
        # neg_topic_model.save(neg_saved_model_folder+f"/{year}_{neg_cluster_num}_{i}",
        #                      serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
        # neu_topic_model.save(neu_saved_model_folder+f"/{year}_{neu_cluster_num}_{i}",
        #                      serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
        print(F"Model fitting for {i}th computation in {year} finished")

        pos_tr_topic_dist, _ = pos_topic_model.approximate_distribution(pos_tr_headlines)
        neg_tr_topic_dist, _ = neg_topic_model.approximate_distribution(neg_tr_headlines)
        neu_tr_topic_dist, _ = neu_topic_model.approximate_distribution(neu_tr_headlines)
        pos_te_topic_dist, _ = pos_topic_model.approximate_distribution(pos_te_headlines)
        neg_te_topic_dist, _ = neg_topic_model.approximate_distribution(neg_te_headlines)
        neu_te_topic_dist, _ = neu_topic_model.approximate_distribution(neu_te_headlines)

        pos_tr_topic_dist = assign_prob(pos_tr_topic_dist,pos_tr_topics)
        neg_tr_topic_dist = assign_prob(neg_tr_topic_dist,neg_tr_topics)
        neu_tr_topic_dist = assign_prob(neu_tr_topic_dist,neu_tr_topics)
        pos_te_topic_dist = assign_prob(pos_te_topic_dist,pos_te_topics)
        neg_te_topic_dist = assign_prob(neg_te_topic_dist,neg_te_topics)
        neu_te_topic_dist = assign_prob(neu_te_topic_dist,neu_te_topics)
        
        # This is separate version of R square
        pos_tr_r2, pos_te_r2 = linear_regression(pos_tr_topic_dist,pos_te_topic_dist,pos_tr_df,pos_te_df)
        neg_tr_r2, neg_te_r2 = linear_regression(neg_tr_topic_dist,neg_te_topic_dist,neg_tr_df,neg_te_df)
        neu_tr_r2, neu_te_r2 = linear_regression(neu_tr_topic_dist,neu_te_topic_dist,neu_tr_df,neu_te_df)
        sep_tr_r2 = (pos_tr_r2* len(pos_embeddings) / len(embeddings)) + (neg_tr_r2* len(neg_embeddings) / len(embeddings)) + (neu_tr_r2* len(neu_embeddings) / len(embeddings))
        sep_tr_r2_list.append(sep_tr_r2)
        sep_te_r2 = (pos_te_r2* len(pos_embeddings) / len(embeddings)) + (neg_te_r2* len(neg_embeddings) / len(embeddings)) + (neu_te_r2* len(neu_embeddings) / len(embeddings))
        sep_te_r2_list.append(sep_te_r2)

        # This is combine version of R square
        combined_tr_df = pd.concat([pos_tr_df,neg_tr_df,neu_tr_df],axis = 0)
        combined_tr_df.reset_index(drop=True,inplace=True)
        combined_tr_topic_dist = block_diag(pos_tr_topic_dist,neg_tr_topic_dist,neu_tr_topic_dist)
        combined_te_df = pd.concat([pos_te_df,neg_te_df,neu_te_df],axis = 0)
        combined_te_df.reset_index(drop=True,inplace=True)
        combined_te_topic_dist = block_diag(pos_te_topic_dist,neg_te_topic_dist,neu_te_topic_dist)
        com_tr_r2, com_te_r2 = linear_regression(combined_tr_topic_dist,combined_te_topic_dist,combined_tr_df,combined_te_df)
        com_tr_r2_list.append(com_tr_r2)
        com_te_r2_list.append(com_te_r2)

        print(F"Computations for {i}th computation in {year} finished")

    mean_sep_tr_r2 = np.mean(sep_tr_r2_list)
    mean_sep_te_r2 = np.mean(sep_te_r2_list)
    sep_tr_r2_dict[year] = sep_tr_r2_list
    sep_te_r2_dict[year] = sep_te_r2_list
    mean_sep_tr_r2_dict[year] = mean_sep_tr_r2
    mean_sep_te_r2_dict[year] = mean_sep_te_r2

    mean_com_tr_r2 = np.mean(com_tr_r2_list)
    mean_com_te_r2 = np.mean(com_te_r2_list)
    com_tr_r2_dict[year] = com_tr_r2_list
    com_te_r2_dict[year] = com_te_r2_list
    mean_com_tr_r2_dict[year] = mean_com_tr_r2
    mean_com_te_r2_dict[year] = mean_com_te_r2

    print(f"comutation for {year} ends")
        
print(f"The insample R square of combined version is {com_tr_r2_dict}")
print(f"The outsample R square of combined version is {com_te_r2_dict}")
print(f"The mean insample R square of combined version is {mean_com_tr_r2_dict}")
print(f"THe mean outsample R square of combined version is {mean_com_te_r2_dict}")
print(f"The insample R square of separate version is {sep_tr_r2_dict}")
print(f"The outsample R square of separate version is {sep_te_r2_dict}")
print(f"The mean insample R square of separate version is {mean_sep_tr_r2_dict}")
print(f"THe mean outsample R square of separate version is {mean_sep_te_r2_dict}")

#!/user/wx2309/.conda/envs/TM/bin/python
import os
import pandas as pd 
from lda import LDA
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.linalg import block_diag
import pickle
print("libs are read")

# Set up part
def tr_te_split(headlines,df):
    indices = np.arange(len(headlines))
    tr_ind, te_ind = train_test_split(indices, test_size=0.2, shuffle= True, random_state=i)
    tr_df = df.iloc[tr_ind,:]
    te_df = df.iloc[te_ind,:]
    tr_headlines = [headlines[ind] for ind in tr_ind]
    te_headlines = [headlines[ind] for ind in te_ind]
    return tr_df,te_df,tr_headlines,te_headlines

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
pos_saved_model_folder = "/shared/share_tm-finance/Stored_model/three_model/lda/pos"
neg_saved_model_folder = "/shared/share_tm-finance/Stored_model/three_model/lda/neg"
neu_saved_model_folder = "/shared/share_tm-finance/Stored_model/three_model/lda/neu"
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
    print(f"The df in {year} read")

    pos_indices = df[df['css'] > 0].index
    neg_indices = df[df['css'] < 0].index
    neu_indices = df[df['css'] == 0].index
    pos_df = df.iloc[pos_indices,:]
    neg_df = df.iloc[neg_indices,:]
    neu_df = df.iloc[neu_indices,:]
    pos_headlines = [headlines[ind] for ind in pos_indices]
    neg_headlines = [headlines[ind] for ind in neg_indices]
    neu_headlines = [headlines[ind] for ind in neu_indices]

    #set pos_cluster_num, neg_cluster_num, neu_cluster_num based on the number of embeddings
    pos_cluster_num = int(num_clusters * len(pos_headlines) / len(headlines))
    neg_cluster_num = int(num_clusters * len(neg_headlines) / len(headlines))
    neu_cluster_num = int(num_clusters * len(neu_headlines) / len(headlines))
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

        pos_tr_df, pos_te_df, pos_tr_headlines,pos_te_headlines = tr_te_split(pos_headlines,pos_df)
        neg_tr_df, neg_te_df, neg_tr_headlines,neg_te_headlines = tr_te_split(neg_headlines,neg_df)
        neu_tr_df, neu_te_df, neu_tr_headlines,neu_te_headlines = tr_te_split(neu_headlines,neu_df)
        pos_tr_df.reset_index(drop=True,inplace=True)
        pos_te_df.reset_index(drop=True,inplace=True)
        neg_tr_df.reset_index(drop=True,inplace=True)
        neg_te_df.reset_index(drop=True,inplace=True)
        neu_tr_df.reset_index(drop=True,inplace=True)
        neu_te_df.reset_index(drop=True,inplace=True)
        vectorizer = CountVectorizer()
        pos_tr_doc_term = vectorizer.fit_transform(pos_tr_headlines)
        neg_tr_doc_term = vectorizer.fit_transform(neg_tr_headlines)
        neu_tr_doc_term = vectorizer.fit_transform(neu_tr_headlines)
        pos_te_doc_term = vectorizer.fit_transform(pos_te_headlines)
        neg_te_doc_term = vectorizer.fit_transform(neg_te_headlines)
        neu_te_doc_term = vectorizer.fit_transform(neu_te_headlines)

        pos_topic_model = LDA(n_topics = pos_cluster_num, n_iter = 100, random_state = 66)
        neg_topic_model = LDA(n_topics = neg_cluster_num, n_iter = 100, random_state = 66)
        neu_topic_model = LDA(n_topics = neu_cluster_num, n_iter = 100, random_state = 66)
        pos_topic_model.fit(pos_tr_doc_term)
        neg_topic_model.fit(neg_tr_doc_term)
        neu_topic_model.fit(neu_tr_doc_term)
        with open(pos_saved_model_folder+f"/{year}_{pos_cluster_num}_{i}", "wb") as file:
            pickle.dump(pos_topic_model,file)
        with open(neg_saved_model_folder+f"/{year}_{neg_cluster_num}_{i}", "wb") as file:
            pickle.dump(neg_topic_model,file)
        with open(neu_saved_model_folder+f"/{year}_{neu_cluster_num}_{i}", "wb") as file:
            pickle.dump(neu_topic_model,file)
        print(F"Model fitting for {i}th computation in {year} finished")

        pos_tr_topic_dist = pos_topic_model.doc_topic_
        neg_tr_topic_dist = neg_topic_model.doc_topic_
        neu_tr_topic_dist = neu_topic_model.doc_topic_
        pos_te_topic_dist = pos_topic_model.transform(pos_te_doc_term)
        neg_te_topic_dist = neg_topic_model.transform(neg_te_doc_term)
        neu_te_topic_dist = neu_topic_model.transform(neu_te_doc_term)
        
        # This is separate version of R square
        pos_tr_r2, pos_te_r2 = linear_regression(pos_tr_topic_dist,pos_te_topic_dist,pos_tr_df,pos_te_df)
        neg_tr_r2, neg_te_r2 = linear_regression(neg_tr_topic_dist,neg_te_topic_dist,neg_tr_df,neg_te_df)
        neu_tr_r2, neu_te_r2 = linear_regression(neu_tr_topic_dist,neu_te_topic_dist,neu_tr_df,neu_te_df)
        sep_tr_r2 = (pos_tr_r2* len(pos_headlines) / len(headlines)) + (neg_tr_r2* len(neg_headlines) / len(headlines)) + (neu_tr_r2* len(neu_headlines) / len(headlines))
        sep_tr_r2_list.append(sep_tr_r2)
        sep_te_r2 = (pos_te_r2* len(pos_headlines) / len(headlines)) + (neg_te_r2* len(neg_headlines) / len(headlines)) + (neu_te_r2* len(neu_headlines) / len(headlines))
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

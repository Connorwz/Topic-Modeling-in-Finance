#!/user/wx2309/.conda/envs/TP/bin/python
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from zipfile import ZipFile
import pandas as pd
with ZipFile("/user/wx2309/Topic-modeling-store/Processed data/past ten years/contem_2014_2023.csv.zip", "r") as unzipped_file:
  with unzipped_file.open("contem_2014_2023.csv") as csv_file:
     df = pd.read_csv(csv_file)
print("df read")
from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
print("package read")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = embedding_model.encode(df.headline.tolist(),show_progress_bar=True)

import numpy as np

scratch_path = "/scratch/wx2309"
# os.makedirs(folder_path,exist_ok = True)
# with open(os.path.join(folder_path,"contem_2014_2023_embeddings.npy"), "wb") as f:
#     np.save(f, embeddings)
with open(os.path.join(scratch_path,"contem_2014_2023_embeddings.npy"), "rb") as f:
   embeddings = np.load(f)
print("embeddings read")
from sklearn.decomposition import PCA 
PCA_model = PCA(n_components=50)
reduced_embeddings = PCA_model.fit_transform(embeddings)
print("embeddings pre-reduced finished")
headline_list = df.headline.tolist()

# create vocabulary in advance to release memory
import collections
from tqdm import tqdm
vocab = collections.Counter()
tokenizer = CountVectorizer().build_tokenizer()
for headline in tqdm(headline_list):
    vocab.update(tokenizer(headline))
vocab = [word for word, count in vocab.items() if count > 200]
len(vocab)
umap_model = UMAP(n_neighbors = 10, n_components = 5, min_dist = 0.2, metric = 'cosine',random_state = 42,verbose = True)
# Upon several fittings, it was found that cluster size around 1000 to a 1.7M headlines create 50-100 topics
hdbscan_model = HDBSCAN(min_cluster_size =20000,  metric='euclidean', cluster_selection_method='eom',\
                        gen_min_span_tree=True,prediction_data=False,min_samples = 1000,verbose = True)
vectorizer_model = CountVectorizer(vocabulary=vocab,stop_words="english")
Topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model, 
                       vectorizer_model=vectorizer_model,calculate_probabilities = False,verbose = True,low_memory = True)

Topic_model.fit(headline_list,reduced_embeddings)

store_path = "/user/wx2309/Topic-modeling-store"
os.makedirs(os.path.join(store_path,"Bertopic_sp500_2014_2023_contem_model"),exist_ok=True)
Topic_model.save(os.path.join(store_path,"Bertopic_sp500_2014_2023_contem_model"),
                 serialization = "safetensors", save_ctfidf = True, save_embedding_model = embedding_model)

topic_dist, _ = Topic_model.approximate_distribution(headline_list)
contem_ret_topic_dist = pd.concat([df.drop(columns = ["rp_entity_id","headline"]),
                                   pd.DataFrame(topic_dist)],axis = 1)
grouped = contem_ret_topic_dist.groupby(['date',"comnam","ret"])
grouped_sum = grouped.sum()

X = np.array(grouped_sum)
ret = []
for ind in list(grouped_sum.index):
  ret.append(ind[2])
Y = np.array(ret).reshape(-1,1)
from sklearn.linear_model import LinearRegression
bert_model = LinearRegression(fit_intercept=True)
bert_model.fit(X,Y)
print(bert_model.score(X,Y))
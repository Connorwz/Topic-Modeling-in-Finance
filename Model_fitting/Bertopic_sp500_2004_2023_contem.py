#!/user/rl3444/.conda/envs/testEnv/bin/python
import pandas as pd
import numpy as np

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from zipfile import ZipFile

with ZipFile('/user/rl3444/NLP/Processed data/past twenty years/contem_2004_2023.csv.zip', "r") as unzipped_file:
  with unzipped_file.open("contem_2004_2023.csv") as csv_file:
     contem_2004_2023 = pd.read_csv(csv_file)

from sentence_transformers import SentenceTransformer
# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(contem_2004_2023.headline.tolist(), show_progress_bar=True)
np.save('/user/rl3444/NLP/dadta/contem_2004_2023_embeddings.npy', embeddings)

from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Reduce dimensionality
umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
# Cluster embeddings
hdbscan_model = HDBSCAN(min_cluster_size =1000,  metric='euclidean', cluster_selection_method='eom',\
                        gen_min_span_tree=True,prediction_data=False,min_samples = 50,verbose = True)
# Vectorize
vectorizer_model = CountVectorizer(stop_words="english", min_df=0.1, max_df = 0.9, ngram_range=(1, 2))

# import openai
# from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech

# # KeyBERT
# keybert_model = KeyBERTInspired()

# # Part-of-Speech
# pos_model = PartOfSpeech("en_core_web_sm")

# # MMR
# mmr_model = MaximalMarginalRelevance(diversity=0.3)

# # GPT-3.5
# client = openai.OpenAI(api_key="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")
# prompt = """
# I have a topic that contains the following documents:
# [DOCUMENTS]
# The topic is described by the following keywords: [KEYWORDS]

# Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
# topic: <topic label>
# """
# openai_model = OpenAI(client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

# # All representation models
# representation_model = {
#     "KeyBERT": keybert_model,
#     "OpenAI": openai_model,  # Uncomment if you will use OpenAI
#     "MMR": mmr_model,
#     "POS": pos_model
# }

from bertopic import BERTopic

# Create BERTopic model
topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  #representation_model=representation_model,

  calculate_probabilities=False,
  low_memory = True,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(contem_2004_2023.headline.tolist(), embeddings)

# chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in topic_model.topic_aspects_["OpenAI"].items()}
# chatgpt_topic_labels[-1] = "Outlier Topic"
# topic_model.set_topic_labels(chatgpt_topic_labels)
# topic_model.get_topic_info()

#save the topic model
folder_path = "/user/rl3444/NLP/Bertopic_sp500_2004_2023_contem_model_folder"
os.mkdir(folder_path)
with open(os.path.join(folder_path,"contem_2004_2023_embeddings.npy"), "wb") as f:
    np.save(f, embeddings)

topic_distr, _ = topic_model.approximate_distribution(contem_2004_2023.headline.tolist())
topic_distr = pd.DataFrame(topic_distr)
future_ret_topic_dist = pd.concat([contem_2004_2023.drop(columns = ["rp_entity_id","headline"]),topic_distr],axis = 1)
agg_df = future_ret_topic_dist.groupby(['date',"comnam","future_ret"])
agg_df_sum = agg_df.sum()

topic_num = topic_distr.shape[1]
topics = np.array(agg_df_sum)
ret = []
for ind in list(agg_df_sum.index):
  ret.append(ind[2])
returns = np.array(ret).reshape(-1,1)
from sklearn.linear_model import LinearRegression
bert_model = LinearRegression(fit_intercept=True)
bert_model.fit(topics,returns)
bert_model.score(topics,returns)

import matplotlib.pyplot as plt
import statsmodels.api as sm

# Add a constant to the independent variable for the intercept
X = sm.add_constant(topics)
# Fit the regression model
model_sm = sm.OLS(returns, X).fit()
# Print the summary of the regression
print(model_sm.summary())
print("\n")

#plot the coefficients
plt.figure(figsize=(10, 5))
plt.bar(range(len(bert_model.coef_[0])), bert_model.coef_[0])
plt.xticks(range(len(bert_model.coef_[0])), ["topic_" + str(i + 1) for i in range(topic_num)], rotation=90)
plt.title("Coefficients of Topics")
plt.show()
# share_tm-finance

## Processed_df: data frames preprocessed whose vocabularies are controlled
### One_year_window: data frames for each year spanning from 2014 to 2023

## Embeddings: embeddings of related data frames' headlines whose vocabularies are controlled (embedding model is the default in BERTopic if not documented)
### One_year_window: embeddings of related data frames for each year spanning from 2014 to 2023

## Weidong
### Bert_var: models saved when computing variations within BERTopic
#### One_year_window
##### pcagmm: BERTopic model having PCA as the dimension reduction model and GMM as the clustering model; the name is as `year_(number of cluster)_(ith model within 5 runs)`
##### umapgmm: BERTopic model having UMAP as the dimension reduction model and GMM as the clustering model; the name is as `year_(number of cluster)_(ith model within 5 runs)`

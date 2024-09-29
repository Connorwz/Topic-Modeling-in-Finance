# share_tm-finance

## Processed_df: data frames preprocessed whose vocabularies are controlled
1. One_year_window: data frames for each year spanning from 2014 to 2023

## Processed_df_Sentiment: prepocessed data with sentiment scores and vocab control.
1. One_year_window: data frames for each year spanning from 2014 to 2023

## Embeddings: embeddings of related data frames' headlines whose vocabularies are controlled (embedding model is the default in BERTopic if not documented)
1. One_year_window: embeddings of related data frames for each year spanning from 2014 to 2023

## Embeddings: embeddings of heandlines in prepocessed data with sentiment scores and vocab control. (embedding model is the default in BERTopic if not documented)
1. One_year_window: embeddings of related data frames for each year spanning from 2014 to 2023

## Stored_model: all Bert and LDA's training model:
1. new_data: contains the orginal one model version using latest cleaned data.
2. old_data: contains the orginal one model version using old data.
3. three_models: contain the sentiment seperate model (positive, negative, neutral)
  * lda: lda models
    * neg: lda models fitted on negative headlines;
    * neu: lda models fitted on neutral headlines;
    * pos: lda models fitted on postive headlines.
  * pcagmm: BERTopic models with PCA as dimension reduction model and GMM as clustering model
    * neg: pcagmm models fitted on negative headlines;
    * neu: pcagmm models fitted on neutral headlines;
    * pos: pcagmm models fitted on postive headlines.
  * pcahdbscan: BERTopic models with PCA as dimension reduction model and HDBSCAN as clustering model
    * neg: pcahdbscan models fitted on negative headlines;
    * neu: pcahdbscan models fitted on neutral headlines;
    * pos: pcahdbscan models fitted on postive headlines.
  * pcakmeans: BERTopic models with PCA as dimension reduction model and KMeans as clustering model
    * neg: pcakmeans models fitted on negative headlines;
    * neu: pcakmeans models fitted on neutral headlines;
    * pos: pcakmeans models fitted on postive headlines.

## Weidong
### Bert_var: models saved when computing variations within BERTopic
1. Old_one_year_window_contem: models saved when computing $R^2$ s in old ways
     * pcagmm: BERTopic model having PCA as the dimension reduction model and GMM as the clustering model; the name is as `year_(number of clusters)_(ith model within 5 runs)`
     * pcahdbscan: BERTopic model having PCA as the dimension reduction model and HDBSCAN as the clustering model; the name is as `year_(number of clusters)`
     * umapgmm: BERTopic model having UMAP as the dimension reduction model and GMM as the clustering model; the name is as `year_(number of clusters)_(ith model within 5 runs)`
2. New_one_year_window_contem: models saved when computing $R^2$ s in new ways: split headlines into training and testing sets

### Lda: models saved when computing LDA's results from 2014 to 2023
1. Old_one_year_window_contem: models saved when computing $R^2$ s in old ways; the name is as `year_(number of clusters)_(ith model within 5 runs)`
2. New_one_year_window_contem: models saved when computing $R^2$ s in new ways: split headlines into training and testing sets


## Kevin
### Bert_var: models saved when computing variations within BERTopic
1. Old_one_year_window_contem: models saved when computing $R^2$ s in old ways (using data without train-test split)
     * pcakmeans: BERTopic model having PCA as the dimension reduction model and K-Means as the clustering model; the name is as `year_(number of clusters)_(ith model within 5 runs)`
     * svdkmeans: BERTopic model having SVD as the dimension reduction model and K-Means as the clustering model; the name is as `year_(number of clusters)_(ith model within 5 runs)`
     * umaphdbscan: BERTopic model having UMAP as the dimension reduction model and HDBSCAN as the clustering model; the name is as `year_(number of clusters)_(ith model within 5 runs)`
2. New_one_year_window_contem: models saved when computing $R^2$ s in new ways: split headlines into training and testing sets

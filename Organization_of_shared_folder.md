# share_tm-finance

## Final: all data for retrain the model (will only keep this part after all calculation)
1. Embeddings: headline embeddings with vocabularies controlled
   * One_year_window: contemporaneous embeddings for each year spanning from 2014 to 2023
2. Processed_df: headline data after prepossesing
   * One_year_window: contemporaneous dataframes for each year spanning from 2014 to 2023
3. Stored_model: model data for all BERT and LDA models on training datasets
   * three_models: contain the sentiment seperate model (positive, negative, neutral)
     * MODEL_TYPE: lda, pcagmm, pcakmeans
       * neg: models fitted on negative headlines;
       * neu: models fitted on neutral headlines;
       * pos: models fitted on postive headlines.

## Processed_df_Sentiment: prepocessed data with sentiment scores and vocab control.
1. One_year_window: data frames for each year spanning from 2014 to 2023

## Embeddings with sentiment
1. One_year_window: embeddings of related data frames for each year spanning from 2014 to 2023

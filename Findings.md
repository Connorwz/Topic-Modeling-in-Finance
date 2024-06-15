# Small findings 
1. BERTopic has randomeness with significant variations from multiple computations (for 2023's contemporaneous returns, two computations give $R^2$ of 0.5% and 0.8% repsectively).
2. Future returns' $R^2$ are lower than contemperaneous returns'
3. Outliers in BERTopic are much more than topics (10 $\times$  even 100 $\times$  more) 
4. For BERTopic with PCA and KMeans replacing UMAP and HDBSCAN, the $R^2$ increased from 0.8% to 1.2% (three runs give 1.208%, 1.217%; 1.205% respectively); BERTopic with PCA and KMeans gives approxmiately 0.7% and 0.3% $R^2$ for ten and twenty years' contemporaneous data respectively.
5. For model fitting of one year's contemperaneous data with LDA, the number of iterations, log likelihood of the final iteration, and related $R^2$ s are as following:

   1. Number of topics is 62:

   |No. of iters | log likelihood | $R^2$|
   |------------|----------------|-------|
   | 1 |-245637922| 0.103%|
   | 10| -206912279|0.291%|
   | 100| -166160262|1.301%|
   | 500| -164896019|1.324%|
   | 1000| -164642525|1.253%|
   
   2. Number of topics is 100:
      
   |No. of iters | log likelihood | $R^2$|
   |------------|----------------|-------|
   | 1 | | |
   | 10| ||
   | 100| ||
   | 500| ||
   | 1000| ||
   
7. $R^2$ of BERT 2023 future returns, with Vectorize model: 
   min_df=0.1, max_df = 0.9: 0.15%
   min_df=0.05, max_df = 0.95: 0.087%
   in_df=0.15, max_df = 0.85: 0.091%
8. $R^2$ of BERT 2021-2023 future returns, min_df=0.1, max_df = 0.9: 0.082%

## Comparisons of variations within BERTopic
   1. Number of topics is 62:
   * Training:

   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means| ||
   |UMAP+GMM|||
   |PCA+GMM|||
   
   * Testing:

   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means| ||
   |UMAP+GMM|||
   |PCA+GMM|||

   2. Number of topics is 100:

   * Training:
   
   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means| ||
   |UMAP+GMM|||
   |PCA+GMM|||

   * Testing:
   
   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means| ||
   |UMAP+GMM|||
   |PCA+GMM|||
   

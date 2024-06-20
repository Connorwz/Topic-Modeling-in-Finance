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
   | 1 |-256676866|0.185%|
   | 10|-217055207|0.478%|
   | 100|-169667195|1.376%|
   | 500|-167988575|1.373%|
   | 1000|-167874214|1.408%|
   
7. $R^2$ of BERT 2023 future returns, with Vectorize model: 
   min_df=0.1, max_df = 0.9: 0.15%
   min_df=0.05, max_df = 0.95: 0.087%
   in_df=0.15, max_df = 0.85: 0.091%
8. $R^2$ of BERT 2021-2023 future returns, min_df=0.1, max_df = 0.9: 0.082%

## Comparisons of variations within BERTopic for contemporaneous returns(reduced dimension before clustering is 10 here)
   ### Number of topics is 62:
   * Training error:

   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|$3.813\times10^{-4}$|$5.783\times10^{-4}$||1.292\times10^{-3}$|
   |UMAP+GMM|$3.828\times10^{-4}$||||||$2.004\times10^{-4}$|$3.659\times10^{-4}$|$3.046\times10^{-4}$|$2.113\times10^{-4}$|
   |PCA+GMM|$3.813\times10^{-4}$|$5.782\times10^{-4}$|$3.477\times10^{-4}$|$1.291\times10^{-3}$|$3.027\times10^{-4}$|$3.554\times10^{-4}$|$2.023\times10^{-4}$|$3.607\times10^{-4}$|$3.049\times10^{-4}$|$2.078\times10^{-4}$|
   
   * Testing error:

   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|$3.927\times10^{-4}$|$6.011\times10^{-4}$||1.276\times10^{-3}$|
   |UMAP+GMM|$3.938\times10^{-4}$||||||$2.135\times10^{-4}$|$3.434\times10^{-4}$|$3.133\times10^{-4}$|$2.013\times10^{-4}$|
   |PCA+GMM|$3.926\times10^{-4}$|$6.012\times10^{-4}$|$3.604\times10^{-4}$|$1.262\times10^{-3}$|$2.901\times10^{-4}$|$3.626\times10^{-4}$|$2.064\times10^{-4}$|$3.608\times10^{-4}$|$3.115\times10^{-4}$|$2.155\times10^{-4}$|

   * $R^2$:

   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.4469%|0.3539%||2.9311%
   |UMAP+GMM|0.0685%||||||0.2147%|0.1361%|0.2617%|0.2337%|
   |PCA+GMM|0.4622%|0.3705%|0.1963%|3.180%|0.4018%|0.3310%|0.3306%|0.3599%|0.3345%|0.2631%|
   
   ### Number of topics is 100:

   * Training error:
   
   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.813\times10^{-4}$|||
   |PCA+K-Means|$3.804\times10^{-4}$|$5.777\times10^{-4}$|$3.507\times10^{-4}$|$1.274\times10^{-3}$|
   |UMAP+GMM|$3.831\times10^{-4}$||||||$2.003\times10^{-4}$|$3.657\times10^{-4}$|$3.050\times10^{-4}$|$2.116\times10^{-4}$|
   |PCA+GMM|$3.799\times10^{-4}$|$5.711\times10^{-4}$|$3.470\times10^{-4}$|$1.279\times10^{-3}$|$3.013\times10^{-4}$|$3.544\times10^{-4}$|$2.017\times10^{-4}$|$3.600\times10^{-4}$|$3.038\times10^{-4}$|$2.071\times10^{-4}$|

   * Testing error:
   
   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.939\times10^{-4}$|||
   |PCA+K-Means|$3.923\times10^{-4}$|$6.005\times10^{-4}$|$3.471\times10^{-4}$|$1.268\times10^{-3}$|
   |UMAP+GMM|$3.938\times10^{-4}$||||||$2.136\times10^{-4}$|$3.432\times10^{-4}$|$3.137\times10^{-4}$|$2.012\times10^{-4}$|
   |PCA+GMM|$3.910\times10^{-4}$|$6.005\times10^{-4}$|$3.602\times10^{-4}$|$1.252\times10^{-3}$|$2.895\times10^{-4}$|$3.622\times10^{-4}$|$2.065\times10^{-4}$|$3.613\times10^{-4}$|$3.111\times10^{-4}$|$2.156\times10^{-4}$|
   
   * $R^2$:

   |Combinatiosn/year|2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|0.4095%|||
   |PCA+K-Means|0.6956%|0.4754%|0.2942%|4.1253%|
   |UMAP+GMM|0.0089%||||||0.2398%|0.2092%|0.1323%|0.1097%|
   |PCA+GMM|0.8537%|0.5620%|0.3734%|4.096%|0.8413%|0.6098%|0.6221%|0.5000%|0.6590%|0.5206%

   ### findings within variations
   1. UMAP is highly unstable (?)
   2. The combination of UMAP and GMM performs bad (?)
   3. There is no significant gap betweem 62 and 100 clutsers' testing errors and in most cases 100 cluster's performances are better
   4. There are circumstances that testing error is larger than training error
   

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
   
6. $R^2$ of BERT 2023 future returns, with Vectorize model: 
   min_df=0.1, max_df = 0.9: 0.15%
   min_df=0.05, max_df = 0.95: 0.087%
   in_df=0.15, max_df = 0.85: 0.091%
7. $R^2$ of BERT 2021-2023 future returns, min_df=0.1, max_df = 0.9: 0.082%

## Comparisons of variations within BERTopic for contemporaneous returns(reduced dimension before clustering is 10 here)
   ### Number of topics is 60:
   * In-sample $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|$3.813\times10^{-4}$|$5.783\times10^{-4}$|$3.513\times10^{-4}$|$1.292\times10^{-3}$|
   |UMAP+GMM|$2.17\times10^{-5}$%|0.0139%|0.1017%|$3.94\times10^{-4}$%|0.0132%|0.1833%|0.0025%|0.0535%|$7.17\times10^{-4}$%|0.0906%
   |PCA+GMM|0.2534%|0.3670%|0.3425%|0.3825%|0.3563%|0.4788%|3.1512%|0.2461%|0.3583%|0.4448%|
   
   * Out-sample $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|$3.927\times10^{-4}$|$6.011\times10^{-4}$|$3.467\times10^{-4}$|$1.276\times10^{-3}$|
   |UMAP+GMM|-0.0188%|0.0320%|0.0294%|-0.0115%|0.0118%|0.1289%|-0.0014%|-0.0063%|-0.0042%|0.0202%|
   |PCA+GMM|0.1112%|-0.0417%|-0.0825%|-0.2930%|0.2023%|0.1409%|3.2702%|-0.0064%|0.1217%|0.2601%

   * Whole $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.4469%|0.3539%|0.1525%|2.9311%
   |UMAP+GMM|$8.00\times10^{-5}$%|0.0182%|0.0927%|0.0015%|0.0148%|0.1874%|0.0030%|0.0450%|$1.21\times10^{-4}$%|0.0825%|
   |PCA+GMM|0.2590%|0.3141%|0.2934%|0.3021%|0.3804%|0.4373%|3.1988%|0.2213%|0.3269%|0.4293%|
   
   ### Number of topics is 120:

   * In-sample $R^2$:
   
   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.813\times10^{-4}$|||
   |PCA+K-Means|$3.804\times10^{-4}$|$5.777\times10^{-4}$|$3.507\times10^{-4}$|$1.274\times10^{-3}$|
   |UMAP+GMM|0.1723%|0.0139%|0.1583%|$3.94\times10^{-4}$%|0.0132%|$5.52\times10^{-6}$%|0.0025%|0.0035%|$7.17\times10^{-4}$%|0.1587%|
   |PCA+GMM|0.6479%|0.7375%|0.7915%|0.9897%|0.6670%|0.9279%|4.4139%|0.5097%|0.6705%|1.0464%

   * Out-sample $R^2$:
   
   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.939\times10^{-4}$|||
   |PCA+K-Means|$3.923\times10^{-4}$|$6.005\times10^{-4}$|$3.471\times10^{-4}$|$1.268\times10^{-3}$|
   |UMAP+GMM|-0.0046%|0.0320%|-0.0071%|-0.0115%|0.0118%|-0.0021%|-0.0014%|-0.0100%|-0.0042%|0.0098%
   |PCA+GMM|0.0241%|0.1020%|-0.2788%|-0.2287%|-0.0115%|0.0250%|4.3624%|-0.0332%|0.1753%|0.0746%|
   
   * Whole $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|0.4095%|||
   |PCA+K-Means|0.6956%|0.4754%|0.2942%|4.1253%|
   |UMAP+GMM|0.1568%|0.0182%|0.1418%|0.0015%|0.0148%|$1.51\times10^{-4}\%$|0.0030%|0.0016%|$1.21\times10^{-4}\%$|0.1440%
   |PCA+GMM|0.6000%|0.6865%|0.6729%|0.8490%|0.6376%|0.8280%|4.4560%|0.4546%|0.6175%|1.0366%|

   ### findings within variations
   1. UMAP is controversial and weird. several runs for one year can raise exactly same $R^2$ while some raise $R^2$ 100\times larger or smaller
   2. The combination of PCA and GMM outperforms the combination of UMAP and GMM
   3. *There is no significant gap betweem 62 and 100 clutsers' testing errors and in most cases 100 cluster's performances are better*
   4. There are circumstances that testing error is larger than training error (lucky draws)

   

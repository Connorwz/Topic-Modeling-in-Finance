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
6. For LDA's insample $R^2$, outsample $R^2$, and whole $R^2$ having clusters of 60 and 120 from 2014 to 2024 are as follow (prior Dirichlet parameters over topics and words distributions $\alpha$ and $\eta$ are default with 0.1 and 0.01 respectively):
   * Number of topics is 60:
     |$R^2$|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
     |-----|----|----|----|----|----|----|----|----|----|----|
     Insample|0.3113%|0.3686%|0.4067%|0.5711%|0.3137%|0.5867%|2.3286%|0.1951%|0.3112%|0.9337%|
     Outsample|-0.0833%|0.0450%|0.0114%|-0.2534%|-0.0236%|0.2174%|2.1339%|-0.0566%|0.1146%|1.0335%|
     Whole|0.2803%|0.3461%|0.3588%|0.4648%|0.2829%|0.5517%|2.3191%|0.1671%|0.2914%|0.9839%|

   * Number of topics is 120:
     |$R^2$|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
     |-----|----|----|----|----|----|----|----|----|----|----|
     Insample|0.6765%|0.7714%|0.7776%|1.2895%|0.6326%|1.0900%|3.8446%|0.4514%|0.5754%|1.5668%|
     Outsample|-0.2209%|0.1464%|-0.0875%|-0.1460%|-0.0929%|-0.1643%|2.6777%|-0.1248%|$-9.8325\times10^{-4}$%|1.6245%|
     Whole|0.5974%|0.7395%|0.6753%|1.1081%|0.5886%|0.9650%|3.6817%|0.3892%|0.5120%|1.6487%|
7. Random state check: check whether 2 runs give out different results under different situations with tuning `random_state` (Y: different; N: same)

   * Data: first 349,844 rows of data in embeddings of 2023 contemperaneous returns' headlines ($\approx 20\$%)
   * PCA: check the first 5 reduced embeddings (`n_componnets = 10`)
   * UMAP: check the first 5 reduced embeddings (`n_componnets = 10`)
   * KMeans: check first 100 reduced embeddings' labels (`n_clusters = 120`, applied on reduced embeddings produced by PCA)

      |Model|No `random_state`|Same `random_state`|Different `random_state`|
      |-----|-----------------|-------------------|------------------------|
      |PCA|N|N|N|
      |UMAP|Y|**Y**|Y|
      |KMeans|Y (17 labels are different)|**Y** (15)|Y (98)|
     
   * HDBSCAN: it is found that HDBSCAN is stable with same hyperparameters
     
   
9. $R^2$ of BERT 2023 future returns, with Vectorize model: 
   min_df=0.1, max_df = 0.9: 0.15%
   min_df=0.05, max_df = 0.95: 0.087%
   in_df=0.15, max_df = 0.85: 0.091%
10. $R^2$ of BERT 2021-2023 future returns, min_df=0.1, max_df = 0.9: 0.082%

## Comparisons of variations within BERTopic for contemporaneous returns(reduced dimension before clustering is 10 here)
   ### Number of topics is 60:
   * In-sample $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.3018%|0.5848%|0.3284%|0.4002%|0.2479%|0.3597%|4.5009%|0.1754%|0.2194%|0.3009%|
   |UMAP+GMM|$2.17\times10^{-5}$%|0.0139%|0.1017%|$3.94\times10^{-4}$%|0.0132%|0.1833%|0.0025%|0.0535%|$7.17\times10^{-4}$%|0.0906%
   |PCA+GMM|0.2534%|0.3670%|0.3425%|0.3825%|0.3563%|0.4788%|3.1512%|0.2461%|0.3583%|0.4448%|
   
   * Out-sample $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.0209%|0.0755%|-0.0074%|0.0010%|-0.0097%|0.0472%|4.5237%|0.0020%|-0.0270%|0.1641%|
   |UMAP+GMM|-0.0188%|0.0320%|0.0294%|-0.0115%|0.0118%|0.1289%|-0.0014%|-0.0063%|-0.0042%|0.0202%|
   |PCA+GMM|0.1112%|-0.0417%|-0.0825%|-0.2930%|0.2023%|0.1409%|3.2702%|-0.0064%|0.1217%|0.2601%

   * Whole $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.2816%|0.5378%|0.2943%|0.3539%|0.2318%|0.3200%|4.5240%|0.1592%|0.1870%|0.3200%|
   |UMAP+GMM|$8.00\times10^{-5}$%|0.0182%|0.0927%|0.0015%|0.0148%|0.1874%|0.0030%|0.0450%|$1.21\times10^{-4}$%|0.0825%|
   |PCA+GMM|0.2590%|0.3141%|0.2934%|0.3021%|0.3804%|0.4373%|3.1988%|0.2213%|0.3269%|0.4293%|
   
   ### Number of topics is 120:

   * In-sample $R^2$:
   
   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.813\times10^{-4}$|||
   |PCA+K-Means|0.5856%|0.7982%|0.4800%|0.6725%|0.6972%|0.8062%|6.5085%|0.3720%|0.5118%|0.6141%|
   |UMAP+GMM|0.1723%|0.0139%|0.1583%|$3.94\times10^{-4}$%|0.0132%|$5.52\times10^{-6}$%|0.0025%|0.0035%|$7.17\times10^{-4}$%|0.1587%|
   |PCA+GMM|0.6479%|0.7375%|0.7915%|0.9897%|0.6670%|0.9279%|4.4139%|0.5097%|0.6705%|1.0464%

   * Out-sample $R^2$:
   
   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.939\times10^{-4}$|||
   |PCA+K-Means|0.0667%|0.2519%|-0.0563%|-0.0132%|-0.1715%|0.0293%|5.9579%|0.1523%|-0.0051%|0.4437%|
   |UMAP+GMM|-0.0046%|0.0320%|-0.0071%|-0.0115%|0.0118%|-0.0021%|-0.0014%|-0.0100%|-0.0042%|0.0098%
   |PCA+GMM|0.0241%|0.1020%|-0.2788%|-0.2287%|-0.0115%|0.0250%|4.3624%|-0.0332%|0.1753%|0.0746%|
   
   * Whole $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|0.4095%|||
   |PCA+K-Means|0.5735%|0.7746%|0.4444%|0.5940%|0.6108%|0.7128%|6.4481%|0.3270%|0.4687%|0.6931%|
   |UMAP+GMM|0.1568%|0.0182%|0.1418%|0.0015%|0.0148%|$1.51\times10^{-4}\%$|0.0030%|0.0016%|$1.21\times10^{-4}\%$|0.1440%
   |PCA+GMM|0.6000%|0.6865%|0.6729%|0.8490%|0.6376%|0.8280%|4.4560%|0.4546%|0.6175%|1.0366%|

   ### findings within variations
   1. UMAP is controversial and weird. several runs for one year can raise exactly same $R^2$ while some raise $R^2$ 100x larger or smaller
   2. There is no significant gap betweem 60 and 120 clutsers' testing errors and in most cases 120 cluster's performances are better
   3. There are circumstances that testing error is larger than training error (lucky draws)




   4. Illustration and explanation:

      1. As from the randomness: PCA is stable because it is linear transformation; UMAP's randomness can't be controlled and anomalies may be produced; KMeans has some randomness from the random state check; GMM also has some randomeness from the results of combination of PCA and GMM ($R^2$s fluctuate within a range); HDBSCAN is stable from the random state check.
      2. As from the performance: PCA outperforms UMAP from the results of combinations of (UMAP,GMM) and (PCA,GMM); KMeans and GMM have similar performances from the results of combinations of (UMAP,KMeans) and (UMAP, GMM).
         
   5. **One problem**: it is found that PCA+GMM's $R^2$s fluctuate in a range, illustring the randomness; However, except for those anomalies, $R^2$s of UMAP+GMM can stay in an exact level. Where do randomnesses of UMAP and GMM go? 
   6. Best variation advice from Weidong:

      1. As for dimension reduction: **PCA should be chosen**; UMAP does perform well in terms of its algorithm, caputuring both local and global features of data and is the recommended way from BERTopic. However, this seems not apply to our research. Some problems may arise from cuML package like having anomalies and uncontrolled randomnesses. Moreover, UMAP randomly throws a computaional error on GRID: `illegal access to memory of cuda` which is found a bug but not fixed yet and this error doesn't happen always.

   

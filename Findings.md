# Small findings 
1. BERTopic has randomeness with significant variations from multiple computations (for 2023's contemporaneous returns, two computations give $R^2$ of 0.5% and 0.8% repsectively).
2. Future returns' $R^2$ are lower than contemperaneous returns.
3. Outliers in BERTopic are much more than topics (10 $\times$  even 100 $\times$  more). Besides, its top 2/3 topics include most of documents. 
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
   
6. Random state check: check whether 2 runs give out different results under different situations with tuning `random_state` (Y: different; N: same)

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
     
   
7. $R^2$ of BERT 2023 future returns, with Vectorize model: 
   min_df=0.1, max_df = 0.9: 0.15%
   min_df=0.05, max_df = 0.95: 0.087%
   in_df=0.15, max_df = 0.85: 0.091%
8. $R^2$ of BERT 2021-2023 future returns, min_df=0.1, max_df = 0.9: 0.082%

## Comparisons of variations within BERTopic and LDA for contemporaneous returns(reduced dimension before clustering is 10 here; for LDA, prior Dirichlet parameters $\alpha$ and $\eta$ are default 0.1 and 0.01)
   ### Number of topics is 60:
   * In-sample $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.3018%|0.5848%|0.3284%|0.4002%|0.2479%|0.3597%|4.5009%|0.1754%|0.2194%|0.3009%|
   |UMAP+GMM|$2.17\times10^{-5}$%|0.0139%|0.1017%|$3.94\times10^{-4}$%|0.0132%|0.1833%|0.0025%|0.0535%|$7.17\times10^{-4}$%|0.0906%
   |PCA+GMM|0.2534%|0.3670%|0.3425%|0.3825%|0.3563%|0.4788%|3.1512%|0.2461%|0.3583%|0.4448%|
   |PCA+HDBSCAN|0.2287%(62)|0.4304%(65)|0.1567%(62)|0.1821%(43)|0.2103%(55)|0.4720%(62)|4.0486%(64)|0.1703%(58)|0.3651%(64)|0.4695%(64)|
   |LDA|0.3113%|0.3686%|0.4067%|0.5711%|0.3137%|0.5867%|2.3286%|0.1951%|0.3112%|0.9337%|
   
   * Out-sample $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.0209%|0.0755%|-0.0074%|0.0010%|-0.0097%|0.0472%|4.5237%|0.0020%|-0.0270%|0.1641%|
   |UMAP+GMM|-0.0188%|0.0320%|0.0294%|-0.0115%|0.0118%|0.1289%|-0.0014%|-0.0063%|-0.0042%|0.0202%|
   |PCA+GMM|0.1112%|-0.0417%|-0.0825%|-0.2930%|0.2023%|0.1409%|3.2702%|-0.0064%|0.1217%|0.2601%|
   |PCA+HDBSCAN|-0.0077%(62)|-0.0432%(65)|-0.0568%(62)|-0.1551%(43)|-0.0527%(55)|0.5618%(62)|3.6357%(64)|0.0635%(58)|0.1941%(64)|0.1530%(64)|
   |LDA|-0.0833%|0.0450%|0.0114%|-0.2534%|-0.0236%|0.2174%|2.1339%|-0.0566%|0.1146%|1.0335%|

   * Whole $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN||||
   |PCA+K-Means|0.2816%|0.5378%|0.2943%|0.3539%|0.2318%|0.3200%|4.5240%|0.1592%|0.1870%|0.3200%|
   |UMAP+GMM|$8.00\times10^{-5}$%|0.0182%|0.0927%|0.0015%|0.0148%|0.1874%|0.0030%|0.0450%|$1.21\times10^{-4}$%|0.0825%|
   |PCA+GMM|0.2590%|0.3141%|0.2934%|0.3021%|0.3804%|0.4373%|3.1988%|0.2213%|0.3269%|0.4293%|
   |PCA+HDBSCAN|0.2136%%(62)|0.3631%%(65)|0.1638%(62)|0.1421%(43)|0.1858%(55)|0.5161%(62)|4.0001%(64)|0.1654%(58)|0.3461%(64)|0.4282%(64)|
   |LDA|0.2803%|0.3461%|0.3588%|0.4648%|0.2829%|0.5517%|2.3191%|0.1671%|0.2914%|0.9839%|
   
   ### Number of topics is 120:

   * In-sample $R^2$:
   
   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.813\times10^{-4}$|||
   |PCA+K-Means|0.5856%|0.7982%|0.4800%|0.6725%|0.6972%|0.8062%|6.5085%|0.3720%|0.5118%|0.6141%|
   |UMAP+GMM|0.1723%|0.0139%|0.1583%|$3.94\times10^{-4}$%|0.0132%|$5.52\times10^{-6}$%|0.0025%|0.0035%|$7.17\times10^{-4}$%|0.1587%|
   |PCA+GMM|0.6479%|0.7375%|0.7915%|0.9897%|0.6670%|0.9279%|4.4139%|0.5097%|0.6705%|1.0464%|
   |LDA|0.6765%|0.7714%|0.7776%|1.2895%|0.6326%|1.0900%|3.8446%|0.4514%|0.5754%|1.5668%|

   * Out-sample $R^2$:
   
   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|$3.939\times10^{-4}$|||
   |PCA+K-Means|0.0667%|0.2519%|-0.0563%|-0.0132%|-0.1715%|0.0293%|5.9579%|0.1523%|-0.0051%|0.4437%|
   |UMAP+GMM|-0.0046%|0.0320%|-0.0071%|-0.0115%|0.0118%|-0.0021%|-0.0014%|-0.0100%|-0.0042%|0.0098%
   |PCA+GMM|0.0241%|0.1020%|-0.2788%|-0.2287%|-0.0115%|0.0250%|4.3624%|-0.0332%|0.1753%|0.0746%|
   |LDA|-0.2209%|0.1464%|-0.0875%|-0.1460%|-0.0929%|-0.1643%|2.6777%|-0.1248%|$-9.8325\times10^{-4}$%|1.6245%|
   
   * Whole $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |UMAP+HDBSCAN|0.4095%|||
   |PCA+K-Means|0.5735%|0.7746%|0.4444%|0.5940%|0.6108%|0.7128%|6.4481%|0.3270%|0.4687%|0.6931%|
   |UMAP+GMM|0.1568%|0.0182%|0.1418%|0.0015%|0.0148%|$1.51\times10^{-4}\%$|0.0030%|0.0016%|$1.21\times10^{-4}\%$|0.1440%
   |PCA+GMM|0.6000%|0.6865%|0.6729%|0.8490%|0.6376%|0.8280%|4.4560%|0.4546%|0.6175%|1.0366%|
   |LDA|0.5974%|0.7395%|0.6753%|1.1081%|0.5886%|0.9650%|3.6817%|0.3892%|0.5120%|1.6487%|
   
   ### findings within variations
   1. UMAP is controversial and weird. several runs for one year can raise exactly same $R^2$ while some raise $R^2$ 100x larger or smaller.
   2. There is no significant gap betweem 60 and 120 clutsers' testing errors and in most cases 120 cluster's performances are better.
   3. When using UMAP+HDBSCAN, by tunning n_neighbors, n_epochs, and learning_rate in UMAP, and min_cluster_size in HDBSCAN, we can control the number of cluster in a range (The outliery probloem still exists).
      1. If we want 60 clusters, we can choose\
         min_cluster_size = 800, n_neighbors = 10;\
         min_cluster_size = 1150, n_neighbors = 30;\
         min_clucster_size = 1450, n_neighbors = 10, n_epochs=1000, learning_rate=0.5;\
         min_clucster_size = 2500, n_neighbors = 30, n_epochs=1000, learning_rate=0.5;
      2. If we want 120 clusters, we can choose\
         min_cluster_size = 400, n_neighbors = 10;\
         min_cluster_size = 750, n_neighbors = 30;\
         min_clucster_size = 790, n_neighbors = 10, n_epochs=1000, learning_rate=0.5;\
         min_clucster_size = 1300, n_neighbors = 30, n_epochs=1000, learning_rate=0.5;
      3. However, the $R^2$ for different combination are similar.
   4. There are circumstances that testing error is larger than training error (lucky draws).
   5. For combination of PCA+K-means, there is no outlier cluste, while UMAP+HDBSCAN has.
   6. Topics from HDBSCAN have a lot of outliers ($\approx$ one half of all) and top 2/3 topics include most of non-outliers.
   7. Illustration and explanation:
      1. As from the randomness: PCA is stable because it is linear transformation; UMAP's randomness can't be controlled and anomalies may be produced; KMeans has some randomness from the random state check; GMM also has some randomeness from the results of combination of PCA and GMM ($R^2$ s fluctuate within a range); HDBSCAN is stable from the random state check.
      2. As from the performance: PCA outperforms UMAP from the results of combinations of (UMAP,GMM) and (PCA,GMM); GMM outperforms KMeans from the results of combinations of (UMAP,KMeans) and (UMAP, GMM); GMM and HDBSCAN have similar performances from the results of combinations of (PCA,GMM) and (PCA,HDBSCAN).
      3. As from the representation of topics (60 topics in 2014,2018,2023): (PCA,HDBSCAN) gives many meanless topics including time (week days, months, seasons) and numbers while (PCA,KMeans) and (PCA,GMM) tend to be more reasonable.
   8. **One problem**: it is found that PCA+GMM's $R^2$ s fluctuate in a range, illustring the randomness; However, except for those anomalies, $R^2$ s of UMAP+GMM can stay in an exact level. Where do randomnesses of UMAP and GMM go? 
   9. Best variation advice from Weidong:
      1. As for dimension reduction: **PCA should be chosen**; UMAP does perform well in terms of its algorithm, caputuring both local and global features of data and is the recommended way from BERTopic. However, this seems not apply to our research. Some problems may arise from cuML package like having anomalies and uncontrolled randomnesses. Moreover, UMAP randomly throws a computaional error on GRID: `illegal access to memory of cuda` which is found a bug but not fixed yet and this error doesn't happen always.
      2. As for clustering reduction: **GMM should be chosen**; GMM outperforms KMeans and has similar performances to HDBSCAN while it has a better topic representation.

## New comparisons of variations within BERTopic and LDA for contemporaneous returns(reduced dimension before clustering is 10 here; for LDA, prior Dirichlet parameters $\alpha$ and $\eta$ are default 0.1 and 0.01)
   ### Number of topics is 60:
   * In-sample $R^2$:

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |PCA+K-Means|0.1535%|0.2169%|0.1293%|0.1211%|0.1500%|0.2210%|0.2096%|0.0824%|0.1773%|
   |PCA+GMM|0.1548%|0.2257%|0.1495%|0.1432%|0.1559%|0.2076%|0.2232%|0.0853%|0.1797%|
   |PCA+HDBSCAN|
   |LDA|0.1550%|0.2087%|0.1316%|0.1520%|0.1898%|0.2354%|0.2468%|0.0974%|0.1687%|

   * Out-sample $R^2$ (old regression):

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |PCA+K-Means|-0.0208%|-0.0243%|-0.0059%|-0.0098%|-0.0011%|-0.0154%|-0.0149%|-0.0047%|-0.0150%|
   |PCA+GMM|-0.0184%|-0.0263%|0.0050%|-0.0079%|-0.0004%|-0.0172%|-0.0152%|-0.0063%|-0.0191%|
   |PCA+HDBSCAN|
   |LDA|-0.0322%|-0.0197%|-0.0099%|-0.0094%|-0.0413%|-0.0170%|-0.0284%|-0.0115%|0.0104%|
   
   ### Number of topics is 120:

   * In-sample $R^2$:
   
   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |PCA+K-Means|0.3250%|0.3856%|0.3081%|0.2485%|0.3106%|0.4191%|0.4026%|0.1784%|0.2718%|
   |PCA+GMM|0.2873%|0.3691%|0.2863%|0.2335%|0.3089%|0.4276%|0.3786%|0.1857%|0.2972%|
   |PCA+HDBSCAN|
   |LDA|0.3448%|0.3601%|0.2701%|0.2906%|0.3321%|0.4180%|0.3965%|0.1869%|0.2859%|

   * Out-sample $R^2$ (old regression):

   |Combinatiosn/year|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|
   |:-----------------:|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
   |PCA+K-Means|-0.0232%|-0.0338%|-0.0089%|-0.0206%|-0.0136%|-0.0330%|-0.0330%|-0.0079%|-0.0211%|
   |PCA+GMM|-0.0304%|-0.0327%|-0.0028%|-0.0195%|-0.0136%|-0.0397%|-0.0284%|-0.0136%|-0.0273%
   |PCA+HDBSCAN|
   |LDA|-0.0885%|-0.0957%|-0.0392%|-0.0453%|-0.0903%|-0.0463%|-0.0803%|-0.0500%|-0.0215%|

   ### Findings
   1. (PCA+GMM) and (PCA+KMeans) have similar performances in terms of $R^2$;
   2. 120 topics have better performance and generalization than 60 topics;
   3. When `.score()` in LinearRegression of sklearn is used to calculate the out-smaple $R^2$, testing y's own mean is used to get SST;

   ### Problems
   1. Outliers in 2020 disappear

   ### New Findings from second cleaned version data and model
   **IMPORTANT: Detailed data will be uploaded as an excel file.**
   
   **IMPORTANT: the split of dataset is locked eachtime. Further experiement will be done later.**
   
   1. As for the new model with updated dataset, sentiment score, and relevent score threshold, the performce of 120 clusters is better than 60's.
   2. Using Per Topic sentiment version does not have improvement in in-sample R^2. Moreover, the aggregate sentiment score based on topic may bring too much future information. More importance will be put on per return version and with sentiment version in the future.
   3. For different sentiment version, per return and with return outperforms no sentiment and only sentiment. The best score shows up in per return.
   4. For in-sample R^2, adding sentiment do explain return better than only sentiment and only topic weight. But the out-sample R^2 is not satisfactory, negative values often appears. Need to improve the model.
   5.  Regression with only sentiment score show jumps in year 2017, 2019, 2023. This jump is more significant in per return version. (?)
   6. Regarding the in-sample R^2, PCA-KMeans, PCA-Gmm out performs PCA-HDBScan, and LDA. For all bertopic model, the biggest gap is around 2 percent. All bertopic model outperforms LDA at an average level around 4%. 120 outperforms 60, with 1% improvement.
   7. R^2 in 2023 is much larger than other years. (?)
   8. As for cos similarity within topics, LDA has the highest score, around 0.72. Score for PCA-Kmeans, PCA-Gmm are around 0.6. For all models, the score for 120 and 60 are similar.
   9. As for cos similarity between topics, PCA-KMeans, PCA-Gmm has the lowest result, around 0.4. When increasing cluster num from 60 to 120, PCA-KMeans, PCA-Gmm has lower score. For          all other models, the score are similar.
   10. As for diversity score, PCA-KMeans, PCA-Gmm has the highest ranging from 0.75 to 0.8. When increasing cluster num from 60 to 120, LDA has wose score, decreasing from 0.8 to 0.7. All others donot have big changes.
   11. Reduce outliers for PCA-HDBScan can increaase the perfromance of in-sample R^2, but the cos similarity between topics increases a lot.
   12. LDA achieves better result in cos similarity with topics, while the score for cos similarity within topics and between topics are similar, indicating LDA's topics donot       have significant difference. (?)
   13. Overall, PCA-Kmeans has the best performance in R^2, cos similarities and model diversity.

   14. Combined version is better than separate version in terms of in-sample and out-sample performances.
   15. Model performs better at most times when CountVectorizer has no parameters.
   16. There are headlines having no exposures to topics generated by topic models (with PCA and KMeans as dimension reduction and clustering techniques).
   17. For LDA, the topic exposures of training dataset are highly concentrated in some numbers which is because the training data is short headlines and they aren't updated far away from the prior distribution but are versatile for testing data, causing bad generalization of LDA model. From this result, it is recommended to use BERTopic other than LDA over short documents.
   18. For segfault thrown out by HDBSCAN, it can be fixed with removing duplicated headlines. However, removing duplicates has a big impact on number of clusters produced. For example, training headlines in 2022 have a length of 315,320 and duplicates-removed headlines are 202,146 but number of clusters produced are 478 and 35 respectively with `min_cluster_size` set as 10.
   19. When training data set is smaller, the outliers produced by HDBSCAN are more.
   20. After python package is changed from cuML to sclearn, there are still headlines of zero topic exposure.
   21. After assigning indicator probability of one to zero topic exposure headlines, there is a slight drop on the $R^2$.
   22. The reason we have zero topic exposures is because there is a [minimum_similarity](https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py#L1223) and those similarities below this minimum is set as zero.

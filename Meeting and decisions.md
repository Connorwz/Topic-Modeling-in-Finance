# 13/9/2024
1. Find the exact reason why there are headlines having zero topic exposure.

# 10/9/2024
1. Keep duplicated headlines and stop fitting model with hdbscan.
2. Try fitting BERTopic with sci-kit learn package and see whether there are zero topic exposure headlines.
3. Assign those headlines having zero topic exposure probability of 1 to the cluster/topic where it's fitted in during model fitting and calculate the $R^2$

# 2/9/2024
1. Check the original codes for python package lda and see whether `.transform()` uses the trained model's parameters.
2. Check why HDBSCAN keeps throwing out segfault by looking at duplicated rows and possible identical reduced embeddings.
3. Look into why there are zero topic exposures when generating then with `.approximate_distribution()` and see whether this happens less if headlines considered as outliers in HDBSCAN are removed before applying `.approximate_distribution()`
4. In the future studies, CountVectorizer() are set without parameters but look into why better $R^2$ s are thrown out when CountVectorizer() has no parameters. 

# 7/30/2024
1. No we choose the three-in-one sentiment model (building model based on dataset with positive, negative, neutral sentiment score seperately). When calculate R squre, we do not add sentiment score into the dataframe that contains topic exposure (no_senti version).
2. Calculate R square of seperate version based on three model emmbeddings' weight. Calculate R square of combined version based on concatinate data frame.
3. seperate version and combined version using the same stored model.
4. Find out why LDA doesnot perform well in terms of words in topics and R square.
5. Make comparison of topic representation words in BERT and LDA. Give examples of the topics generate.

# 7/23/2024
1. Build three-model version of topic model. Seperate the dataset into news with positive, negative, neutral sentiment score. Train three models seperately.
2. Do linear-regression on seperate, combine basis.
    * Seperate: regress three times, calculate R^2, model score.
    * Combine: concat three dataframe from three models together, regress one time, calculate R^2, model score.
3. Compare the topic data of LDA with Bertopic.

# 7/18/2024
1. Remove weekdays, months, years from dataset. ### Final version of data prepossesing.
2. Train models: PCA-Kmeans, PCA-Gmm, PCA-Hdbscan, PCA-Hdbscan (reduce outliers), LDA.   
3. Model Parameters include:
    * sentiment type: no_senti, only_senti, with_senti, per_return.
    * cluster number: 60, 120.

# 7/16/2024
1. As for the dataset:
    * Remove headlines with a low relevance score. Set the threshold to be 75.
    * Replace numbers with their magnitudes – i.e., numbers in millions become “mln”. Keep percentage numbers, remove all other numbers.
    * Remove words that appear too frequently or once. Create a list contain useless frequent words from top 150 words in dataset. Add function that ask user to manualy remove words from 150 top frequent words
2. focus on per return version of sentiment analysis. Drop per topic version

# 7/11/2024
1. Clean the dataset. Drop all 'inst holders' headlines. Some headlines corresponding entities are “source” role we may want to dro these headlines also.
2. Calculate insample and outsample $R^2$:
   *In-sample: using training data to fit linear regression. Use the score function to get in-sample $R^2$.
   *Out-sample: fitting the topic model with new data. Use the in-sample liner regression result and in-sample mean to calculate out-sample $R^2$.
3. Calculate the coherence/diversity scores for topic representations as a metric to make comparisons in BERTopic.
   * The cohenrence score (TOPIC BASED)
   * The diversity score (MODEL BASED)
   * The similarity score
   * The significance score
4. Build the score model for LDA.

# 7/5/2024
1. Updates the codes to generate new dataframes with sentiment scores related to headlines and data frames and embeddings in the shared folder (only contemperaneous returns, specifying the files' names).
2. Regression with sentiment score:
   * First way: Sentiment per return. Sum up headlines' sentiment score for each company each day, and times topic weight (per company per day), then do the regression.
   * Second way: Sentiment per topic. Sum up headlines' sentiment score for each topic after model fitting, standardize the socre, and times topic weight, then do the regression.
3. Calculate insample and outsample $R^2$ in the way that: fit the topic models in training headlines and freeze them to allocate topics on testing headlines (training:testing = 0.8:0.2), then caluclate insample and outsample $R^2$ s for 60 and 120 clusters respectively.
4. Calculate the coherence/diversity scores for topic representations as a metric to make comparisons.
   * The cohenrence score measures how the words in topic representation related to each other. If the score is higer, the topic represnetation is more interpretable to human.
   * The cohenrence score can be calculated for the entire model or for each topic.
   * The diversity score: the number of unique representation words in a model / the number of all representation wordss in a model.
5. UMAP is abandonded due to its computational error and unreliability.

# 6/27/2024
1. Run LDA with cluster size 120 on one year data from 2014 to 2023.
2. Run UMAP + HDBSCAN, compare different combination of parameters that lead to same clster range.
3. Compare results of LDA with Bertopic based on $R^2$ table.
4. Finish topic representation score: coherence score + diversity score + $R^2$. Compare the performance of different models.
5. Combined the sentiment analysis with regression to see if the result is predictive.
   * One document or One day’s sentiment
   * Using sentiment analysis on the topic to find good news and bad news. Or
   * Building positive, neutral, or negative sentiment topic model separately.

# 6/20/2024
1. Comparing results for 60 and 120 topics each year, still recording the average of five runs, save topic representation from a random one trial. Calculating in-sample, out-sample and whole $R^2$.
2. Tune the HDBCAN to shrink topic number range. Control to +- 5. Find the relationship between topic number and R^2 of HDBSCAN.
3. To address the randomness of HDBSCAN, for 60 and 120 topics' comparisons, those results from the same parameters within HDBSCAN having clusters within 55-65 ro 115-125 can be used.
4. Find another dimentionality reduction algorithim, compare different models.
5. Analyzing topic quality, especially 2020 and 2014, make adjustment.
   * number of meaningful words in each topic.
   * human interpretable (?)
6. Finish hyperparameter tuning on BERTopic and apply the optimal ones on LDA.

# 6/14/2024
1. Focus on one year's data now
2. Document trails within BERTopic in a table that each column is a year and each row is a combination of models. Results in cells should be the average of **5** runs
3. Explore and compare results for 62 (or (57,67) for random algorithms) and 100 (or (95,105) for random algorithms) topics for each year
4. Explore the overfitting with spliting data into training set and testing set in the ration of 80% : 20%
5. For research on variations within BERTopic:
  * Vocabulary control: (1) remove stop words in English (2) remove top 100 frequent words (3) remove words appearing once (account for approximately 25% of the whole vocabulary for each year) (4) If feasible, remove news source
  * Embedding model is assumed to be the default one in BERTopic
  * Reduced dimension is set as 10
  * Three values are calculated: training error and testing error and $R^2$ based on the whole data

# 6/7/2024
1. Document influences of **LDA**'s iterations on $R^2$
2. Compare different choices of models or variations within **BERTopic**

# 5/31/2024
1. Document findings through research
2. There are variations and randomeness in **Bertopic**. The different setting-ups causing variations should be documented
3. Vocabulary should be set properly: dropping those words very frequentaly or unusally; and they should be set similarly for **LDA** and **BERTopic**

# 5/8/2024
1. Drop those *entity_ids* in the mapping file which don't exist in RavenPack
2. Include future returns (as in **4/26/2024**) and contemporaneous  returns (as in **4/18/2024**)

# 4/26/2024
1. It's more interested in investigating the trading returns so headlines and trading returns are linked in the way that ($t+1$ means next trading day):
   * $P_{t}$:opening price on the same trading day; $P_{t}'$: closing price on the same trading day; $P_{t+1}$: opening price on the next trading day ...
   * Headlines are released on holidays--returns are calculated $r = \frac{P_{t+1}' - P_{t+1}}{P_{t+1}} = COret_{t+1}$ 
   * Headlines are released before trading day's opening time (9:00 AM ET for NYSE)--returns are calculated $r = \frac{P_{t}' - P_{t}}{P_{t}} = COret_{t}$
   * Headlines are released after trading day's opening time (9:00 AM ET for NYSE) and before closing time (4:00 PM ET for NYSE)--returns are calculated $r = \frac{P_{t+1}' - P_{t}'}{P_{t}'} = ret_{t+1}$
2. Drop those companies whose ***permno*** in CRSP are not mapped to ***Entity_id*** in RavenPack

# 4/18/2024
1. Headlines should be associated with returns based on timestamps in the way that ($t+1$ means next trading day):
   * Headlines are released on holidays--next trading day's return $= ret_{t+1}$
   * Headlines are released before trading day's closing time (4:00 PM ET for NYSE)--same day's return $= ret_{t}$
   * Headlines are released after trading day's closing time (4:00 PM ET for NYSE)--next trading day's return $= ret_{t+1}$
2. Based on previous research experiences, 200 is a good choice of topics' size for 2M articles and **50-100** is good for **2.8M** headlines
3. Research on BERT: Topic number chosen; Different topic numbers each time; How to transfer embedding back to words
4. Research on LDA: Gibbs sampling, check convergence tools or parameters

# 4/12/2024
1. Companies invovled in the research should be at least monthly constituents of SP500 stocks
2. Create a time-varying list available on CRSP
3. Link the stock market data from CRSP with the company identifier from Raven Pack
4. Same headlines happening in consecutive days should be removed
5. For LDA: Pick a subset of the words and specifiy the vocab.
6. Evaluate Computing-complexities: BERT: Embeddings and Model Fitting.
7. Find research support in Parallel Processing and Running over-night.

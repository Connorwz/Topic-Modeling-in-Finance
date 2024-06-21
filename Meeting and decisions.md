# 4/12/2024
1. Companies invovled in the research should be at least monthly constituents of SP500 stocks
2. Create a time-varying list available on CRSP
3. Link the stock market data from CRSP with the company identifier from Raven Pack
4. Same headlines happening in consecutive days should be removed
5. For LDA: Pick a subset of the words and specifiy the vocab.
6. Evaluate Computing-complexities: BERT: Embeddings and Model Fitting.
7. Find research support in Parallel Processing and Running over-night.

# 4/18/2024
1. Headlines should be associated with returns based on timestamps in the way that ($t+1$ means next trading day):
   * Headlines are released on holidays--next trading day's return $= ret_{t+1}$
   * Headlines are released before trading day's closing time (4:00 PM ET for NYSE)--same day's return $= ret_{t}$
   * Headlines are released after trading day's closing time (4:00 PM ET for NYSE)--next trading day's return $= ret_{t+1}$
2. Based on previous research experiences, 200 is a good choice of topics' size for 2M articles and **50-100** is good for **2.8M** headlines
3. Research on BERT: Topic number chosen; Different topic numbers each time; How to transfer embedding back to words
4. Research on LDA: Gibbs sampling, check convergence tools or parameters


# 4/26/2024
1. It's more interested in investigating the trading returns so headlines and trading returns are linked in the way that ($t+1$ means next trading day):
   * $P_{t}$:opening price on the same trading day; $P_{t}'$: closing price on the same trading day; $P_{t+1}$: opening price on the next trading day ...
   * Headlines are released on holidays--returns are calculated $r = \frac{P_{t+1}' - P_{t+1}}{P_{t+1}} = COret_{t+1}$ 
   * Headlines are released before trading day's opening time (9:00 AM ET for NYSE)--returns are calculated $r = \frac{P_{t}' - P_{t}}{P_{t}} = COret_{t}$
   * Headlines are released after trading day's opening time (9:00 AM ET for NYSE) and before closing time (4:00 PM ET for NYSE)--returns are calculated $r = \frac{P_{t+1}' - P_{t}'}{P_{t}'} = ret_{t+1}$
2. Drop those companies whose ***permno*** in CRSP are not mapped to ***Entity_id*** in RavenPack

# 5/8/2024
1. Drop those *entity_ids* in the mapping file which don't exist in RavenPack
2. Include future returns (as in **4/26/2024**) and contemporaneous  returns (as in **4/18/2024**)

# 5/31/2024
1. Document findings through research
2. There are variations and randomeness in **Bertopic**. The different setting-ups causing variations should be documented
3. Vocabulary should be set properly: dropping those words very frequentaly or unusally; and they should be set similarly for **LDA** and **BERTopic**

# 6/7/2024
1. Document influences of **LDA**'s iterations on $R^2$
2. Compare different choices of models or variations within **BERTopic**

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

# 6/20/2024
1. Comparing results for 60 and 120 topics each year, still recording the average of five runs, save topic representation. Calculating in sample and out sample $R^2$.
2. Tune the HDBCAN to shrink topic number range. Control to +- 5. Find the relationship between topic number and R^2 of HDBSCAN.
3. Find another dimentionality reduction algorithim, compare different models
4. Analyzing topic quality, especially 2020 and 2014, make adjustment.
   * number of meaningful words in each topic.
   * human interpretable (?)
5. Finish hyperparameter tuning on BERTopic and apply the optimal ones on LDA.


   

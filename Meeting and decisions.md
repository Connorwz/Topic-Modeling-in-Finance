# 4/12/2024
1. Companies invovled in the research should be at least monthly constituents of SP500 stocks
2. Same headlines happening in consecutive days should be removed

# 4/18/2024
1. Headlines should be associated with returns based on timestamps in the way that:
   * Headlines are released on holidays--next trading day's return
   * Headlines are released before trading day's closing time (4:30 PM ET for NYSE)--same day's return 
   * Headlines are released after trading day's closing time (4:30 PM ET for NYSE)--next opeing day's return
2. Based on previous research experiences, 200 is a good choice of topics' size for 2M articles and **50-100** is good for **2.8M** headlines

# 4/26/2024
1. It's more interested in investigating the trading returns so headlines and trading returns are linked in the way:
   * $P_{t}$:opening price on the same trading day; $P_{t}'$: closing price on the same trading day; $P_{t+1}$: opening price on the next trading day ...
   * Headlines are released on holidays--returns are calculated $r = \frac{P_{t+1}' - P_{t+1}}{P_{t+1}}$
   * Headlines are released before trading day's opening time (9:00 AM ET for NYSE)--returns are calculated $r = \frac{P_{t}' - P_{t}}{P_{t}}$
   * Headlines are released after trading day's opening time (9:00 AM ET for NYSE)--returns are calculated $r = \frac{P_{t+1}' - P_{t}'}{P_{t}'}$
2. Drop those companies whose ***permno*** in GRSP are not mapped to ***Entity_id*** in RavenPack 

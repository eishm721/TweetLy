# Tweet.Ly 

This is a text prediction algorithm based off scraped Twitter history to predict w/ 96% accuracy, future user speech patterns.

This API has two modes, analyze and compare. Analyze reports a users tweeting patterns and other fun statistics. Compare takes in a sample of text and a list of 2 or more users and predicts which user is most likely to say that tweet based on a probabilistic model, along with a similarity index. All input handled through command line. Requires Twitter developer account with access keys.

Framework is developed in Python using Tweepy API and twitter developer account. Initial steps were designing a scraping algorithm to process user tweets via Tweepy. Next steps were designing a machine-learning classification algorithm (NumPy/SciPy) using a Bayesian model and a multinomial random variable distribution to use a "Bag of Words" model. Simulations were run w/ 100,000 user tweets to test the accuracy of the model in many different categories.

Features:
- 2 modes of analyzing scraped data
  - Analyze: Tweeting statistics and patterns for set of users
  - Compare: apply text predictionl algorithm to predict future speech patterns for users
- Reads up-to-date tweet data on each call for an automatically updating dataset
- Set number of tweets to base models from

Libraries Used:
- Tweepy - handle twitter API and scraping
- NumPy/SciPy - mathematical analysis with large data sets

Set-Up:
- Requires twitter developer account w/ consumer key & access token (https://developer.twitter.com/en)
  - Replace Lines 18-21 in analyzetweets.py with developer account information
- Run analyzetweets.py in command line with specified parameters

Most Recent Changes (05/07/20):
- Dramatic efficiency improvements - reduced tweet parsing from O(n^2) to O(n)
    - Improved efficiency of building dictionary by 2-fold
- Ran simulations with 100,000 user tweets to determine accuracy of prediction algorithm in many different categories
- Removed retweets from calculation
- Optional parameter to limit number of tweets to parse (for lighter CPU loads)
- Bug fixes


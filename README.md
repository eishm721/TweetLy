# TweetLy - a text prediction algorithm based of scraped Twitter history

This is a Twitter prediction API that uses past tweet history to predict future user speech patterns. Has two modes, analyze and compare. Analyze reports a users tweeting patterns and other fun statistics. Compare takes in a sample of text and a list of 2 or more users and predicts which user is most likely to say that tweet based on a probabilistic model, along with a similarity index. All input handled through command line. Requires Twitter developer account with access keys.

Framework is developed in Python using Tweepy API and twitter developer account. Initial steps were designing a scraping algorithm to process user tweets via Tweepy. Next steps were designing a probablistic modeling algorithm (NumPy/SciPy) using a Bayesian model and a multinomial random variable distribution to use a "Bag of Words" model.

Features:
- 2 modes of analyzing scraped data
- Reads up-to-date tweet data on each call for an automatically updating dataset
- Set number of tweets to base models from

Libraries Used:
- Tweepy - handle twitter API and scraping
- NumPy/SciPy - mathematical analysis with large data sets

Current work is being done on developing a newer, simpler algorithm for computing log-similarities based off a similar framework and doing algorithmic analysis. Also working on improving efficiency.

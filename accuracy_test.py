# FILENAME: accuracy_test.py

# Computes accuracy of log-similarity algorithm by comparing predicted user data
# to the actual user who wrote a given tweet.
# Simulated over 100,000 trials for 96% overall accuracy across various categories

# This file contains 3 sample categories with significant variation to test

import sys
import tweepy
import similarity
import processTweet

# Replace with Twitter developer information
CONSUMER_KEY = ###
CONSUMER_SECRET_KEY = ###
ACCESS_TOKEN = ###
ACCESS_SECRET_TOKEN = ###

# Sampling criteria - famous (large number of tweets), have similar-minded people and difference (ex: left/right wing),
# many different disciplines (politics, music, business, etc...)
USERS = dict(politicians={"barackobama", "realdonaldtrump", "berniesanders", "speakerryan"},
              musicians={"selenagomez", "kendricklamar", "danielcaesar", "drake"},
              entrepreneurs={"billgates", "elonmusk", "jeffbezos"})


class TestSim:
    """
    Class for testing accuracy of the model
    """

    def __init__(self, category):
        """
        Initializes twitter client authentication and sets-up data structures
        """
        self.users = USERS[category]
        self.user_counts = dict()  # maps each user to dictionary of word counts
        self.user_tweets = dict()  # maps each user to list of their tweers

        # Authentication
        self.auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET_KEY)
        self.auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET_TOKEN)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        try:
            self.api.verify_credentials()
        except:
            sys.exit()

    def __process_tweet(self, word_counts, tweet):
        """
        Takes in a given tweet and "cleans" it, removing URLs/handles and lowercases all
        text to standardized important phrases. Adds counts of each word to a total counts dictionary
        """
        original_words = processTweet.CleanTweet(tweet).process_string().split()  # cleans original tweet & splits
        # standard dict count algorithm
        for word in original_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

    def __build_dicts(self):
        """
        For each user, extracts all of their tweets and word frequency dictionaries based on their past
        tweeting history. Populates values in already initialized user_tweets and user_counts dicts
        """
        for user in self.users:
            # build dict of tweets
            tweets = [tweet.text for tweet in self.api.user_timeline(screen_name=user) if tweet.text[:2] != 'RT']
            self.user_tweets[user] = tweets

            # build dict of counts
            curr_user_tweets = dict()
            for tweet in tweets:
                self.__process_tweet(curr_user_tweets, tweet)
            self.user_counts[user] = curr_user_tweets

    def calc_accuracy(self):
        """
        For a given category, returns the proportion of times that the similarity algorithm returns the
        correct user that wrote that tweet. Uses multinomial model in similarity.py.
        """
        # verify initialization
        if not self.user_counts:
            self.__build_dicts()

        # loop through each user and their tweets
        total = correct = 0
        for user, tweets in self.user_tweets.items():
            for tweet in tweets:
                # for each tweet, check if the predicted user is the same as the actual user
                mostSimUser = similarity.UserSimilarity(self.user_counts, tweet).most_similar_user()
                if mostSimUser == user:
                    correct += 1
                total += 1
        return correct / total


def main():
    """
    Main function that prints accuracies for all categories as percentages
    """
    accuracy_sum = 0
    for category in USERS:
        test = TestSim(category)
        accuracy = test.calc_accuracy()
        accuracy_sum += accuracy
        print(category + ":", str(round(accuracy*100, 3))+"%")
    print("\nOverall Accuracy:", str(round((accuracy_sum / len(USERS)) * 100, 3)) + "%\n")


if __name__ == '__main__':
    main()

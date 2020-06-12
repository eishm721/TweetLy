# FILENAME: ml_testing.py

# Contains class for testing Naive Bayes and Logistic Regression algorithms

# Accuracy of these models tested over 100,000 trials across various categories
# with up-to-date tweeting data scraped w/ Tweepy
# Uses customizable training-testing_ratio (default 0.7) to specify how many
# tweets to use in training and how many to use in testing

# Sample users in authentication.py


import sys
import math
import tweepy
from random import randrange
from authentication import *
import ml_algs

TRAINING_TESTING_RATIO = 0.7


class TestClassifier:
    def __init__(self, category):
        """
        Initializes twitter client authentication and sets-up data structures
        """
        self.users = USERS[category]
        self.testing_users = list()
        self.user_tweets = dict()  # maps each user to list of their tweets
        self.training_tweets = dict()
        self.testing_tweets = dict()

        # Authentication
        self.auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET_KEY)
        self.auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET_TOKEN)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        try:
            self.api.verify_credentials()
        except:
            sys.exit()

    def __pick_testing_users(self):
        """
        From all potential users, pick two unique users to analyze
        """
        count = 0
        while count < 2:
            user = self.users[randrange(0, len(self.users))]
            if user not in self.testing_users:
                self.testing_users.append(user)
                count += 1

    def __build_data_dicts(self):
        """
        Built 3 dictionaries mapping users to an array of their tweets:
            - all tweets
            - tweets in training (specified by TRAINING_TESTING_RATIO)
            - tweets in testing
        """

        # builds dicts of ALL tweets to start
        self.__pick_testing_users()
        for user in self.testing_users:
            self.user_tweets[user] = [tweet.text for tweet in self.api.user_timeline(screen_name=user) if tweet.text[:2] != 'RT']

        # populate training and testing tweets
        max_training_size = min(list(map(lambda person: len(self.user_tweets[person]), self.testing_users)))
        training_size = math.ceil((max_training_size * TRAINING_TESTING_RATIO) + 1)
        for user in self.testing_users:
            tweets = self.user_tweets[user]
            self.training_tweets[user] = tweets[0: training_size]
            self.testing_tweets[user] = tweets[training_size:]

    def calc_accuracy(self):
        """
        Calculates accuracy of both a naive bayes and logistic regression algorithm
        across all testing tweets
        """
        self.__build_data_dicts()
        print("Data dictionaries built...")
        classifier = ml_algs.Classifier(self.training_tweets)

        # TESTING DATA: loop through each user and their tweets
        total = naive_correct = log_correct = 0
        for user, tweets in self.testing_tweets.items():
            for tweet in tweets:
                # for each tweet, check if the predicted user is the same as the actual user
                most_sim_naive = classifier.predict_user("naive", tweet)
                most_sim_log = classifier.predict_user("log", tweet)
                if most_sim_log == user:
                    log_correct += 1
                if most_sim_naive == user:
                    naive_correct += 1
                total += 1
            print("Finished predicting " + "@" + user)
        return naive_correct / total,  log_correct / total

    def print_accuracy(self):
        """
        Prints accuracy of both models across all trials in accuracy test
        """
        print("Starting accuracy test...")
        results = self.calc_accuracy()
        print("Naive Bayes:", str(round(results[0] * 100, 4)) + "%")
        print("Logistic Regression:", str(round(results[1] * 100, 4)) + "%")


def main():
    """
    Main function that prints accuracies for all categories as percentages
    """
    for category in USERS:
        tests = TestClassifier(category)
        for i in range(5):
            tests.print_accuracy()


if __name__ == '__main__':
    main()

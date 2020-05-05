# FILENAME: analyzetweets.py

# Uses Tweepy API to read tweets from a user's profile and process their speech habits
# Two modes: analyze and compare
# -analyze: determines most common words used by users
# -compare: predicts which user is most likely to say a given phrase

# Command Line Interface:
# pass in which mode, followed by set of users
# python3 analyzetweets.py -command user1 user2...

import sys
import tweepy
import similarity
import processTweet

# Replace with Twitter developer information
CONSUMER_KEY = ###
CONSUMER_SECRET_KEY = ###
ACCESS_TOKEN = ###
ACCESS_SECRET_TOKEN = ###

VALID_COMMANDS = ['-analyze', '-compare']
NUM_MAX = None   # specify max number of tweets to analyze per user, keep low to reduce runtime, None for no limit


class UserProcessor:
    """
    Class for accessing/processing tweets for a set of entered twitter users
    """
    def __init__(self, users):
        """
        Initializes twitter client authentication with user information and extract info into api variable
        Verifies authentication and reports error if needed
        """
        self.users = users
        self.auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET_KEY)
        self.auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET_TOKEN)
        self.api = tweepy.API(self.auth)

        # Set/up and verify authentication
        try:
            self.api.verify_credentials()
            print("Authentication OK\n")
        except:
            print("Error during authentication\n")
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

    def process_user(self, user):
        """
        Takes in a specific user and creates a word frequencies dictionary with tweets read from that users
        timeline. By default, reads all tweets. Reads NUM_MAX tweets if parameter specified in header.
        """
        # create list of all tweets using user_timeline() method
        word_freq = {}
        tweets = [tweet.text for tweet in self.api.user_timeline(screen_name=user)]

        # processes each tweet
        num_to_analyze = NUM_MAX if NUM_MAX is not None else len(tweets)  # set num_tweets to read
        for i in range(num_to_analyze):
            self.__process_tweet(word_freq, tweets[i])
        return word_freq

    def create_freq(self):
        """
        Creates frequency dictionary with word counts for all users entered
        """
        freq = {}
        for user in self.users:
            freq[user] = self.process_user(user)
        return freq


class Output:
    """
    Class for using processed information to compute predictions and display information
    to a client
    """
    def __init__(self, command, users):
        """
        Initialize word counts dictionary for all users
        """
        self.command = command
        self.users = users  # list of users read in from command line
        self.all_user_freqs = UserProcessor(self.users).create_freq()  # dict of all users w/ word counts
        self.common_words = set()

    def __read_common_words(self, filename="common_words.txt"):
        """
        Creates set of common prepositional phrases to be removed in processing (EX: the, and, or...)
        Reads in data from provided common_words.txt file by default but can be configured
        """
        with open(filename, 'r') as f:
            for line in f:
                self.common_words.add(line.strip())

    def analyze_users(self):
        """
        For all twitter users specified, prints top 5 most used words that are not prepositional
        words (roughly same for all people). Also prints retweet frequency
        """
        self.__read_common_words()
        for user in self.all_user_freqs:
            # Extracts all unique words and sorts in order of most frequent to least
            counts_list = self.all_user_freqs[user]
            words_used = sorted(counts_list.items(), key=lambda pair: pair[1], reverse=True)
            print("\n@" + user + ": Top used non-prepositional words")

            counter = 0
            index = 1
            while counter < 5 and index < len(words_used):
                # extract word/count
                word = words_used[index - 1][0]
                word_count = words_used[index - 1][1]
                # only add if not a common word
                if word not in self.common_words:
                    print("#" + str(index) + ":", word, word_count)
                    counter += 1
                index += 1
            print("Total Number of Retweets:", counts_list['rt'])

    def compare_users(self):
        """
        Given a set of users, prompts client to enter a piece of text and prints a prediction of
        which user is most likely to have that speech pattern. Loops until user specifies stop.
        :return:
        """
        if len(self.users) < 2:
            raise ValueError
        while True:
            # Read text to analyze from user
            text = input("Enter the text to examine (enter exit() to stop): ")
            if text.strip().lower() == "exit()":
                break
            # Calculate similarity using log-similarity functions in similarity.py file
            sim = similarity.UserSimilarity(self.all_user_freqs, text)
            most_sim = sim.most_similar_user()
            print("\nText sample is most similar to:", "@" + most_sim)
            sim.print_similarities()

    def printTweet(self):
        """
        Depending on specified command line argument, runs either analyze or compare
        methods on tweeting data
        """
        if self.command == '-analyze':
            self.analyze_users()
        if self.command == '-compare':
            self.compare_users()
        print()


def main():
    """
    Main function to handle user I/O through command line
    """
    args = sys.argv[1:]
    if len(args) > 1:
        # Extract command to perform, invalid if not -analyze or -compare
        if args[0] not in VALID_COMMANDS:
            print("Command must be -analyze or -compare\n")
            sys.exit()
        users = args[1:]   # Extract users from command line
        output = Output(args[0], users)
        output.printTweet()


if __name__ == '__main__':
    main()







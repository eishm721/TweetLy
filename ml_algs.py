# FILENAME: ml_algs.py

# Implements two machine learning algorithms, Naive Bayes and Logistic Regression
# for classifying which of 2 users is more likely to post a set of tweets

# Naive Bayes computes probability of labels and predicts label with highest
# probability using feature/label combinations
# Logistic Regression predicts labels by learning weights for each feature
# using gradient ascent and applying linear transformation


import sys
import processTweet
import numpy as np

ENGLISH_WORDS_FILE = "word_files/englishwords.txt"


class NaiveBayes:
    """
    Class implementation of a Naive Baye's machine learning classifier
    Can use either maximum likelihood estimation or maximum a posteriori
    estimation with laplace smoothing (specified in constructor, default MAP)
    """
    def __init__(self, use_mle=False):
        """
        Initializes instance variables for training
        """
        self.label_counts = dict()
        self.feature_counts = dict()
        self.use_mle = use_mle  # True for MLE, False for MAP with Laplace add-one smoothing

    def fit(self, train_features, train_labels):
        """
        Trains NB model using 2D numpy matrix of training features (each row
        is a diff sample and each column is a diff feature) and a 1D numpy array
        of training_labels.
        """
        self.label_counts[0] = self.label_counts[1] = 0
        # for each training data point
        for row in range(train_features.shape[0]):
            self.label_counts[train_labels[row]] += 1  # set label

            # for each feature in training data
            for col in range(train_features.shape[1]):
                point = (col, train_features[row][col], train_labels[row])
                if point not in self.feature_counts:
                    self.feature_counts[point] = 0
                self.feature_counts[point] += 1

    def __estimate_arg_max(self, a, prob_a, sample, features, test_features):
        """
        Estimates arg_max of a given sample using provided features.
        Used in prediction
        """
        bay_prob = 1
        for feature in range(test_features.shape[1]):
            value = test_features[sample][feature]
            bay_prob *= features[feature][a][value]
        return bay_prob * prob_a

    def __prob_label(self, i):
        """
        Computes probability of getting a specific label (0, 1) from training data
        """
        return self.label_counts[i] / sum(self.label_counts.values())

    def __prob_cond(self, a, b):
        """
        Computes conditional probability of a specific feature.
        Incorporates laplace smoothing if necessary
        """
        laplace = 0 if self.use_mle else 1
        return (a + laplace) / (a + b + (2 * laplace))

    def predict(self, test_features):
        """
        Predicts 0 or 1 for each sample in a 2D matrix of training features
        Returns 1D numpy array of predictions
        """
        preds = np.zeros(test_features.shape[0], dtype=np.uint8)
        prob_label0 = self.__prob_label(0)
        prob_label1 = self.__prob_label(1)

        # builds array of feature matrices, showing conditional prob of each
        # combination of X_i and Y for all features i
        features = []
        for i in range(test_features.shape[1]):
            x0y0 = self.feature_counts.get((i, 0, 0), 0)
            x1y0 = self.feature_counts.get((i, 1, 0), 0)
            x0y1 = self.feature_counts.get((i, 0, 1), 0)
            x1y1 = self.feature_counts.get((i, 1, 1), 0)
            prob_x0y0 = self.__prob_cond(x0y0, x1y0)
            prob_x0y1 = self.__prob_cond(x0y1, x1y1)
            features.append([(prob_x0y0, 1 - prob_x0y0), (prob_x0y1, 1 - prob_x0y1)])

        # for each sample in testing data, make prediction based on if prob(1) or prob(0) is higher
        for sample in range(test_features.shape[0]):
            prob_y0 = self.__estimate_arg_max(0, prob_label0, sample, features, test_features)
            prob_y1 = self.__estimate_arg_max(1, prob_label1, sample, features, test_features)
            preds[sample] = 0 if prob_y0 > prob_y1 else 1

        return preds


class LogisticRegression:
    """
    Class implementation of a Logistic Regression machine learning classifier
    Can specify learning_rate and max_steps to be taken in gradient ascent
    """
    def __init__(self, learning_rate=0.0001, max_steps=10000):
        """
        Initialize instance variables for training
        """
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weights = None

    def __sigmoid(self, n):
        """
        Return sigmoid function applied to each element of n
        """
        return 1 / (np.exp(-n) + 1)

    def fit(self, train_features, train_labels):
        """
        Trains NB model using 2D numpy matrix of training features (each row
        is a diff sample and each column is a diff feature) and a 1D numpy array
        of training_labels. Initializes weights for model
        """
        # initialize features and weights
        train_features = np.insert(train_features, 0, 1, axis=1)
        theta = np.zeros(train_features.shape[1])  # weights for model

        for iterations in range(self.max_steps):
            gradient = np.zeros(train_features.shape[1])
            # for each n training example, update gradient
            for i in range(train_features.shape[0]):
                diff = train_labels[i] - self.__sigmoid(np.transpose(theta) @ train_features[i])
                gradient += np.multiply(np.full(train_features.shape[1], diff), train_features[i])
            theta += np.multiply(np.full(train_features.shape[1], self.learning_rate), gradient)
        self.weights = theta

    def predict(self, test_features):
        """
        Predicts 0 or 1 for each sample in a 2D matrix of training features
        Returns 1D numpy array of predictions
        """
        test_features = np.insert(test_features, 0, 1, axis=1)
        preds = np.zeros(test_features.shape[0])

        # for each test prediction
        for i in range(test_features.shape[0]):
            preds[i] = 1 if np.transpose(self.weights) @ test_features[i] > 0 else 0
        return preds


class Classifier:
    """
    Class for making predictions using Naive Bayes and Logistic regression
    machine learning models using file of top 58,000 english words and a
    dictionary of 2 users mapped to an array of their tweets.
    MUST ONLY SPECIFY 2 USERS
    """
    def __init__(self, tweets, english_file=ENGLISH_WORDS_FILE):
        """
        Initializes text data and user information
        """
        self.english_file = english_file
        self.tweets = tweets
        self.users = list(self.tweets.keys())

        self.tweet_cleaner = processTweet.CleanTweet()
        self.nb = NaiveBayes()  # use MAP w/ Laplace smoothing b/c many words will not be seen
        self.lr = LogisticRegression()

        self.train_features = self.train_labels = self.test_features = None
        self.english = list()

        if len(tweets) != 2:
            print("ERROR: Must specify exactly 2 users for Naive Bayes or Logistic Regression classification\n")
            sys.exit()

    def __read_english_words(self):
        """
        Creates list of all english words in alpahbetical ordering by reading
        provided english file
        """
        with open(self.english_file, 'r') as f:
            for line in f:
                self.english.append(line.strip())

    def __total_tweets(self):
        """
        Returns the total number of tweets by both users combined in training data
        """
        total = 0
        for user in self.users:
            total += len(self.tweets[user])
        return total

    def __create_matrices(self, test_text):
        """
        Takes tweet dictionaries and converts them into 2D numpy matrices for classification
        Each row in the matrix represents a specific tweet and each column represents all words
        in the english file (~58,000) with a 1 if the word is present in the tweet and 0 if
        it is not. Builds train_features, train_labels, and test_features from a provided text
        """
        # initialize empty arrays and matrices to fill in
        if len(self.english) == 0:
            self.__read_english_words()
        self.train_features = np.zeros(shape=(self.__total_tweets(), len(self.english)), dtype=np.uint8)
        self.train_labels = np.zeros(self.__total_tweets(), dtype=np.uint8)
        self.test_features = np.zeros(shape=(1, len(self.english)), dtype=np.uint8)

        test_words = self.tweet_cleaner.process_string(test_text).split()

        row_count = 0
        for user in self.users:
            for tweet in self.tweets[user]:
                tweet_words = self.tweet_cleaner.process_string(tweet).split()
                # iterate through each english word, treating its presence as a feature
                for col, word in enumerate(self.english):
                    if row_count == 0:
                        # set test features on the first iteration
                        if word in test_words:
                            self.test_features[row_count][col] = 1
                    if word in tweet_words:
                        # set train features
                        self.train_features[row_count][col] = 1
                self.train_labels[row_count] = 0 if user == self.users[0] else 1
                row_count += 1

    def predict_naive_bayes(self):
        """
        Predict 0 or 1 based on naive bayes model on test features
        """
        self.nb.fit(self.train_features, self.train_labels)
        return self.nb.predict(self.test_features)

    def predict_log_reg(self):
        """
        Predict 0 or 1 based on logistic regression model on test features
        """
        self.lr.fit(self.train_features, self.train_labels)
        return self.lr.predict(self.test_features)

    def predict_user(self, alg, text):
        """
        Given a text sample and a machine learning algorithm to use, predicts which user
        (out of 2 provided) is most likely to say that text sample based on twitter history
        """
        self.__create_matrices(text)
        pred = self.predict_log_reg()[0] if alg == "log" else self.predict_naive_bayes()[0]
        return self.users[int(pred)]

    def print_prediction(self, text):
        """
        Prints predictions for both naive bayes and logistic regression models
        """
        print("Naive Bayes:", self.predict_user("naive", text))
        print("Logistic Regression:", self.predict_user("log", text))
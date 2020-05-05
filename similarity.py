# FILENAME: similarity.py

# Probability algorithm implementation to find which user most closely
# matches a set of data based on "Bag of Words" probability model.

# Uses Bayesian model to create indicator for approximating P(writer|unknownTest)
# using P(unknownText|writer). Models probabilities with multinomial
# distribution. Logarithmic similarity index computed to prevent underflow.


import math
import processTweet


class UserSimilarity:
    def __init__(self, user_frequencies, txt):
        self.text = txt
        self.all_user_freqs = user_frequencies
        self.all_user_probs = None
        self.correction = 0.000001

    def create_user_prob_map(self):
        """
        Takes a user frequency map that maps users to a dictionary
        of each word they used and its count and returns a probability map
        replacing counts with frequencies (out of total)
        """
        user_prob_map = {}
        for user in self.all_user_freqs:
            freq_map = self.all_user_freqs[user]
            total_words = sum(freq_map.values())

            # creates new dictionary with frequencies instead of counts
            new_counts = {}
            for word in freq_map:
                new_counts[word] = freq_map[word] / total_words
            user_prob_map[user] = new_counts
        self.all_user_probs = user_prob_map

    def create_word_counts(self):
        """
        Takes a sample of text and returns a map of each word with
        its frequency in the text, along with the number of total words
        """
        # creates cleaned string
        word_counts = {}
        cleaner = processTweet.CleanTweet(self.text)
        original_words = cleaner.process_string().split()

        # creates counts map for cleaned string
        for word in original_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        return word_counts

    def __get_word_prob(self, word_prob_map, word):
        """
        Takes a probability map for a given user mapping words to their probabilities
        (modeled by frequencies) and return the probability. Returns small value
        1e-6 if not found to account for very small probability, instead of 0
        """
        if word in word_prob_map:
            return word_prob_map[word]
        return self.correction

    def __calc_log_similarity(self, user):
        """
        Takes a user and reports their logarithmic similarity index compared to the
        instance text variable. Implements multinomial random variable probabilistic
        distributions with Bayesian dynamics
        """
        # extract user words probabilities and text word counts
        curr_user_probs = self.all_user_probs[user]
        text_word_counts = self.create_word_counts()
        log_prob = 0

        # for each word, add logarithmic multinomial computation
        for word_i in text_word_counts:
            count_i = text_word_counts[word_i]
            prob_i = self.__get_word_prob(curr_user_probs, word_i)
            log_prob += count_i * math.log(prob_i)
        return log_prob

    def __get_similarities(self):
        """
        Returns a sorted list of tuples containing a user matched with its similarity index
        from highest similarity to lowest
        """
        # initializes probability map of word frequencies for all users
        if self.all_user_probs is None:
            self.create_user_prob_map()

        # adds pair of user and its similarity
        similarities = []
        for user in self.all_user_probs:
            similarities.append((user, self.__calc_log_similarity(user)))
        return sorted(similarities, key=lambda pair: pair[1], reverse=True)  # sorted reverse on similarity

    def most_similar_user(self):
        """
        Returns the user with the highest similarity index out of all users to the instance text
        """
        return self.__get_similarities()[0][0]

    def print_similarities(self):
        """
        Prints all users with their similarity index from highest to lowest
        """
        similarities = self.__get_similarities()
        low_bound = -1.5 * similarities[-1][1]  # logarithmic transformation constant - lowest similarity
        for user, log_sim in similarities:
            print(user, round(log_sim + low_bound, 3))


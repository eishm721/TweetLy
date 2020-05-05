# FILENAME: processTweet.py

# Collection of functions for processing a line of text in a tweet to remove
# irrelevant data such as links and tags


class CleanTweet:
    def __init__(self, tweet):
        self.text = tweet

    def process_word(self, word):
        """
        Takes in a specific word and "cleans" it, converting to lowercase, removing
        user handles and links, and leading/ending nonalphabetical characters
        """
        curr_word = word.lower()
        # remove user handles or links
        if curr_word[0] == '@' or curr_word[0:4] == "http":
            return ""

        # removing leading nonalpha chars
        first = 0
        while first < len(curr_word) and not curr_word[first].isalpha():
            first += 1
        if first >= len(curr_word):
            return ""

        # remove ending nonalpha chars
        last = len(curr_word) - 1
        while last >= 0 and not curr_word[last].isalpha():
            last -= 1
        return curr_word[first:last + 1]

    def process_string(self):
        """
        Takes in a string and performs process_word operation in each word (separated with a space)
        in the string and returns a "cleaned", lowercase string.
        """
        cleaned = []
        original_words = self.text.split()   # words separated w/ space
        for word in original_words:
            processed = self.process_word(word)
            # if valid cleaned word
            if processed != "":
                cleaned.append(processed)
        return ' '.join(cleaned)

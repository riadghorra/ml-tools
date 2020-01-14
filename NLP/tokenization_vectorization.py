"""Tokenization"""

import nltk
# Dictionnaire
from nltk.corpus import brown

# Fonctions de tokenization
from nltk import sent_tokenize, word_tokenize

# Stopwords et ponctuation
from nltk.corpus import stopwords
from string import punctuation

stopwords_en = set(stopwords.words('english'))
stopwords_en_withpunct = stopwords_en.union(set(punctuation))

# Lemmatization
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()


def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'  # if mapping isn't found, fall back to Noun.


def lemmatize_sent(text):
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(word_tokenize(text))]


def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    return [word for word in lemmatize_sent(text) if word not in stopwords_en_withpunct and not word.isdigit()]


""" Vectorization"""
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vect = CountVectorizer(analyzer=preprocess_text)
tfidf_transformer = TfidfTransformer()

def transform_train_test(X_train, X_test):
    # On fit et transforme le train
    train_set = count_vect.fit_transform(X_train)

    # On transforme uniquement le test
    test_set = count_vect.transform(X_test)

    tfidf_train = tfidf_transformer.fit_transform(train_set)
    tfidf_test = tfidf_transformer.fit_transform(test_set)

    return tfidf_train, tfidf_test
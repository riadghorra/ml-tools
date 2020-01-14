"""This file contains visualizations such as WordCloud"""

from nltk.corpus import stopwords
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pylplot as plt


def build_wordcloud(list_of_texts):
    stopwords_en = set(stopwords.words('english'))
    stopwords_en_withpunct = stopwords_en.union(set(punctuation))
    text = " ".join(review for review in list_of_texts)
    print("There are {} words in the combination of all reviews.".format(len(text)))

    wordcloud = WordCloud(stopwords=stopwords_en_withpunct, background_color="white").generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

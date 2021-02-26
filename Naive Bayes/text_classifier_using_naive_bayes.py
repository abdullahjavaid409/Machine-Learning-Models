"""Multinomial naive bayes is most appropiate for features that represent counts
because multinomial distribution describes the probablity of counts that occur among
different categories. And thus we can use it for the classification of text and identify that
our text belong to which class. Here we'll classify newsgroups corpus."""

# importing necessary libs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names

"""Multinomial naive bayes is most appropiate for features that represent counts
because multinomial distribution describes the probablity of counts that occur among
different categories. And thus we can use it for the classification of text and identify that
our text belong to which class. Here we'll classify newsgroups corpus."""

# importing necessary libs
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

# loading corpus of text
data = fetch_20newsgroups()
data.target_names


categories = ['sci.space', 'soc.religion.christian', 'sci.space', 'talk.politics.guns']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

train.data[3]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)  # traingin the model

train.data[9]  # gives the data in the corpus
train.target[9]  # gives the labels for the data

test_pred = model.predict(test.data)  # make predictions on the test data

confusion_mat = confusion_matrix(test.target, test_pred)
sns.heatmap(confusion_mat, fmt='d', annot=True,
            xticklabels=train.target_names, yticklabels=train.target_names)
# print(sns.heatmap.__doc__)

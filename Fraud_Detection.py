# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Loading data
df = pd.read_csv("E:\DataScience & AI\Github_repo\creditcard.csv\creditcard.csv")
df.shape
df.describe()
df.info


df.Class.value_counts()  # gives the number of 1's and 0's in the the class column

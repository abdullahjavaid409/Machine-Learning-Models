"""Here we'll explore the dataset for the credit card faraud detection using naive bayes.
And predict what will the fraudelent transactions!"""

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

val_count = df.Class.value_counts()  # gives the number of 1's and 0's in the the class column
# visualizing class column
# Class column specifies that the traction belong to which class
# 1 = Fraudelent transaction; 0 = Non-fradulaent  transaction
print("Class column:")
fig, ax = plt.subplots(1, 1)
# here "%1.1f%%" says represents width of 1 and precision of 1 in the output
ax.pie(val_count, explode=[0, 0.1], autopct='%1.1f%%',
       labels=['Not fraud', 'Fraud'], colors=['orange', 'blue'])
plt.axis('equal')
# print(plt.pie.__doc__)
# ----------------

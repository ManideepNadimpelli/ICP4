import pandas as pd


# visualization
import seaborn as sns
import matplotlib.pyplot as plt

##reading data set
train_df = pd.read_csv('/content/train.csv')



train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} )
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(train_df['Survived'].corr(train_df['Sex']))

##Analyze by visualizing data
####Correlating numerical features
train_df.corr()
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Sex', bins=20)
plt.show()
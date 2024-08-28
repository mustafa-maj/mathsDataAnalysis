# rr = Recycling Rate, gr= GCSE grade %, mdi = median income, mi = mean income, mdh = median house price
# er = employment rate, Area, Inner or Outer London, Area code
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# Reading the file MathsData

csv = pd.read_csv('MathsData.csv')

# Cleaning the file, and saving it into variable new_csv

new_csv = csv.drop([33, 34, 35, 36, 37])
new_csv = new_csv.drop(['Area code', 'Unnamed: 34', 'Unnamed: 66', 'Unnamed: 76', 'Unnamed: 50', 'Unnamed: 18'], axis=1)
# print(new_csv.head())
data = new_csv[['er2005', 'rr2005/06', 'mdi2005/06', 'mi2005/06', 'mdh2005']]

# using linear regression model for predicting. Finding the correlation between er2005 and rr2005/06

prediction = 'er2005'
X = new_csv[['rr2005/06', 'mdh2005', 'mi2005/06']] # features
y = new_csv[prediction] # labels
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
linear = linear_model.LinearRegression().fit(X_train, y_train)
modelScore = linear.score(X, y)
print('Test Score: ', linear.score(X_test, y_test))
print('Train Score: ', linear.score(X_train, y_train))
print(linear.predict([[0, 1000000, 1000000]]))
"""plt.scatter(X_train, y_train)
plt.xlabel('Employment Rate (2005)')
plt.ylabel('Recycling rate (2005/06)')
plt.title('Finding the correlation between recycling rate and employment rate')
plt.plot(X_train, linear.predict(X_train))
print('Score of Test data: ', linear.score(X_test, y_test))
print('Score of Training data: ', linear.score(X_train, y_train))"""
"""predictionNumber = 0
predict = linear.predict([[predictionNumber]])
print('If the employment rate is ', predictionNumber, ' then the prediction for the recycling rate is', predict)
plt.show()
"""

cityCount = new_csv.value_counts(['Inner or Outer London'])

# inner and outer cities data only.
"""
outer = new_csv[new_csv['Inner or Outer London'] == 'Outer']
inner = new_csv[new_csv['Inner or Outer London'] == 'Inner']
"""
# boxplot for er2005 with mdi/other stats plots.

boxplot = sns.boxplot(y='er2007', data=new_csv)
distancePlot = sns.distplot(new_csv['er2007'])
histogram = new_csv.hist()
barChart = pd.value_counts(new_csv['Inner or Outer London']).plot.bar()
group = new_csv.groupby(['Area', 'Inner or Outer London']).mean()
corr = new_csv.corr()
heatmap = sns.heatmap(corr)
scatGraph = sns.relplot(x='mdi2005/06', y='mdh2005', hue='Inner or Outer London', data=new_csv)
catplot = sns.catplot(x='mdi2005/06', y='mdh2005', hue='Inner or Outer London', data=new_csv, kind='box')
plt.show()

# using k-nearest neighbour algorithm (doesn't work with data lmao)

model = KNeighborsClassifier(n_neighbors=12)
le = preprocessing.LabelEncoder()
bins = [0, 50, 70, 80, 100]
incomeGroups = [0, 1, 2, 3]
new_csv['er2005'] = pd.cut(new_csv['er2005'], bins, labels=incomeGroups, include_lowest=True)
X = new_csv[['er2005']]
binsRR = [0, 20, 40, 50, 100]
incomeGroupsRR = [0, 1, 2, 3]
new_csv['rr2005/06'] = pd.cut(new_csv['rr2005/06'], binsRR, labels=incomeGroupsRR, include_lowest=True)
y = new_csv[['rr2005/06']]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
graph = sns.scatterplot(x=new_csv['er2005'], y=new_csv['rr2005/06'], data=new_csv)
plt.show()
print(model.predict([[0]]))
print(acc)


# data wrangling

InnerMean = inner['er2005'].mean()
OuterMean = outer['er2005'].mean()
replaceER2005 = new_csv['er2005'].replace(np.nan, InnerMean)
minmaxTest = (new_csv['er2005']-new_csv['er2005'].min()) / (new_csv['er2005'].max()-new_csv['er2005'].min())
dummyLondon = pd.get_dummies(new_csv['Inner or Outer London'])
bins = [50, 60, 70, 80, 100]
incomeGroups = ['bad', 'okay', 'good', 'vgood']
employment = pd.cut(new_csv['er2005'], bins, labels=incomeGroups, include_lowest=True)
print(employment)

max = int(inner['mdh2008'].max())
min = int(inner['mdh2008'].min())
range = max - min
print(range)
scatGraph = sns.relplot(x='mdh2005', y='rr2005/06', hue='Inner or Outer London', data=new_csv)
print(scatGraph)
plt.show()
sns.relplot(
    data=new_csv, x="mdh2005", y="rr2005/06", col="Inner or Outer London",
    hue="Inner or Outer London", style="Inner or Outer London", kind="line",)

# image recognition

digits = load_digits()

X = digits.data  # getting the features
y = digits.target  # getting the labels

plt.figure(figsize=(20, 4))
for i, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
    plt.subplot(1, 10, i + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

log_reg = LogisticRegression().fit(X_train, y_train)

predict = log_reg.predict(X_test)
score = log_reg.score(X_test, y_test)
confusionMatrix = metrics.confusion_matrix(y_test, predict)

plt.figure(figsize=(9, 9))
sns.heatmap(confusionMatrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Spectral')
plt.ylabel('Actual Output')
plt.xlabel('Predicted')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
"""

# decision tree classifier

X = new_csv[['rr2005/06']]
y = new_csv['rr2006/07']
model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
score = accuracy_score(prediction, y_test)
print(score)
plot_tree(model, filled=True)
plt.title('Recycling rate comparison')
plt.savefig('tiff_compressed.tiff', dpi=600, format='tiff', facecolor='white', edgecolor='none', pil_kwargs={'compression':'tiff_lzw'})

""""""
# random forest model

X = new_csv[['rr2005/06']]
y = new_csv['rr2006/07']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
prediction = model.predict(X_test)
score = model.score(X_test, y_test)
cm = metrics.confusion_matrix(y_test, prediction)
print(cm)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Spectral')
plt.ylabel('Actual Outputs')
plt.xlabel('Predicted Outputs')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()


# polynomial regression using np

X = new_csv['rr2005/06']
y = new_csv['er2006']
model = np.poly1d(np.polyfit(X, y, 3))
line = np.linspace(0, 40, 100)
plt.scatter(X, y)
plt.plot(line, model(line))
plt.show()
print(r2_score(y, model(X)))
predict = model(10)

# polynomial regression using sklearn

X = new_csv[['rr2005/06']]
y = new_csv['er2006']
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
pol = LinearRegression()
pol.fit(X_poly, y)
plt.scatter(X, y, color='red')
plt.plot(X, pol.predict(poly.fit_transform(X)), color='blue')
predict = pol.predict(poly.fit_transform([[10]]))
print(predict)
plt.show()


# bayes classification

X = new_csv[['er2005']]
y = new_csv['Inner or Outer London']
model = GaussianNB().fit(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_pred = model.predict(X_test)
print(y_pred)
score = metrics.accuracy_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Spectral')
plt.ylabel('Actual Outputs')
plt.xlabel('Predicted Outputs')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

# Support vector machines

X = new_csv[['er2005']]
y = new_csv['Inner or Outer London']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
# what is recall score?
print(score)
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Spectral')
plt.ylabel('Actual Outputs')
plt.xlabel('Predicted Outputs')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
"""
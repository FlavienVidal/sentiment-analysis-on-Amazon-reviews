import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import warnings
import random
import math
import matplotlib.style as style
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("whitegrid")


# # # # CONTROLLING AESTHETICS # # # #
sns.set(context="paper", style="darkgrid", palette="deep")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# # # # LOAD THE DATA # # # #
print("--------------------------------")
df1 = pd.read_csv("1429_1.csv", sep=",")
df2 = df1[['reviews.title', 'reviews.rating', 'reviews.text']]
df_reviews_rating = df1[['reviews.rating']]
df_reviews_numHelpful = df1[['reviews.numHelpful']]
print("Total number of reviews: ", len(df1), "\n")
print(df1.head().to_string(), "\n")
print(df2.head().to_string(), "\n")
print(df1.info())
print("Reviews ratings: \n: ", df_reviews_rating.head().to_string(), "\n")
print("Reviews examples: \n: ", df1['reviews.text'].head().to_string(), "\n")
print("Worst reviews: \n", df1.sort_values(by='reviews.rating')['reviews.text'].head(), "\n")


# # # # UNDERSTANDING AND IDENTIFYING A GIVEN PRODUCT # # # #
print("--------------------------------")
data = df1.copy()
print("Number & List of all different id: ", len(data['id'].unique()), " & ", data['id'].unique(), "\n")
print("Number & List of all different names: ", len(data['name'].unique()), " & ", data['name'].unique(), "\n")
print("Number & List of all different Amazon Identifiers: ", len(data["asins"].unique()), " & ", data["asins"].unique(), "\n")
print("Number & List of all different keys: ", len(data["keys"].unique()), " & ", data["keys"].unique(), "\n")

print("--------------------------------")
print("DATA DESCRIPTION:")
print(data.describe(), "\n")
print("DATA INFORMATION:")
print(data.info(), "\n")


# # # # VISUALIZE THE RATINGS OF PRODUCTS & NUMBER OF USEFUL REVIEWS # # # #
print("--------------------------------")

# Pie chart
labels_classes = ['1', '2', '3', '4', '5']
labels_names = ['Rating = 1', 'Rating = 2', 'Rating = 3', 'Rating = 4', 'Rating = 5']
print("Count per classes: \n", df1['reviews.rating'].value_counts())
print("Shape = ", df1['reviews.rating'].value_counts().shape)
print("Len = ", len(df1['reviews.rating'].value_counts()))
sizes = [410, 402, 1499, 8541, 23775]
colors = ['cadetblue', 'orange', 'yellowgreen', 'indianred', 'mediumpurple']
fig, ax = plt.subplots()
ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
ax.legend(labels_names, loc="upper right")
ax.axis('equal')
plt.title('Quantity of examples per classes', fontsize=15, weight='bold')
plt.show()

# Reviews ratings distribution
df_reviews_rating.hist(grid=True, bins=25, color='darkslategray', alpha=0.8)
plt.title('Average rating distribution on the whole stock', fontsize=15, weight='bold')
plt.xlabel('rating')
plt.ylabel('number of ratings')
plt.show()

# Number of helpful reviews distribution
df_reviews_numHelpful.hist(grid=True, bins=25, color='olive', alpha=0.8)
plt.title('Number of useful reviews distribution on Amazon products', fontsize=15, weight='bold')
plt.xlabel('number of people who find the review useful')
plt.ylabel('quantity')
plt.show()

print('Are all values in column reviews.numHelp equal to 0: ', (df_reviews_numHelpful == 0).all())
print("Average rating per product: ", data.groupby("id")["reviews.rating"].mean())

for i in data['id'].unique():
    print(type(data['reviews.rating'].mean()))
    if data['reviews.rating'].mean().astype(int) < 4:
        print("The article can be dropped from the store")


# # # # TRAINING SET - TEST SET # # # #
print("--------------------------------")

# remove all products that doesn't have any "reviews.rating"
data = df1.copy()
print("Original dataset {}".format(len(data)))
data_new = data.dropna(subset=["reviews.rating"])  # removes all NAN in reviews.rating
print("New dataset {}".format(len(data_new)))
data_new["reviews.rating"] = data_new["reviews.rating"].astype(int)

print("--------------------------------")
split = StratifiedShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2)
for train_index, test_index in split.split(data_new, data_new["reviews.rating"]):
    stratified_training = data_new.reindex(train_index)
    stratified_test = data_new.reindex(test_index)
print("len(stratified_training) =", len(stratified_training), " | len(stratified_test) =", len(stratified_test))

strat_train_set_percent_per_cat = stratified_training["reviews.rating"].value_counts()/len(stratified_training)
strat_test_set_percent_per_cat = stratified_test["reviews.rating"].value_counts()/len(stratified_test)
df_percent_per_cat = data_new["reviews.rating"].value_counts()/len(data_new)
df_reviews_rating_cat_prop = pd.DataFrame({'Df': df_percent_per_cat, 'Strat Train Set': strat_train_set_percent_per_cat,
                                           'Strat Test Set': strat_test_set_percent_per_cat})
print("Comparison of the proportions in each set \n", df_reviews_rating_cat_prop.to_string())

reviews = stratified_training.copy()

plt.figure()  # figsize=(20, 7)
reviews["id"].value_counts().plot(kind="bar", color=['b', 'c', 'g', 'y', 'r', 'm'], alpha=0.6)
plt.title("Number of products per id", fontsize=15, weight='bold')
plt.xlabel("Product id")
plt.ylabel("Number of products")
plt.show()

# Entire training dataset average rating
reviews["reviews.rating"].mean()


# # # # SENTIMENT ANALYSIS # # # #
print("--------------------------------")


def sentiments(rating):
    if rating == 5:
        return "Very Positive"
    elif rating == 4:
        return "Neutral"
    elif rating == 3:
        return "Neutral"
    elif rating == 2:
        return "Negative"
    elif rating == 1:
        return "Very Negative"


# def sentiments(rating):
#     if (rating == 5) or (rating == 4):
#         return "Positive"
#     elif rating == 3:
#         return "Neutral"
#     elif (rating == 2) or (rating == 1):
#         return "Negative"


# Add a column of sentiment to the dataframe
stratified_training["Sentiment"] = stratified_training["reviews.rating"].apply(sentiments)
stratified_test["Sentiment"] = stratified_test["reviews.rating"].apply(sentiments)

# Prepare data
X_train = stratified_training["reviews.text"]
Y_train = stratified_training["Sentiment"]
X_test = stratified_test["reviews.text"]
Y_test = stratified_test["Sentiment"]


# # # # EXTRACT FEATURES - BAG OF WORDS MODEL # # # #
print("--------------------------------")

# Replace "nan" with space
X_train = X_train.fillna(' ')
X_test = X_test.fillna(' ')
Y_train = Y_train.fillna(' ')
Y_test = Y_test.fillna(' ')

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
print("X_train_counts = ", X_train_counts, "\n")
print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

print("--------------TEST------------------")
corpus = ['This is the first document.', 'This document is the second document.', 'And this is the third one.',
          'Is this the first document?']
X = count_vect.fit_transform(corpus)
print("count_vect.get_feature_names() : ", count_vect.get_feature_names())
print("X.toarray() : \n", X.toarray())
print("X.shape = ", X.shape)
print("X = ", X)
print("--------------TEST------------------")


# # # # PIPELINE: SELECT AND TRAIN MODELS # # # # #
print("--------------------------------")

# MULTINOMIAL NAIVE BAYES
clf_multiNB_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_nominalNB", MultinomialNB())])
clf_multiNB_pipe.fit(X_train, Y_train)
predictedMultiNB = clf_multiNB_pipe.predict(X_test)
clf_multiNB_pipe_training_score = clf_multiNB_pipe.score(X_train, Y_train)
clf_multiNB_pipe_test_score = clf_multiNB_pipe.score(X_test, Y_test)
print('Accuracy of the Multinomial NB classifier on training set: {:.2f}'.format(clf_multiNB_pipe_training_score))
print('Accuracy of the Multinomial NB classifier on test set: {:.2f}'.format(clf_multiNB_pipe_test_score))

X = np.array(X_test)
print("Test input feature       Predicted sentiment")
for x in range(int(len(predictedMultiNB)/200)):
    print(X[x], predictedMultiNB[x])

# LOGISTIC REGRESSION
print("--------------------------------")
clf_logReg_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_logReg", LogisticRegression())])
clf_logReg_pipe.fit(X_train, Y_train)
predictedLogReg = clf_logReg_pipe.predict(X_test)
clf_logReg_pipe_training_score = clf_logReg_pipe.score(X_train, Y_train)
clf_logReg_pipe_test_score = clf_logReg_pipe.score(X_test, Y_test)
print('Accuracy of the logistic regression classifier on training set: {:.2f}'.format(clf_logReg_pipe_training_score))
print('Accuracy of the logistic regression classifier on test set: {:.2f}'.format(clf_logReg_pipe_test_score))

# LINEAR SUPPORT VECTOR MACHINE
print("--------------------------------")
clf_linearSVC_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_linearSVC", LinearSVC())])
clf_linearSVC_pipe.fit(X_train, Y_train)
predictedLinearSVC = clf_linearSVC_pipe.predict(X_test)
clf_linearSVC_pipe_training_score = clf_linearSVC_pipe.score(X_train, Y_train)
clf_linearSVC_pipe_test_score = clf_linearSVC_pipe.score(X_test, Y_test)
print('Accuracy of the linear SVM classifier on training set: {:.2f}'.format(clf_linearSVC_pipe_training_score))
print('Accuracy of the linear SVM classifier on test set: {:.2f}'.format(clf_linearSVC_pipe_test_score))

# DECISION TREE
print("--------------------------------")
clf_decisionTree_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()),
                                  ("clf_decisionTree", DecisionTreeClassifier())])
clf_decisionTree_pipe.fit(X_train, Y_train)
predictedDecisionTree = clf_decisionTree_pipe.predict(X_test)
clf_decisionTree_pipe_training_score = clf_decisionTree_pipe.score(X_train, Y_train)
clf_decisionTree_pipe_test_score = clf_decisionTree_pipe.score(X_test, Y_test)
print('Accuracy of the Decision Tree classifier on training set: {:.2f}'.format(clf_decisionTree_pipe_training_score))
print('Accuracy of the Decision Tree classifier on test set: {:.2f}'.format(clf_decisionTree_pipe_test_score))

# RANDOM FOREST
print("--------------------------------")
clf_randomForest_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_randomForest", RandomForestClassifier())])
clf_randomForest_pipe.fit(X_train, Y_train)
predictedRandomForest = clf_randomForest_pipe.predict(X_test)
clf_randomForest_pipe_training_score = clf_randomForest_pipe.score(X_train, Y_train)
clf_randomForest_pipe_test_score = clf_randomForest_pipe.score(X_test, Y_test)
print('Accuracy of the Random Forest classifier on training set: {:.2f}'.format(clf_randomForest_pipe_training_score))
print('Accuracy of the Random Forest classifier on test set: {:.2f}'.format(clf_randomForest_pipe_test_score))

# Ranking of models by test accuracy
print("--------------------------------")
classifiers = ["MultiNB", "LogReg", "LinearSVM", "DecTree", "RandFor"]
training_scores = [clf_multiNB_pipe_training_score, clf_logReg_pipe_training_score, clf_linearSVC_pipe_training_score,
                   clf_decisionTree_pipe_training_score, clf_randomForest_pipe_training_score]
test_scores = [clf_multiNB_pipe_test_score, clf_logReg_pipe_test_score, clf_linearSVC_pipe_test_score,
               clf_decisionTree_pipe_test_score, clf_randomForest_pipe_test_score]

n_groups = len(classifiers)
bar_width = 0.10

fig, ax = plt.subplots()
index = np.arange(n_groups)
training_acc = ax.bar(index, training_scores, bar_width, color='darkred', alpha=0.6, label="Training Accuracy")
test_acc = ax.bar(index + bar_width, test_scores, bar_width, color='darkgreen', alpha=0.6, label="Test Accuracy")
ax.set_title('Ranking of models by test accuracy', fontsize=15, weight='bold')
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(classifiers)
ax.legend()
plt.show()

fig, ax = plt.subplots()
index = np.arange(n_groups)
training_acc = plt.barh(index, training_scores, bar_width, color='darkred', alpha=0.6, label="Training Accuracy")
test_acc = plt.barh(index + bar_width, test_scores, bar_width, color='darkgreen', alpha=0.6, label="Test Accuracy")
ax.set_title('Ranking of models by test accuracy', fontsize=15, weight='bold')
ax.set_xlabel('Accuracy')
ax.set_ylabel('Classifiers')
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(classifiers)
ax.legend()
plt.show()


# # # # FINE TUNE THE MODEL # # # #
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
             }
gs_linearSVC_pipe = GridSearchCV(clf_linearSVC_pipe, parameters, n_jobs=-1)
gs_linearSVC_pipe = gs_linearSVC_pipe.fit(X_train, Y_train)


# # # # PREDICTIONS # # # #
print("--------------------------------")
target = pd.Series(['I love this product. I am very happy!', 'I like the product.',
                    'I like it but it’s too large', 'I don’t really like it', 'A bit confused',
                    'Bad', 'worst'])

target = target.fillna(' ')
count_vect = CountVectorizer()
target_counts = count_vect.fit_transform(target)
# print("target_counts = \n", target_counts.toarray(), "\n")
tfidf_transformer = TfidfTransformer(use_idf=False)
target_tfidf = tfidf_transformer.fit_transform(target_counts)
# print("target_tfidf = \n", target_tfidf.toarray())

predictions_multiNB = clf_multiNB_pipe.predict(target)
predictions_logreg = clf_logReg_pipe.predict(target)
predictions_linearSVC = clf_linearSVC_pipe.predict(target)
predictions_decisiontree = clf_decisionTree_pipe.predict(target)
predictions_randomforest = clf_randomForest_pipe.predict(target)
predictions_gridsearch_linearSVC = gs_linearSVC_pipe.predict(target)

target_array = np.array(target)

df_predicted_sentiments = pd.DataFrame({'Reviews': target_array, 'Multi NB': predictions_multiNB,
                                        'Log Reg': predictions_logreg,
                                        'Linear SVC': predictions_linearSVC,
                                        'Decision Tree': predictions_decisiontree,
                                        'Random Forest': predictions_randomforest,
                                        'GSCV Lin SVC': predictions_gridsearch_linearSVC})

print("--------------------------------")
print("Some prediction examples: \n", df_predicted_sentiments.to_string())
print("--------------------------------")










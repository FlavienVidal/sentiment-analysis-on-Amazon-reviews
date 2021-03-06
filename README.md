# Sentiment Analysis on Amazon Reviews



## What is opinion mining?

In computer science, opinion mining also referred to as sentiment analysis is an approach of natural language processing referring to an analysis that identifies the tone emerging from a body of text on large amounts of data. It aims to determine an evaluation, a judgement or more overall an emotional state with regard to a given topic.

Opinion mining is a growing and promising research field widely applied to reviews and social media to categorize opinions.



## Why is it so useful nowadays?

In the recent years, we have been witnessing the explosion of what is commonly referred to as participatory sensing. Ordinary people take a proactive role in posting comments and complaints online, increasingly using technology to record information about events and issues in all dimensions of their political and social life.  It can be about a political candidate they like, an article they didn't like, ...
Hence the reason why opinion mining approaches are seen as the cornerstones of largescale collaborative policymaking. 

*	Political Sphere: 
> In the context of political activism, everyone is not only eager to share their political orientation but also curious to know the voice of others. Social media platforms make it easy to capture the numerous aspects of public opinion. These sites have begun to have a significant impact on how people think and act and thus are becoming essential to provide support to the verification of population trends in politics domain. Extracting citizens' opinions on government decisions from social media is therefore an increasingly important issue.

*	Retail Sales
> With the evolution of traditional stores to online shopping over the time, product reviews are becoming more and more important. Consumers around the world are sharing reviews directly on product pages in real time creating one huge database which is constantly being updated. For example, the amount of reviews on Amazon has increased tremendously over the past years. This vast amount of consumer reviews creates an opportunity for businesses to see how the market reacts to a specific product and how a given company can adapt its stocks. If businesses are able to categorize products according to certain patterns based on their reviews it could help them choose which type of products should be dropped from their stock, which one should they keep, and which others could they get. For example, if a group of different suitcases is highly rated and that people seem to like the materials, the dimensions and the colors, it probably means that products with similar proprieties should be kept.



## Project

The objective of this project is to analyze the correlation between the Amazon product reviews and the rating of the products given by consumers. I would like to create a supervised learning model able to classify a given customer review as positive, neutral and negative, and thus affect an overall score to a given review: is the consumer happy of his purchase? Is he disappointed? Is he just neutral? 

![pos_neg_neut](pos_neg_neut.png)



## Difficulties
It is possible to give a computer the ability to understand the overall emotion emanating from a body of text, but it needs to understand which are the most influential words in the text, which word should be priotarized, and it should be able to understand the meaning of successions of words that can mean a whole different thing than taken apart. Indeed, meaning is built from ordered sequences of words that refer to each other but the vast majority of text analyses techniques ignore this complexity and rely only on counts on language tokens (words, sentences, ...).
Also, it is very difficult for a computer to recognize sarcasm. 
Since the data is written by different customers, there may be various typos, nonstandard spellings, and other variations that may not be found in curated sets of published text. 
Comments must all be written in the same language (English in our case). 
The last difficulty concerns the dataset: it must contain as many negative, neutral and positive examples in order to train our model correctly.

Lastly, the process of *tokenization* results in very high dimensional matrices: the more text data you accumulate, the larger the dimensionality of the dataset. We are never in a safe place of having as many more observations than variables! 



## Dataset
The dataset I will be using comes from Kaggle.com and is a sample of a larger dataset available through Datafiniti. It lists 34,660 consumer reviews of Amazon manufactured products including their ratings, reviews, names and more.

Let’s first have a quick look at the top five rows of this dataset and get a description and distribution of the data:
```
                     id                                                                                     name       asins   brand                                                                       categories                                                                                                                                                                                                                 keys manufacturer              reviews.date     reviews.dateAdded                                   reviews.dateSeen reviews.didPurchase reviews.doRecommend  reviews.id  reviews.numHelpful  reviews.rating                                                                                                                                         reviews.sourceURLs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           reviews.text                            reviews.title  reviews.userCity  reviews.userProvince reviews.username
0  AVqkIhwDv8e3D1O-lebb  All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Magenta  B01AHB9CN2  Amazon  Electronics,iPad & Tablets,All Tablets,Fire Tablets,Tablets,Computers & Tablets  841667104676,amazon/53004484,amazon/b01ahb9cn2,0841667104676,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/5620406,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/b01ahb9cn2       Amazon  2017-01-13T00:00:00.000Z  2017-07-03T23:33:15Z  2017-06-07T09:04:00.000Z,2017-04-30T00:45:00.000Z                 NaN                True         NaN                 0.0             5.0  http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=200,http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=166                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        This product so far has not disappointed. My children love to use it and I like the ability to monitor control what content they see with ease.                                   Kindle               NaN                   NaN          Adapter
1  AVqkIhwDv8e3D1O-lebb  All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Magenta  B01AHB9CN2  Amazon  Electronics,iPad & Tablets,All Tablets,Fire Tablets,Tablets,Computers & Tablets  841667104676,amazon/53004484,amazon/b01ahb9cn2,0841667104676,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/5620406,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/b01ahb9cn2       Amazon  2017-01-13T00:00:00.000Z  2017-07-03T23:33:15Z  2017-06-07T09:04:00.000Z,2017-04-30T00:45:00.000Z                 NaN                True         NaN                 0.0             5.0  http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=200,http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=167                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            great for beginner or experienced person. Bought as a gift and she loves it                                very fast               NaN                   NaN           truman
2  AVqkIhwDv8e3D1O-lebb  All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Magenta  B01AHB9CN2  Amazon  Electronics,iPad & Tablets,All Tablets,Fire Tablets,Tablets,Computers & Tablets  841667104676,amazon/53004484,amazon/b01ahb9cn2,0841667104676,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/5620406,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/b01ahb9cn2       Amazon  2017-01-13T00:00:00.000Z  2017-07-03T23:33:15Z  2017-06-07T09:04:00.000Z,2017-04-30T00:45:00.000Z                 NaN                True         NaN                 0.0             5.0  http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=200,http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=167                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Inexpensive tablet for him to use and learn on, step up from the NABI. He was thrilled with it, learn how to Skype on it already...  Beginner tablet for our 9 year old son.               NaN                   NaN            DaveZ
3  AVqkIhwDv8e3D1O-lebb  All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Magenta  B01AHB9CN2  Amazon  Electronics,iPad & Tablets,All Tablets,Fire Tablets,Tablets,Computers & Tablets  841667104676,amazon/53004484,amazon/b01ahb9cn2,0841667104676,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/5620406,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/b01ahb9cn2       Amazon  2017-01-13T00:00:00.000Z  2017-07-03T23:33:15Z  2017-06-07T09:04:00.000Z,2017-04-30T00:45:00.000Z                 NaN                True         NaN                 0.0             4.0  http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=200,http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=167                      I've had my Fire HD 8 two weeks now and I love it. This tablet is a great value.We are Prime Members and that is where this tablet SHINES. I love being able to easily access all of the Prime content as well as movies you can download and watch laterThis has a 1280/800 screen which has some really nice look to it its nice and crisp and very bright infact it is brighter then the ipad pro costing $900 base model. The build on this fire is INSANELY AWESOME running at only 7.7mm thick and the smooth glossy feel on the back it is really amazing to hold its like the futuristic tab in ur hands.                                  Good!!!               NaN                   NaN           Shacks
4  AVqkIhwDv8e3D1O-lebb  All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Magenta  B01AHB9CN2  Amazon  Electronics,iPad & Tablets,All Tablets,Fire Tablets,Tablets,Computers & Tablets  841667104676,amazon/53004484,amazon/b01ahb9cn2,0841667104676,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/5620406,allnewfirehd8tablet8hddisplaywifi16gbincludesspecialoffersmagenta/b01ahb9cn2       Amazon  2017-01-12T00:00:00.000Z  2017-07-03T23:33:15Z  2017-06-07T09:04:00.000Z,2017-04-30T00:45:00.000Z                 NaN                True         NaN                 0.0             5.0  http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=200,http://reviews.bestbuy.com/3545/5620406/reviews.htm?format=embedded&page=167  I bought this for my grand daughter when she comes over to visit. I set it up with her as the user, entered her age and name and now Amazon makes sure that she only accesses sites and content that are appropriate to her age. Simple to do and she loves the capabilities. I also bought and installed a 64gig SD card which gives this little tablet plenty of storage. For the price I think this tablet is best one out there. You can spend hundreds of dollars more for additional speed and capacity but when it comes to the basics this tablets does everything that most people will ever need at a fraction of the cost.                Fantastic Tablet for kids               NaN                   NaN        explore42 
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 34660 entries, 0 to 34659
Data columns (total 21 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   id                    34660 non-null  object 
 1   name                  27900 non-null  object 
 2   asins                 34658 non-null  object 
 3   brand                 34660 non-null  object 
 4   categories            34660 non-null  object 
 5   keys                  34660 non-null  object 
 6   manufacturer          34660 non-null  object 
 7   reviews.date          34621 non-null  object 
 8   reviews.dateAdded     24039 non-null  object 
 9   reviews.dateSeen      34660 non-null  object 
 10  reviews.didPurchase   1 non-null      object 
 11  reviews.doRecommend   34066 non-null  object 
 12  reviews.id            1 non-null      float64
 13  reviews.numHelpful    34131 non-null  float64
 14  reviews.rating        34627 non-null  float64
 15  reviews.sourceURLs    34660 non-null  object 
 16  reviews.text          34659 non-null  object 
 17  reviews.title         34655 non-null  object 
 18  reviews.userCity      0 non-null      float64
 19  reviews.userProvince  0 non-null      float64
 20  reviews.username      34658 non-null  object 
dtypes: float64(5), object(16)
memory usage: 5.6+ MB
```

```
        reviews.id  reviews.numHelpful  ...  reviews.userCity  reviews.userProvince
count          1.0        34131.000000  ...               0.0                   0.0
mean   111372787.0            0.630248  ...               NaN                   NaN
std            NaN           13.215775  ...               NaN                   NaN
min    111372787.0            0.000000  ...               NaN                   NaN
25%    111372787.0            0.000000  ...               NaN                   NaN
50%    111372787.0            0.000000  ...               NaN                   NaN
75%    111372787.0            0.000000  ...               NaN                   NaN
max    111372787.0          814.000000  ...               NaN                   NaN
```

![numb_prod_per_id](numb_prod_per_id.png)

This figure shows that some products have many reviews while others have very little reviews. This can be a problem because the more different reviews (and therefore the more words) we have, the better we can train the model. Here, most of the reviews refer to the same product, which can limit the range of emotions and words. This will be the first difficulty in our dataset. We need to get an overall picture of the distribution of the ratings to see if there are other problems with our dataset.

Ratings:
> We notice that over the 36640 data points, only 34627 have a rating value. Thus 2013 data points won’t be useful in our analysis. We can drop them from the dataset.

Overall idea:
> In order to have a brief overview of the dataset, we plot the distribution of the ratings. We have 5 classes (ratings from 1 to 5). We notice that the data we have is not well distributed, the classes are not represented equally: the majority of the products that were rated, were rated highly. There is more than twice as many 5-star ratings as all other ratings combined and about 70% of the dataset belongs only to 1 class (5-star ratings). This is an imbalanced dataset.

![avg_rating](avg_rating.png)

Most classification datasets do not have exactly equal number of instances in each class, but a small difference often does not matter. However, in our case we have a significant class imbalance and it can cause problems. This imbalance is expected since the dataset characterize the overall appreciation of Amazon manufactured products. The vast majority of the products will be highly rated otherwise they will be dropped of the stock. Here are a few examples of ratings: 
```
Average rating per product:  id
AV1YE_muvKc47QAVgpwE    4.707278
AV1YnR7wglJLPUi8IJmi    4.424731
AV1YnRtnglJLPUi8IJmV    4.772355
AVpe7AsMilAPnD_xQ78G    4.666667
AVpe8PEVilAPnD_xRYIi         NaN
AVpe9CMS1cnluZ0-aoC5    4.000000
AVpfBEWcilAPnD_xTGb7         NaN
AVpfIfGA1cnluZ0-emyp    4.205479
(...)
```
Among the data set is also the number of helpful votes for a given review. The distribution of useful reviews is as follow:
![numb_useful_reviews](numb_useful_reviews.png)

At first it seems that attribute `reviews.numHelp` has no value since no user has considered any of the comments as useful but looking more closely:

`Are all values in column reviews.numHelp equal to 0:  reviews.numHelpful    False`

It is noticeable that very few comments were designated as "helpful" by other consumers and this is why outliers are valuable. We may want to weight reviews that had many people who find them helpful.



## Stratified sampling

We need to split our dataset into a training set and a test set. Since the absolute majority of reviews are positive reviews, we will need to do a stratified distribution of the scores to **ensure that we do not train the classifier on imbalanced data**. We will first remove all samples whose review score is NAN, then convert all reviews scores to integer datatype and finally stratify our sampling.
```
Original dataset 34660
New dataset 34627
len(stratified_training) = 27701  | len(stratified_test) = 6926
```
We can measure and compare the proportions of reviews ratings in the overall dataset with the proportions in the train and test sets generated with stratified sampling to ensure the population is divided into homogeneous subgroups (strata):
```
           Df  Strat Train Set  Strat Test Set
1.0  0.011840         0.011949        0.009962
2.0  0.011609         0.011516        0.011839
3.0  0.043290         0.042923        0.044470
4.0  0.246657         0.248005        0.240832
5.0  0.686603         0.684668        0.691886
```



## Sentiment analysis

We need to segregate ratings from 1 to 5 into positive, neutral, and negative sentiments and then create a sentiment category attribute.


### Feature engineering

To select the useful features for the model, I will be using the bag of words strategy. It is a way of extracting features from text by describing the occurrence of words within the text. For that, I will be using the scikitlearn library: CountVectorizer() and TfidfTransformer().
CountVectorizer() enables to convert a collection of text documents to a matrix of token counts whereas TfidfTransformer() transforms a count matrix to a normalized tf or tf-idf representation.


## Tokenization
The *bag-of-words* representation technique summarizes a document in terms of word counts and throws away all irelevant information or information relevant to any more complex processes of document construction. The reason for eliminating all these data is computational and statistical efficiency, we need to store and work on data that are important. The huge advantage of this technique is its simplicity.
Usually ... :
removes the too common or rare words, isolated punctuation and numbers as well as common suffixes like -s or -ing.
The remaining tokens are then counted


### Building models

We will train and evaluate several models able to recognize the overall feeling of the consumer and then fine tune the most promising ones.
•	Logistic Regression
•	Naïve Bayes Analysis
•	Support Vector Machines
•	Decision Trees
•	Random Forests

![models_rank](models_rank.png)

```
Some prediction examples: 
                                  Reviews       Multi NB        Log Reg     Linear SVC  Decision Tree  Random Forest   GSCV Lin SVC
0  I love this product. I am very happy!  Very Positive  Very Positive  Very Positive  Very Positive  Very Positive  Very Positive
1                    I like the product.  Very Positive  Very Positive  Very Positive  Very Positive  Very Positive  Very Positive
2           I like it but it’s too large  Very Positive        Neutral        Neutral  Very Positive  Very Positive  Very Positive
3                 I don’t really like it  Very Positive        Neutral        Neutral  Very Positive  Very Positive        Neutral
4                         A bit confused  Very Positive        Neutral        Neutral                                      Neutral
5                                    Bad        Neutral        Neutral        Neutral                                      Neutral
6                                  worst        Neutral        Neutral  Very Negative                                Very Negative
```







# Team Hopf Bundle: Foursquare Location Matching

This problems purpose is to deduplicate raw location data. 

This project is part of The Erdos Institute's [data science boot camp](https://www.erdosinstitute.org/code) 
and is a proposed model for a [Kaggle competition sponsored by Foursquare](https://www.kaggle.com/competitions/foursquare-location-matching).
Raw data contains over half a million locations of commercial points of interest around the world. 
This raw data may also contain incorrect or artificial entries as well as additional noise. 

## Table of contents
- [Exploratory Data Analysis](#EDA)
- [Process](#process)
- [Models](#models)
- [Future Improvements](#futureimprovements)
- [Slides](#slides)

## Exploratory Data Analysis

All data points have the features: id, name, latitude, longitude, country, and point of interest (POI).
However a data point can also have features: address, city, state, zip, url, phone, and categories. 
A lot of data points are sparse and missing these additional features, instead they contain a NaN entry. 
The data supplied by [Kaggle](https://www.kaggle.com/competitions/foursquare-location-matching)
also contained a pairs dataset where each entry is data points from our train set and the additional feature ‘match’ which is boolean and specifies whether two points are a match, ie. they describe the same POI.

Initally, we assumed location would be an important category. We were only given latitude and longitude coordinates so we used these to compute a simple Euclidean distance between two points.
We added this to our pairs data set as the 'distance' feature.
We found that there are many data points which are within short distances from one another that are not matches. 
Conversely, a couple of hundred pairs were also given to us as matches, while being nowhere near each other in terms of location.

We first began to train models on the features: categorgies, name, and distance. Both 'name' and 'distance' are features which are guaranteed with every observation. The feature 'categories' is an optional feature with the lowest proportion of missing data. 

## Process

1. Before training our models we had to vectorize our data to train on all of our feature entries given by strings. 
These features include 'name', 'address', 'categories', 'city', 'state', 'url', and 'country'. 
We used the [Tfidf vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
in sk-learn to vectorize our training set.

2. We also attempted to clean our data by removing matches which were above the 99th quartile for geodesic distance. 
However, this had no effect on any of our models accuracy and so we decided to continue training and testing models with unclean data.

3. With our vectorized data, we computed similarities between the features of the observations in pairs. That is, we used cosine-similarity to calculate a similarity score between the names, categories, and any other features we wanted to use. For example, the strings "1 Towne Centre Blvd #2800" and "1 Towne Centre Blvd" might have a similarity index of 0.8036. 

4. Sometimes we were unable to compute similarities scores due to missing data. If one observation had a recorded address "1 Towne Centre Blvd #2800" and another had a "NaN" address, we had to choose how to impute this similarity. We initially went with a method of zero-imputation, designating any similarity to "NaN" values as 0. Once we found our most accurate model we also trained and tested with mean inputed data from those similarity scores which could be computed in the previous step. 

## Models

Our first models we trained were our baseline models trained only on the features: name, distance and categories.

|Model                  |Accuracy|
---                     |---
|Logistic Regression (distance only)     | 0.6889|
|Logistic Regression | 0.7205|
|K Nearest Neighbors | 0.7269|
|Feed-Forward Neural Network | 0.7259|
|Random Forrest | 0.7285 |

The random forrest model gave us the best results for accuracy we looked into training an XGBoost model over all features:

|Model      |Accuracy|
---         |---
|XGBoost (Zero Inputation)  |0.7642 |
|XGBoost (Mean Inputation)  |0.7842 |

This model yields the highest accuracy and we even see a 2 point increase in accuracy when we use mean inputation on missing values from the data.


## Future Improvements

1. We realized a little too late that using Tfidf vectorizer to train our models is not a pretrained process. 
This means our model does not handle new data entries well, and so it does not evaluate well on the test set supplied by the Kaggle submission. 
We hope to revectorize our data using BERT, a pretrained natural language processor. This should overall improve our model significantly. 

2. Data cleaning is a difficult problem to solve with this data set. 
We tested cleaning based on distance, but we also want to look into cleaning our data through other features as well. 
We believe that erroneous data possibly occurs in the 'categories' and 'address' features.

3. Our model had a precision of 0.8988. We want to look into false positives and false negatives in our model so that we can fine tune
our models matching capabilities. We are also hoping that BERT will help improve our precision.

# Slides
Additional information may be found within our project slides which can be found above. 




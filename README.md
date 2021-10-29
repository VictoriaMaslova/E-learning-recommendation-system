# E-learning recommendation system

In this era when every aspect of society is accelerating, people are always seeking improvement to stay competitive in their careers. E-learning systems fit into the ever challenging situation and provide learners with remote learning opportunities and abundant learning resources. Facing with the numerous resources online, users need support in deciding which course to take, thus recommender systems are applied in E-learning to provide learners with personalized services by automatically identifying their preferences. 

There are a lot of online educational platfroms, I picked Udemy to build recommendation system to boost students satisfaction in learning process.

Udemy is an online learning platform with more than 100.000 courses and over 30 million students all over the world. The platform offers courses in different categories e.g. Business, Design or Marketing. With all the available options it is very hard to choose the proper course, since everyone has a different taste. A recommender system helps students choose the next course, without spending hours reading different course descriptions. It does not only spare time for the user, but helps to find something interesting based on their previous course choices.

My project is built up the following way: At first, I clusterize the Business courses based on their description by means of NLP (natural language processing) techniques. Secondly, based on the new clusters and other relevant features (e.g. price or length of course) I build a recommender system.

### Structure

The project consists of 4 main parts:

```Import_data.py```: Getting the data through Udemy API

```Data_cleaning.py```: Cleaning the data

```EDA.ipynb```: Exploratory Data Analysis

```Clustering_and_RS.ipynb```: Clustering and recommender system

Other files of the project:

```Recommender_system.py```: The relevant commands to run the recommender system

```kmeans8.sav``` : the final clustering algorithm exported

```Udemy_funcitons.py```: all the functions which are needed for the projects

# Approach

### Step 1: Organizing the Data

Udemy has an API, where all the course informations and user ratings can be achieved. I downloaded 10.000 Business courses through the API and loaded them in a Dataframe. (The limit is 10.000 records per query). For all of these courses I also downloaded the available user ratings.

### Step 2: Cleaning the data

I followed these steps while cleaning the data:

* import the raw data
* transform the relevant columns
* filter the dataset
* keep the relevant columns
* drop duplicates
* treat the missing values
* save the cleaned data

The dataset was filtered so that it only contains live and public courses. After removing the duplicates and missing values, the cleaned dataset for courses (df_courses) consisted of 8.834 courses, which were analysed during the exploratory data analysis (EDA) part. The non-relevant information, which cannot be used for modelling, was dropped from the database.

While cleaning the reviews data I realized, that unfortunately the user name of the review is not unique: it resulted of hundreds of Davids or Michaels. Due to this challenge, instead of using collaborative filtering I preferred to build a content-based recommender system which uses the course features and the new clusters of the courses. I utilize the reviews data to find the courses the users took, and recommend them other courses similar to their previous ones.


### Step 3: EDA

For the numeric features, the distribution and the boxplot was used. For the categorical features a barchart was plotted. The following graphs show an example of a numeric and a categorical feature:

Summary of the most important findings:

1. There are around 900 courses with no reviews/ratings, but most of them are between rating 4 and 4.5
2. The price ranges between 0 and 199 EUR
3. There are some really popular courses with a lot of subscribers. The top 3 are:

- machinelearning   with more than 300T subscribers
- python-for-data-science-and-machine-learning-bootcamp with 192T subscribers
- an-entire-mba-in-1-courseaward-winning-business-school-prof with 187T subscribers


4. The average age of a course is 26 months (since it was published). There are more recently published courses than older ones.
5. The majority of the courses is for all levels. Only a few courses requires an advanced level.
6. The courses are divided into 16 subcategories, whereas the two most significant are Finance and Entrepreneurship.
7. Two subcategories have an average price higher than 100 dollars : The subcategory Data & analytics with 112, and Project Management with 104
8. These two attributes were analyzed deeper: “objectives” and “description”, since one of these features will be the basis of the clustering part in Step 4. The texts were tokenized and stemmed using the Natural Language Toolkit (nltk library) in Python. After removing the stopwords and punctuations, the following plots show the top 25 most popular words in the subcategories ‘Data & Analytics’.

![](https://github.com/VictoriaMaslova/E-learning-recommendation-system/blob/main/images/wordcloud_data.png)


After the univariate analysis I also executed multivariate analysis:

- There is a positive correlation between the number of reviews/number of subscribers and the average rating - students normally give good ratings for courses they liked
- There is also a positive correlation between published since and the average rating -> older courses have better ratings. This seems logic, since I would expect that courses which aren't popular won't stay long

Most important findings on the reviews dataset:
- The users are unfortunately not unique. Because if this reason, it is not possible to build a recommender system on the user ratings.


### Step 4: Clustering:

I tried to analyse both the objectives and the descriptions of the courses to find new clusters. Finally the clustering algorithm based on the course descriptions showed better results.

The texts were first tokenized and stemmed, the stopwords and punctuations were removed. Afterwards the TfidfVectorizer was applied on the data, which helps to identify words which are frequent in the text but rare in the corpus. Based on this frequency-matrix I applied the k-means and hierarchical clustering algorithms.

For the k-means algorithm I tried out multiple k-s (number of clusters). I checked how the inertia (within-cluster sum-of-squares) changes to look for an optimal number of clusters. According to the elbow method, the line is an arm and the "elbow" on the arm is the value of k that is the best. Finally I implemented Hierarchical cluster analysis and got k=8 to be the optimal number of clusters.

![](https://github.com/VictoriaMaslova/E-learning-recommendation-system/blob/main/images/dendrorgamm.png)

After fitting a k-means algorithm with k=8 clusters, the following graph shows the distribution of the clusters with their most common words in the description. The most courses can be found in cluster 0. The top 5 words in each cluster are in the label to help to identify what kind of courses belong the the clusters.

To plot the clustering, I used a dimension reduction technique called PCA (Principal Component Analysis) and kept the first 2 principle components. The two components (from the total 1000 components) explain more than 4% of the total information. I plotted the clusters on a sample data to check if the clusters can be well distinguished in this reduced dimension. As the next graph shows, almost all of the clusters are distributed nicely in 2-D. 

![](https://github.com/VictoriaMaslova/E-learning-recommendation-system/blob/main/images/bar_kmeans8_words.png)

To plot the clustering, I used a dimension reduction technique called PCA (Principal Component Analysis) and kept the first 2 principle components. The two components (from the total 1000 components) explain more than 4% of the total information. I plotted the clusters on a sample data to check if the clusters can be well distinguished in this reduced dimension. As the next graph shows, almost all of the clusters are distributed nicely in 2-D.

![](https://github.com/VictoriaMaslova/E-learning-recommendation-system/blob/main/images/pca.png)


### Step 5: Building a recommender system:

There are two main types of recommender systems: collaborative filtering and content/item based recommender systems.

* Collaborative filtering : uses the similarities in users’ behaviors and preferences to predict what users will like.
* Content-based filtering: use the description of the item and a profile of the user’s preferences to recommend new items.

As I already mentioned, the reviews data (with the users’ behaviors) was not appropriate to build a recommender system with collaborative filtering. I used the clusters of step 4 and other features of the courses to build a content-based recommender system.

Before building the recommender system, I transformed the feature matrix:

Introduced dummy variables: I transformed the categorical features to dummies, that we can use these features as well in the recommender system. I also transformed the new clusters, since the cluster number doesn’t have any meaning (e.g. there is no relationship between cluster 1 and cluster 2)
Scaled the features: Since the features of the courses have different magnitude, it is important to scale the feature matrix: the price of the courses varies between 0 and 199, but the average rating has a range between 0 and 5. I used the StandardScaler from the scikit-learn library, which standardizes all features by removing their mean and scaling them to unit variance.

I also defined a similarity measure to compare the courses: I used the cosine similarity, which calculates the cosine of the angle between two vectors projected in a multidimensional space. In this context, the two vectors I am talking about are the arrays containing the transformed features of the courses.

There are 2 functions, which can be used to recommend courses:

```Function recommend_for_user``` recommends courses for the user based on his/her previous courses.

```Function recommend_courses ``` recommends courses based on another course_id. This function takes the course_id instead of the user_name as input and looks for the courses that are similar to the original course.


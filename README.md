<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 120px">

# Evocative of One Another: Dungeons & Dragons and Pathfinder RPG

*Deval Mehta*

## Table of Contents
1) [Overview](#Overview) 
2) [Data Dictionary](<#Data Dictionary>)
3) [Requirements](#Requirements)
4) [Executive Summary](<#Executive Summary>)
    1) [Purpose](<#Purpose>)
    2) [Data handling](<#Data Handling>)
    3) [Analysis](#Analysis)
    4) [Findings and Implications](<#Findings and Implications>)
    5) [Next Steps](#Next-Steps)

## Overview
Tabletop Role Playing Games (TTRPGs) experienced a massive popularity boom in the latter half of the 2010s due to the advent of live-streaming in-person games and online gaming communities, which has continued into the present day. Two giants stand atop the high-fantasy TTRPG genre: *Dungeons & Dragons*, maintained by Wizards of the Coast, a division of Hasbro and *Pathfinder*, maintained by Paizo. The two systems are strikingly similar -- so much so that upon its inception in 2009, Pathfinder was dubbed by fans as "D&D 3.75." Among other things, the two systems share relatively similar settings, rules, and character classes. With so much in common, one might wonder from the perspective of a player or game master: are the two truly distinct enough to be considered separate systems? From a more corporate lens, one might frame the question: are Dungeons & Dragons and Pathfinder sufficiently similar that Hasbro should explore the possibility of suing Paizo for infringement?

Our goal is to consider this question through the lens of natural language processing by analyzing posts on the subreddits r/DungeonsAndDragons and r/Pathfinder_RPG. We've retrieved nearly 2,000 posts from each subreddit using the Python Reddit API Wrapper (PRAW) to train and test our models. We generate an incredible number of features from our data by considering 1,2, and 3-grams, explore some factors that might distinguish posts on one subreddit from those of another, and finally compare the logistic regression, random forest, and support vector machine approaches to the problem.

Our selection of models have varied performance on the testing data, after optimization. Two models perform above our baseline set by our initial logistic regression model of 80% accuracy in classification, but all of our models present as overfit to the data. Under such circumstances, the data proves inconclusive and further exploration is required. Further analysis would reduce the complexity of the strongest models by considering fewer features and continuing to tune regularization strength.

## Data Dictionary
Rather than enumerating the numerous tokens of each document in our corpus, we summarize the information collected from each Reddit post.

### Collected Features
| Information | Data Type | Description | Notes |
|---|---|---|---|
| id | `string` | The individual ID number assigned to each post by Reddit | We use this to remove any possible duplicates. |
| created_utc | `float64` | The number of seconds that elapsed between 1970-Jan-01 00:00 UTC and the creation of the post. | Provided an option for de-duplicating, in case ID numbers were corrupted. |
| title | `string` | The title of each Reddit post | The most complete text data from each subreddit. |
| self_text | `string` | The body text of each Reddit post | The intended corpus. With so many missing values, we have pivoted a concatenation of `title` and `self_text`.|
| subreddit | `string` | The subreddit from which each post was retrieved | Our response variable. |

### Engineered Features
To conduct our analysis, we concatenate the `title` and `self_text` to create a new `all_text` column, which we tokenized. In addition, we created a binarized version of `subreddit` called `isDnD`, which is an integer representation of whether a post belongs to r/DungeonsAndDragons. This is done to ensure our `LogisticRegression` object will be able to interpret the two categories.

## Requirements
### Hardware
Many of the procedures in the project are parallelized on 12 threads. As such, we recommend that a prospective colleague or student seeking to replicate this work either operate on a machine or server with a CPU that has **at least** 6 cores and 12 threads or modifies the `n_jobs` arguments to a lower number. The latter option will increase the computation time significantly.

### Software
We employ the following libraries, modules, and functions to run the project.
| Library | Module | Purpose |
|---|---|---|
| `numpy` | | Ease of basic aggregate operations on data. |
| `pandas`| | Read our data into a DataFrame, clean it, engineer new features, and write it out to submission files. |
| `matplotlib` | `pyplot` | Basic plotting functionality. |
| `seaborn` | | More control over plots. |
| `nltk` | `pos_tag` | Retrieve and access "part of speech" tags. |
| | `tokenize` | `RegexpTokenizer` to tokenize data for EDA purposes. |
| | `stem` | `WordNetLemmatizer` to build a custom lemmatizer for TFIDF Vectorization. |
| | `corpus` | `wordnet`, `stopwords`  to contribute to our custom lemmatizer. |
| | `sentiment.vader` | `SentimentIntensityAnalyzer` to conduct sentiment analysis. |
| `sklearn` | `ensemble` | `RandomForestClassifier` for random forest classification. |
| | `linear_model` | `LogisticRegression` for logistic classification. |
| | `model_selection` | `train_test_split` to split our data into training and testing sets and `GridSearchCV`to optimize models via hyperparameter tuning. |
| | `neighbors` | `KNeighborsClassifier` for k-nearest neighbors classification. |
| | `svm` | `SVC` for support vector classification. |
| | `metrics` | `classification_report` to generate a classification report for each model with salient metrics and `ConfusionMatrixDisplay` to help us visualize the confusion matrix |
| | `preprocessing` | `FunctionTransformer`, `StandardScaler` |
| | `pipeline` | `Pipeline` |
| | `compose` | `ColumnTransformer` to consolidate the transformations for text data.|
| | `feature_extraction.text` | `CountVectorizer`, `TfidfVectorizer` to tokenize, lemmatize, and vectorize text data. |
| `praw` | | The Python Reddit API Wrapper. Imported as a fail-safe, in case data collection goes wrong. |
| `reddit_scraper` | | Collect the $n$ top (within a timeframe) and $k$ newest posts from a subreddit and write out their id, time of creation, title, text, and subreddit to a `.csv` file. |

The file `reddit_scraper.py` requires an existing `.env` file in the working directory of the project from which to retrieve the API Key and password. For security, this has been omitted from the repository, so a prospective user will have to initialize their own `.env` file with their own API key and password for Reddit.

## Executive Summary
### Purpose
To determine whether we can make a firm recommendation on exploring an intellectual properties suit, we consider the text data from nearly 2,000 posts on each of r/DungeonsAndDragons and r/Pathfinder_RPG. We employ Natural Language Processing (NLP) techniques to dissect the text data and format it into something our classifiers will accept for analysis.

In total we consider four classes of classification models:
* Logistic Regression
* K-Nearest Neighbors
* Random Forest
* Support Vector Machines

### Data Handling
Using the Python Reddit API Wrapper (PRAW), we collect a total of 3,964 posts split nearly evenly between r/DungeonsAndDragons and r/Pathfinder_RPG. Many of the posts lack body text, so we create a new feature which contains all of the title and body text for analysis. This notably has to do with the low verbosity of r/DungeonsAndDragons, where users are more likely to post images and files. r/Pathfinder does have its fair share of posts lack body text, but there are far fewer.

To prepare the data for analysis and the modeling, we:
* De-duplicate the data within the script `reddit_scraper.py` before writing it out to a `.csv` file
* Engineer a new `all_text` column which combined the title and body text to effectively deal with null values without losing information or upsetting the balance of our response variable.
* Consolidate all the data into a single DataFrame
* Engineer another new feature `isDnD` to binarize the `subreddit` feature
* Create a copy of a subset of the DataFrame that is tokenized, standardized, and stripped of all punctuation
* Create another copy of a subset of the DataFrame that is vectorized without frequency weighting

### Analysis
Prior to fitting models to the data, we explore the data to see if there is any apparent difference between the posts. We consider the distribution of the posts by log(word count), the general tone of posts via VADER sentiment analysis, and the 20 most frequent (1,3)-grams on the two subreddits. We establish through exploration that there ought to be a sufficient difference between the two communities and TTRPG systems and leave it to our models to confirm. From this analysis, we opt to include the compound sentiment score among our features to which we fit our models.

### Findings and Implications
Of the four classes of models we tested, two outperformed the baseline (if only just) and two were far outclassed by the baseline. All of our models were fed the same preprocessed data, the pipeline for which was as follows:
1) Employ TFIDF Vectorization to lemmatize and tokenize, remove all (English) stop words, and consider only those words which appear in at most 0.25% of our corpus.
2) Convert the output from a sparse matrix to a dense matrix.
3) Scale the data so that our compound sentiment score is not washed out.

Our baseline logistic regression was able to correctly sort 80% of the testing data. After optimizing the models, the new logistic regression and the random forest classifiers outperformed the baseline slightly, yielding accuracies of 83% each. The k-nearest neighbors and support vector machine classifiers did not bode nearly as well, with each performing below 55% accuracy. Since our models were not able agree on whether the two subreddits are sufficiently distinct, we must concede that the investigation is inconclusive and further analysis is required.

### Next Steps
The models were all overfit, despite our attempt to restrict the sheer number of features. Further analysis would reduce the complexity of the models by further restricting the number of features, increasing the strength of regularization, or fitting less complex models. We could also reduce the scope of our corpus to more verbose documents (those containing body text).

### Links to Non-Course Sources Consulted
[GeeksforGeeks' Comprehensive Guide to Classification Models in Sci-Kit Learn](https://www.geeksforgeeks.org/comprehensive-guide-to-classification-models-in-scikit-learn/)

[The RexEgg RegEx Cheat Sheet](https://www.rexegg.com/regex-quickstart.php)

[Reddit user nullus_72 provides a comprehensive answer about the differences between D&D and Pathfinder for the uninitiated](https://www.reddit.com/r/DnD/comments/11i9vjg/comment/jax4p5x/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)

[Conversation with ChatGPT 4o mini about building a transformer that chains TFIDFVectorizer() into StandardScaler()](https://chatgpt.com/share/674401c4-2ad0-800c-b95f-498e594383f9)
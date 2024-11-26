<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 120px">

# Evocative of One Another: Dungeons & Dragons and Pathfinder RPG

*Deval Mehta*

## Table of Contents
1) [Overview](#Overview) 
2) [Data Dictionary](<#Data Dictionary>)
3) [Requirements](#Requirements)
4) [Executive Summary](#Executive-Summary)
    1) [The Data](<#The Deta>)
    2) [Baseline Values](<#Baseline Values>)
    3) [Data Transformation](<#Data Transformation>)
    4) [Model Selection Criteria](<#Model Selection Criteria>)
    5) [Analysis](#Analysis)
    6) [Implications](#Implications)
    7) [Next Steps](#Next-Steps)

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

## Executive Summary
### The Data

### Baseline Values

### Data Transformation

### Model Selection Criteria

### Analysis

### Implications

### Next Steps

### Links to Non-Course Sources Consulted
[GeeksforGeeks' Comprehensive Guide to Classification Models in Sci-Kit Learn](https://www.geeksforgeeks.org/comprehensive-guide-to-classification-models-in-scikit-learn/)

[The RexEgg RegEx Cheat Sheet](https://www.rexegg.com/regex-quickstart.php)

[Reddit user nullus_72 provides a comprehensive answer about the differences between D&D and Pathfinder for the uninitiated](https://www.reddit.com/r/DnD/comments/11i9vjg/comment/jax4p5x/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)

[Conversation with ChatGPT 4o mini about building a transformer that chains TFIDFVectorizer() into StandardScaler()](https://chatgpt.com/share/674401c4-2ad0-800c-b95f-498e594383f9)

---

## Rubric
Your instructors will evaluate your project (for the most part) using the following criteria.  You should make sure that you consider and/or follow most if not all of the considerations/recommendations outlined below **while** working through your project.

For Project 3 the evaluation categories are as follows:<br>
**The Data Science Process**
- Problem Statement
- Data Collection
- Data Cleaning & EDA
- Preprocessing & Modeling
- Evaluation and Conceptual Understanding
- Conclusion and Recommendations

**Organization and Professionalism**
- Organization
- Visualizations
- Python Syntax and Control Flow
- Presentation

**Scores will be out of 30 points based on the 10 categories in the rubric.** <br>
*3 points per section*<br>

| Score | Interpretation |
| --- | --- |
| **0** | *Project fails to meet the minimum requirements for this item.* |
| **1** | *Project meets the minimum requirements for this item, but falls significantly short of portfolio-ready expectations.* |
| **2** | *Project exceeds the minimum requirements for this item, but falls short of portfolio-ready expectations.* |
| **3** | *Project meets or exceeds portfolio-ready expectations; demonstrates a thorough understanding of every outlined consideration.* |


### The Data Science Process

**Problem Statement**
- Is it clear what the goal of the project is?
- What type of model will be developed?
- How will success be evaluated?
- Is the scope of the project appropriate?
- Is it clear who cares about this or why this is important to investigate?
- Does the student consider the audience and the primary and secondary stakeholders?

**Data Collection**
- Was enough data gathered to generate a significant result? (At least 1000 posts per subreddit)
- Was data collected that was useful and relevant to the project?
- Was data collection and storage optimized through custom functions, pipelines, and/or automation?
- Was thought given to the server receiving the requests such as considering number of requests per second?

**Data Cleaning and EDA**
- Are missing values imputed/handled appropriately?
- Are distributions examined and described?
- Are outliers identified and addressed?
- Are appropriate summary statistics provided?
- Are steps taken during data cleaning and EDA framed appropriately?
- Does the student address whether or not they are likely to be able to answer their problem statement with the provided data given what they've discovered during EDA?

**Preprocessing and Modeling**
- Is text data successfully converted to a matrix representation?
- Are methods such as stop words, stemming, and lemmatization explored?
- Does the student properly split and/or sample the data for validation/training purposes?
- Does the student test and evaluate a variety of models to identify a production algorithm (**AT MINIMUM:** two models)?
- Does the student defend their choice of production model relevant to the data at hand and the problem?
- Does the student explain how the model works and evaluate its performance successes/downfalls?

**Evaluation and Conceptual Understanding**
- Does the student accurately identify and explain the baseline score?
- Does the student select and use metrics relevant to the problem objective?
- Does the student interpret the results of their model for purposes of inference?
- Is domain knowledge demonstrated when interpreting results?
- Does the student provide appropriate interpretation with regards to descriptive and inferential statistics?

**Conclusion and Recommendations**
- Does the student provide appropriate context to connect individual steps back to the overall project?
- Is it clear how the final recommendations were reached?
- Are the conclusions/recommendations clearly stated?
- Does the conclusion answer the original problem statement?
- Does the student address how findings of this research can be applied for the benefit of stakeholders?
- Are future steps to move the project forward identified?


### Organization and Professionalism

**Project Organization**
- Are modules imported correctly (using appropriate aliases)?
- Are data imported/saved using relative paths?
- Does the README provide a good executive summary of the project?
- Is markdown formatting used appropriately to structure notebooks?
- Are there an appropriate amount of comments to support the code?
- Are files & directories organized correctly?
- Are there unnecessary files included?
- Do files and directories have well-structured, appropriate, consistent names?

**Visualizations**
- Are sufficient visualizations provided?
- Do plots accurately demonstrate valid relationships?
- Are plots labeled properly?
- Are plots interpreted appropriately?
- Are plots formatted and scaled appropriately for inclusion in a notebook-based technical report?

**Python Syntax and Control Flow**
- Is care taken to write human readable code?
- Is the code syntactically correct (no runtime errors)?
- Does the code generate desired results (logically correct)?
- Does the code follows general best practices and style guidelines?
- Are Pandas functions used appropriately?
- Are `sklearn` and `NLTK` methods used appropriately?

**Presentation**
- Is the problem statement clearly presented?
- Does a strong narrative run through the presentation building toward a final conclusion?
- Are the conclusions/recommendations clearly stated?
- Is the level of technicality appropriate for the intended audience?
- Is the student substantially over or under time?
- Does the student appropriately pace their presentation?
- Does the student deliver their message with clarity and volume?
- Are appropriate visualizations generated for the intended audience?
- Are visualizations necessary and useful for supporting conclusions/explaining findings?
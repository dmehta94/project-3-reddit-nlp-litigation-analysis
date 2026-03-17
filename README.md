# Evocative of One Another: D&D and Pathfinder NLP Classification

*Deval Mehta · General Assembly Data Science Bootcamp, Project 3 · November 2024*

**Stack:** Python · PRAW · NLTK · scikit-learn · pandas · seaborn  
**GitHub:** [@dmehta94](https://github.com/dmehta94)

---

## What It Does

Pathfinder RPG launched in 2009 as a direct descendant of D&D 3.5e — so
similar that fans nicknamed it "D&D 3.75." The two systems share settings,
rules architecture, and character class designs to a striking degree. That
raises a question: are they *legally* distinct enough that Hasbro (which owns
D&D publisher Wizards of the Coast) could explore an intellectual property
suit against Pathfinder publisher Paizo?

I built an NLP classification pipeline to approach that question through
community language. If the two systems are genuinely distinct, their Reddit
communities should talk about them differently — and a classifier should be
able to tell the posts apart. Using PRAW to collect nearly 4,000 posts from
r/DungeonsAndDragons and r/Pathfinder_RPG, I trained and compared four
classifiers — Logistic Regression, K-Nearest Neighbors, Random Forest, and
Support Vector Machine — on TF-IDF vectorized text augmented with VADER
compound sentiment scores. The best model (Random Forest) reached 83.9% test
accuracy, but all models overfit significantly, leaving the central question
genuinely inconclusive.

---

## Why I Built This

This was my third project at General Assembly. The prompt asked us to scrape
two related subreddits and train a binary classifier on the text data — the
specific pairing and framing were up to us. The IP litigation angle appealed to
me immediately: it transforms an inconclusive classification result from a
disappointment into a meaningful finding. If models *can't* reliably separate
the communities, that's a result worth reporting, not a failure to explain away.

I chose this pairing specifically because I could anticipate a real data
challenge before writing a single line of code. r/DungeonsAndDragons skews
heavily toward image posts — memes, character art, session photos — which meant
most posts would have no body text at all. My decision to concatenate title and
body text into a single `all_text` feature was a direct response to that
constraint, made before collecting the data.

**The question I was answering:** Should Hasbro explore an IP suit against
Paizo — and can NLP-based community analysis shed any light on it?

**Result:** The two communities are distinguishable, but not cleanly enough to
make a firm recommendation either way. Every model overfit significantly, and
the best test accuracy before hitting a complexity ceiling was 83.9%. The
models agree that *some* signal exists, but they don't agree on what it is —
which is exactly the kind of inconclusive finding that warrants further analysis
before drawing legal conclusions.

---

## What I Learned

### Technical skills

- **PRAW API** — Learned how to authenticate, rate-limit, and paginate Reddit
  data collection. PRAW doesn't always return the full `limit` you request
  because deleted posts and moderation gaps reduce the available pool.
- **TF-IDF vectorization** — The `min_df` parameter is critical in high-dimensional
  text problems. Setting it too low (say, 0.001) explodes the feature space and
  makes training prohibitively slow; too high and you lose signal from
  domain-specific vocabulary. I settled on `min_df=0.0025` after testing.
- **Custom POS-aware lemmatization** — Standard lemmatizers default to treating
  everything as a noun. I built a custom lemmatizer using NLTK's `pos_tag` to
  pass part-of-speech context to `WordNetLemmatizer`, so "casting" becomes
  "cast" rather than "casting."
- **ColumnTransformer + Pipeline** — Combining TF-IDF vectorization with a
  numeric sentiment feature required `ColumnTransformer` to handle mixed input
  types, then `Pipeline` to chain the output through `StandardScaler`. This was
  the hardest architectural problem in the project, and I consulted a ChatGPT
  conversation specifically to understand how to chain a sparse TF-IDF output
  into `StandardScaler` (linked in credits).
- **GridSearchCV** — Ran parallel hyperparameter searches across regularization
  strength (Logistic Regression), number of neighbors (KNN), tree depth and
  split parameters (Random Forest), and kernel regularization (SVC).

### Data science insights

- **Pathfinder posts are longer and more polarized than D&D posts.** The word
  count distribution confirmed that r/Pathfinder_RPG is more verbose — a
  reflection of Pathfinder's reputation as a crunchier, more rules-intensive
  system. VADER sentiment analysis showed Pathfinder posts cluster more strongly
  at the positive and negative extremes, while D&D posts cluster near neutral.
  Both of these patterns boded well for classification.
- **KNN fails badly in high-dimensional text space.** The best KNN model reached
  only 54.3% test accuracy — barely better than a coin flip. This is a known
  phenomenon: in high-dimensional space, the notion of "nearest neighbor"
  breaks down because all points are approximately equidistant. The result was a
  useful reminder that KNN's failure mode isn't random; it's predictable from
  the geometry of the data.
- **Overfitting was universal and stubborn.** Every model, including the one with
  the strongest regularization, showed a large train-test gap. The Random Forest
  was 97.1% train vs. 83.9% test; the baseline Logistic Regression was 99.9%
  train vs. 80.4% test. This suggests that the feature space, even with `min_df`
  filtering, still has more dimensions than the ~3,000-post training set can
  support. A next step would be more aggressive dimensionality reduction (higher
  `min_df`, PCA, or both).

### Software engineering practices

- **Separating data collection from analysis.** `reddit_scraper.py` handles all
  API calls and writes to CSV. The notebook reads from those CSVs and never
  touches the API. This means the analysis is reproducible without re-hitting
  Reddit — important since "newest posts" changes by the minute.
- **Environment variable management.** API credentials live in `.env` and are
  loaded with `python-dotenv`. The repository contains `.env.example` with
  placeholder values and `.gitignore` excludes the real `.env`.
- **Explicit credential validation.** The scraper raises a clear `EnvironmentError`
  if `REDDIT_ID` or `REDDIT_SECRET` are missing from the environment, rather than
  letting PRAW fail with a cryptic authentication error downstream.

### Unexpected learnings

- **r/DungeonsAndDragons is primarily an image community.** The vast majority of
  posts have no body text at all — people share art, character sheets, and
  screenshots without writing much. Pathfinder, being more rules-focused, has far
  more written discussion. This asymmetry nearly broke my initial plan to use body
  text as the corpus, and it's the reason I pivoted to concatenating title and
  body.
- **The top five longest posts by word count were all from r/Pathfinder_RPG.** One
  user posted a 7,173-word comprehensive system review, and two near-duplicates of
  that same review also appeared in the top five. Deduplication by post ID handled
  true duplicates, but near-duplicates (slightly different edits of the same post)
  survived into the corpus.
- **TF-IDF's `min_df` cut my feature space from potentially 50,000+ tokens to
  3,254.** That reduction made the difference between a model that trains in minutes
  and one that might take hours, with no meaningful accuracy loss.

### Design decisions

- **TF-IDF over raw counts:** Word frequency counts inflate the importance of
  words that are common to both subreddits (articles, prepositions, common nouns).
  TF-IDF down-weights these and highlights words that are distinctive within a
  document relative to the full corpus.
- **L1 regularization for Logistic Regression:** L1 drives less informative
  coefficients to exactly zero, effectively performing feature selection during
  training. With 3,254 features, this was more appropriate than L2, which
  shrinks all coefficients but keeps them non-zero.
- **VADER over a custom sentiment model:** VADER is pre-trained on social media
  text and handles informal language, capitalization patterns, and punctuation
  well without requiring fine-tuning. For a bootcamp project with a defined scope,
  it was the right tool.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Reddit developer account](https://www.reddit.com/prefs/apps) with an app
  registered as a "script" type

### Setup

```bash
# Clone the repo
git clone https://github.com/dmehta94/project-3-reddit-nlp.git
cd project-3-reddit-nlp

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows (GitBash)
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API credentials
cp .env.example .env
# Open .env and fill in your REDDIT_ID and REDDIT_SECRET
```

### Data Collection (optional — data already included)

```python
# Run once from within the code/ directory
from reddit_scraper import unified_data

unified_data('DungeonsAndDragons', 'all', 1000, 1000)
unified_data('Pathfinder_RPG', 'all', 1000, 1000)
```

The collected CSVs are included in `data/` and current as of 2024-11-25.
Re-running will produce different results since new posts change by the minute.

### Analysis

Open `code/analysis.ipynb` in JupyterLab and run all cells from top to bottom.
The data collection cell is intentionally commented out so the notebook runs
reproducibly without re-hitting the API.

---

## Sample Output

```
Baseline Logistic Regression (train / test):   99.9% / 80.4%
GridSearch Logistic Regression (train / test): 98.9% / 82.8%
K-Nearest Neighbors (train / test):            57.9% / 54.3%
Random Forest (train / test):                  97.1% / 83.9%  ← best
Support Vector Machine (train / test):         97.8% / 76.5%
```

Confusion matrices for each model are saved to `images/`.

---

## Technical Details

### Project structure

```
project-3-reddit-nlp/
├── code/
│   ├── analysis.ipynb       # Full EDA and modeling notebook
│   └── reddit_scraper.py    # PRAW data collection module
├── data/
│   ├── DungeonsAndDragons.csv
│   └── Pathfinder_RPG.csv
├── images/                  # Confusion matrix outputs
├── .env.example             # Credential template (copy to .env)
├── .gitignore
├── README.md
└── requirements.txt
```

### Key functions

| Function | Location | Purpose |
|---|---|---|
| `get_top_data(subreddit, time_filter, limit)` | `reddit_scraper.py` | Retrieve top posts from a subreddit |
| `get_new_data(subreddit, limit)` | `reddit_scraper.py` | Retrieve newest posts from a subreddit |
| `unified_data(subreddit, time_filter, limit_top, limit_new)` | `reddit_scraper.py` | Collect, deduplicate, and write posts to CSV |
| `custom_lemmatize(word, tag)` | `analysis.ipynb` | POS-aware lemmatization of a single token |
| `lemmatize(text)` | `analysis.ipynb` | Lemmatize a full document string (used as TF-IDF preprocessor) |
| `get_sentiment_score(doc)` | `analysis.ipynb` | Return VADER compound sentiment score for a document |

### Preprocessing pipeline

```
all_text (str)
    → TfidfVectorizer (lemmatize preprocessor, stop words, min_df=0.0025)
    → FunctionTransformer (sparse → dense)
compound_sentiment (float)
    → ColumnTransformer (pass through)
Combined → StandardScaler → Model
```

**Features after vectorization:** 3,254 TF-IDF features + 1 sentiment score = 3,255 total  
**Train / test split:** 75% / 25% (stratification not needed — classes are nearly balanced)

### Dependencies

```
praw
python-dotenv
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
```

---

## Limitations

- **All models overfit significantly.** The train-test gap ranges from 13 to 20
  percentage points. The feature space is large relative to the training data,
  and I did not explore dimensionality reduction techniques like PCA or SVD.
- **The data is not fully reproducible.** "Newest posts" is a snapshot in time.
  The included CSVs are current as of 2024-11-25; re-running collection will
  produce different results.
- **Near-duplicate posts are not removed.** One prolific Pathfinder reviewer
  posted slightly different versions of the same 7,000-word review, and all three
  versions survived into the corpus. A more robust pipeline would detect and
  handle near-duplicates (e.g., via cosine similarity).
- **VADER is a general-purpose sentiment tool.** It wasn't fine-tuned on TTRPG
  community language, so domain-specific tone ("this spell is broken" as a
  compliment) may be misclassified.
- **Community language is a proxy, not a legal test.** A classifier that
  distinguishes subreddit posts is not the same as a test for intellectual
  property similarity at the rules, mechanics, or narrative level. The
  inconclusive results here suggest the communities are meaningfully different
  in how they communicate — but that finding would need to be paired with
  analysis of the actual game texts before saying anything credible about
  infringement.

---

## Credits

- **ChatGPT 4o mini** — One consultation during development to understand how to
  chain a sparse `TfidfVectorizer` output into `StandardScaler` using
  `FunctionTransformer`. That conversation is
  [linked in the notebook](https://chatgpt.com/share/674401c4-2ad0-800c-b95f-498e594383f9).
- **Claude AI (Anthropic)** — Code cleanup, documentation, and standardization
  during portfolio preparation (March 2026).
- [Reddit user nullus_72](https://www.reddit.com/r/DnD/comments/11i9vjg/comment/jax4p5x/)
  for a thorough explanation of D&D vs. Pathfinder differences.
- [RexEgg RegEx Cheat Sheet](https://www.rexegg.com/regex-quickstart.php) for
  regex reference during tokenization.
- [GeeksforGeeks scikit-learn classification guide](https://www.geeksforgeeks.org/comprehensive-guide-to-classification-models-in-scikit-learn/)
  as a reference for model implementation patterns.

---

## License

MIT License — see `LICENSE` for details.  
Contact: [GitHub @dmehta94](https://github.com/dmehta94)
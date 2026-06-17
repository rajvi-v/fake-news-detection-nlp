# Fake News Detection

This project contains a Jupyter notebook, `fake-news-detection.ipynb`, for detecting fake news articles using natural language processing and machine learning.

The notebook loads real and fake news datasets, cleans the article text, visualizes common words, trains a TF-IDF based classifier, and evaluates the model using standard classification metrics.

## Dataset

The notebook uses the Kaggle Fake and Real News dataset by Clement Bisaillon.

Expected Kaggle input files:

- `Fake.csv`
- `True.csv`

Notebook dataset path:

```text
/kaggle/input/datasets/clmentbisaillon/fake-and-real-news-dataset/
```

The notebook assigns labels as follows:

- `1`: fake news
- `0`: true news

## Workflow

The notebook follows this process:

1. Load `Fake.csv` and `True.csv` with pandas.
2. Inspect article subjects and sample rows.
3. Add category labels for fake and true news.
4. Combine both datasets into one dataframe.
5. Visualize class and subject distributions with seaborn.
6. Keep only the `text` and `category` columns.
7. Check for missing values and blank text rows.
8. Clean article text by:
   - Lowercasing text.
   - Expanding common contractions.
   - Removing special characters and extra whitespace.
   - Removing stopwords from spaCy and NLTK.
   - Lemmatizing words with WordNet.
9. Generate word clouds for true and fake news articles.
10. Split the dataset into training and testing data.
11. Convert text into TF-IDF features.
12. Train a `LinearSVC` classifier.
13. Evaluate predictions with:
   - Classification report
   - Accuracy score
   - Confusion matrix

## Model

The final model is built with a scikit-learn pipeline:

```python
Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])
```

The notebook prints the model's classification report, accuracy score, and confusion matrix after predicting on the test set.

## Requirements

Install the main Python dependencies before running the notebook locally:

```bash
pip install numpy pandas matplotlib seaborn nltk spacy wordcloud pillow scikit-learn
python -m spacy download en_core_web_sm
```

The notebook also uses NLTK stopwords and WordNet data. If they are not already installed, run:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

## Running On Kaggle

1. Open `fake-news-detection.ipynb` in Kaggle.
2. Add the Fake and Real News dataset as notebook input.
3. Add the image dataset used for the word cloud masks if you want to run the masked word cloud cells:

```text
/kaggle/input/datasets/rajvivc/images/thumbs up.jpg
/kaggle/input/datasets/rajvivc/images/skull.jpg
```

4. Run all notebook cells.

## Running Locally

1. Install the dependencies listed above.
2. Download the Fake and Real News dataset from Kaggle.
3. Place `Fake.csv` and `True.csv` in a local data folder.
4. Update the notebook paths from Kaggle paths to your local paths, for example:

```python
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
```

5. If you do not have the word cloud mask images, either download them and update the paths or skip the masked word cloud cells.
6. Open and run the notebook:

```bash
jupyter notebook fake-news-detection.ipynb
```

## Project File

```text
fake-news-detection.ipynb
```

This notebook contains the complete data loading, preprocessing, visualization, training, and evaluation workflow.

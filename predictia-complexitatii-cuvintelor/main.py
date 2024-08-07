import os
import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
import nltk
from tqdm import tqdm
from collections import Counter
import wordfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import plotly.express as px
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import StandardScaler

BASE_DIR = 'predictia-complexitatii-cuvintelor'
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'test.csv')

# Load train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Plot distribution of complexity in training set
train['complexity'].plot(kind='hist', bins=20, title='Complexity Distribution')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Helper function to plot histogram by language
def plot_hist(df, criterion='language'):
    fig = px.histogram(df, x='complexity', color=criterion, marginal='rug', nbins=20, barmode='overlay',
                       title=f'Density Plot of Complexity by {criterion}',
                       labels={'complexity': 'Complexity', criterion: 'Language', 'count': 'Density'})
    fig.show()

# Plot histogram of complexity by language in training set
plot_hist(train)

# Display basic statistics of complexity by language
print(train.groupby('language')['complexity'].describe())
print(test.groupby('language')['sentence'].count())

# Tokenization function using NLTK
def tokenize(text):
    return nltk.tokenize.word_tokenize(text)

# Tokenize sentences in train and test sets and count token frequencies
all_tokens = []
for sentence in tqdm(train.sentence.values):
    all_tokens.extend(tokenize(sentence))

for sentence in tqdm(test.sentence.values):
    all_tokens.extend(tokenize(sentence))

counts = Counter(all_tokens)
print(counts.most_common(100))

# Custom evaluation metric combining R^2 and Pearson correlation
def pearson_r2(preds, y_true):
    r2 = r2_score(y_true, preds)
    r2 = max(0, r2)
    pears = pearsonr(y_true, preds)[0]
    pears = np.abs(np.nan_to_num(pears, 0))
    return (pears + r2) / 2

# Function to evaluate predictions
def evaluate(predictions, y_true):
    cust = pearson_r2(predictions, y_true)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    return {
        'r2_pearson': cust,
        'mae': mean_absolute_error(y_true, predictions),
        'mse': mean_squared_error(y_true, predictions),
        'rmse': rmse
    }

# Define language codes for wordfreq library
lang_code = {
    "catalan": "ca",
    "german": "de",
    "english": "en",
    "spanish": "es",
    "filipino": "fil",
    "french": "fr",
    "italian": "it",
    "japanese": "ja",
    "portuguese": "pt",
    "sinhala": "si"
}

# Load spacy models for each language
spacy_models = {
    'catalan': spacy.load('ca_core_news_md'),
    'german': spacy.load('de_core_news_md'),
    'english': spacy.load('en_core_web_md'),
    'spanish': spacy.load('es_core_news_md'),
    'filipino': spacy.load('en_core_web_md'),  # Use multilingual model for unsupported languages
    'french': spacy.load('fr_core_news_md'),
    'italian': spacy.load('it_core_news_md'),
    'japanese': spacy.load('ja_core_news_md'),
    'portuguese': spacy.load('pt_core_news_md'),
    'sinhala': spacy.load('en_core_web_md')  # Use multilingual model for unsupported languages
}

def get_frequency(word, language):
    language_code = lang_code.get(language.lower())
    if language_code:
        frequency = wordfreq.word_frequency(word, language_code)
        return frequency
    else:
        raise ValueError(f"Language '{language}' is not supported.")

def syllable_count(word):
    word = word.lower()
    vowels = 'aeiouy'
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count += 1
    return count

def vowel_count(word):
    word = word.lower()
    vowels = 'aeiouy'
    count = sum(1 for char in word if char in vowels)
    return count

def get_word_embeddings(word, language):
    doc = spacy_models[language].vocab[word]
    return doc.vector

def is_named_entity(word, sentence, language):
    doc = spacy_models[language](sentence)
    for ent in doc.ents:
        if word in ent.text:
            return 1
    return 0

def get_syntax_tree_features(word, sentence, language):
    doc = spacy_models[language](sentence)
    token = next((token for token in doc if token.text == word), None)
    if token:
        return len(list(token.children)), token.dep_
    return 0, ''

def get_pos_tag(word, language):
    doc = spacy_models[language](word)
    pos_tag = doc[0].pos_
    return pos_tag

# Function to get number of hypernyms and hyponyms for a word
def get_hypernym_hyponym_count(word):
    synsets = wn.synsets(word)
    hypernyms = set()
    hyponyms = set()

    for synset in synsets:
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym)
        for hyponym in synset.hyponyms():
            hyponyms.add(hyponym)

    return len(hypernyms), len(hyponyms)

# Function to check if the word is a title
def is_title(word, sentence):
    words = sentence.split()
    if word in words and words.index(word) != 0 and word[0].isupper():
        return 1
    return 0

def featurize_row(row):
    word = row['word']
    language = row['language']
    sentence = row['sentence']

    # Existing features
    all_features = [
        len(word),  # Word length
        get_frequency(word, language),  # Word frequency
        vowel_count(word),  # Vowel count
        syllable_count(word),  # Syllable count
        is_named_entity(word, sentence, language),  # Named entity
        is_title(word, sentence),  # Title
        *get_word_embeddings(word, language),  # Word embeddings
    ]

    # Add POS tag as a feature
    pos_tag = get_pos_tag(word, language)
    pos_tag_numeric = sum(ord(char) for char in pos_tag)  # Convert POS tag to a numeric value
    all_features.append(pos_tag_numeric)

    # Add hypernym and hyponym counts
    hypernyms_count, hyponyms_count = get_hypernym_hyponym_count(word)
    all_features.extend([hypernyms_count, hyponyms_count])

    # Add syntax tree features
    syntax_children, syntax_dep = get_syntax_tree_features(word, sentence, language)
    syntax_dep_numeric = sum(ord(char) for char in syntax_dep)  # Convert dependency tag to a numeric value
    all_features.extend([syntax_children, syntax_dep_numeric])

    return np.array(all_features)

# Update featurize_df to handle the new feature size
def featurize_df(df):
    nr_of_features = len(featurize_row(df.iloc[0]))
    nr_of_examples = len(df)
    features = np.zeros((nr_of_examples, nr_of_features))
    for i, (index, row) in tqdm(enumerate(df.iterrows()), total=nr_of_examples):
        row_ftrs = featurize_row(row)
        features[i, :] = row_ftrs
    return features

# Featurize train and test datasets with new features
X_train = featurize_df(train)
y_train = train.complexity.values
X_test = featurize_df(test)

# Ensure standardization of the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split data into train and validation sets
X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train and evaluate different models
models = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Initialize a dictionary to store model performances
model_performances = {}

for name, model in models.items():
    model.fit(X_train_full, y_train_full)
    val_predictions = model.predict(X_val)
    performance = evaluate(val_predictions, y_val)
    model_performances[name] = performance

# Print model performances
for name, performance in model_performances.items():
    print(f"{name} performance: {performance}")

# Select the best model based on r2_pearson score
best_model_name = max(model_performances, key=lambda name: model_performances[name]['r2_pearson'])
best_model = models[best_model_name]

# Train best model on full training data
best_model.fit(X_train, y_train)

# Make predictions on test data
test_predictions = best_model.predict(X_test)

submission = pd.DataFrame({'cur_id': test.cur_id.values, 'complexity': test_predictions})
submission.to_csv('submission.csv', index=False)


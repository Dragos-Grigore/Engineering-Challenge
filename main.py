import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.word as naw
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Downloaded any required resources
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Load data
company_df = pd.read_csv("ml_insurance_challenge.csv")
taxonomy_df = pd.read_csv("insurance_taxonomy - insurance_taxonomy.csv")

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):           #returns an empty string if the text isn't a string
        return ""
    text = text.lower()                     #lowercase the text
    text = text.translate(str.maketrans('', '', string.punctuation))            #eliminate the punctuation
    tokens = nltk.word_tokenize(text)       #tokenize the text
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]       #transform every word to it's dictionary form
    return ' '.join(lemmatized_tokens)

# Preprocess columns in company_df
for col in ['description', 'business_tags', 'sector', 'category', 'niche']:
    company_df[col] = company_df[col].apply(preprocess_text)

# Preprocess taxonomy labels
taxonomy_df['label'] = taxonomy_df['label'].dropna().apply(preprocess_text)
taxonomy_labels = taxonomy_df['label'].tolist()

# Set feature weights
feature_weights = {
    'description': 3.0,
    'business_tags': 2.5,
    'sector': 1.5,
    'category': 3.5,
    'niche': 1.5
}

# Combine features into one weighted string per company
def combine_features(row):
    combined = ""
    for col, weight in feature_weights.items():
        combined += (" " + row[col]) * int(weight)
    return combined.strip()

company_df['combined_text'] = company_df.apply(combine_features, axis=1)

# Combine all texts (companies + labels) for TF-IDF vectorizer
all_texts = company_df['combined_text'].tolist() + taxonomy_labels

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Split back to company vectors and label vectors
company_vectors = tfidf_matrix[:len(company_df)]
label_vectors = tfidf_matrix[len(company_df):]

# Cosine similarity between each company and each label
similarity_matrix = cosine_similarity(company_vectors, label_vectors)
row_sums = similarity_matrix.sum(axis=1, keepdims=True)

# Avoid division by zero using np.divide with where clause
prob_matrix = np.divide(
    similarity_matrix,
    row_sums,
    out=np.zeros_like(similarity_matrix),
    where=row_sums != 0
)
# Create a DataFrame of label probabilities
# Get top 3 labels for each company
top3_indices = np.argsort(-prob_matrix, axis=1)[:, :3]  # top 3 label indices per row
top3_labels = [[taxonomy_labels[i] for i in row] for row in top3_indices]
top3_probs = [[prob_matrix[row_idx, i] for i in row] for row_idx, row in enumerate(top3_indices)]

# Add to company_df
company_df['top1_label'] = [labels[0] for labels in top3_labels]
company_df['top2_label'] = [labels[1] for labels in top3_labels]
company_df['top3_label'] = [labels[2] for labels in top3_labels]

company_df['top1_prob'] = [round(probs[0], 4) for probs in top3_probs]
company_df['top2_prob'] = [round(probs[1], 4) for probs in top3_probs]
company_df['top3_prob'] = [round(probs[2], 4) for probs in top3_probs]

# Sorted values by first probability to see where would be the baseline where cosine similarity starts to give errors
company_df = company_df.sort_values(by='top1_prob', ascending=False)

# Ensure top1_prob is numeric
company_df['top1_prob'] = pd.to_numeric(company_df['top1_prob'], errors='coerce')

# Split the dataset
train_df = company_df[company_df['top1_prob'] > 0.10].copy()
test_df = company_df[company_df['top1_prob'] <= 0.10].copy()
print(train_df['top1_label'].value_counts())

# Drop probabilities and labels from test
test_features_df = test_df.drop(columns=['top1_prob','top1_label','top2_prob', 'top2_label','top3_prob', 'top3_label'])
test_df = test_df.drop(columns=['top1_prob','top1_label','top2_prob', 'top2_label','top3_prob', 'top3_label'])
# Created new dataframe for test without the probabilities
train_features_df = train_df.drop(columns=['top1_prob','top1_label','top2_prob', 'top2_label','top3_prob', 'top3_label'])
# Took all features and put in one column
train_features_df['combined_text'] = train_features_df.apply(combine_features, axis=1)
test_features_df['combined_text'] = test_features_df.apply(combine_features, axis=1)

# Create the augmenter using WordNet
syn_aug = naw.SynonymAug(aug_src='wordnet')

# Apply augmentation to a sample (or to multiple rows)
augmented_samples = train_features_df['combined_text'].apply(syn_aug.augment)
augmented_samples= augmented_samples.apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

# Use the combined_text column for vectorization
X_train_text = train_features_df['combined_text']
# Combined the original dataset with the created one
X_train_text=pd.concat([X_train_text,augmented_samples],ignore_index=True)

X_test_text = test_features_df['combined_text']
# Fit TF-IDF on training text and transform both train and test
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Encode the training labels (top1_label from train_df)
label_encoder = LabelEncoder()
y_df=train_df['top1_label']
y_df=pd.concat([y_df,y_df],ignore_index=True)
y_train = label_encoder.fit_transform(y_df)



# Define models to compare
models = {"""
"RandomForest1":RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
),
    "RandomForest2":RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=5,
    random_state=42
),
    "RandomForest3":RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    max_features=0.5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
),
"""
    "RandomForest4":RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    max_features='log2',
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42
)

}

# Evaluate using cross-validation on the train set
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

# Train a Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    max_features='log2',
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)

# Predict labels for test set
y_pred = rf.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Add predictions to test_df
test_df['insurance_label'] = y_pred_labels

# Save results
test_df = test_df.drop(columns=['combined_text'])
train_df=train_df.drop(columns=['combined_text','top1_prob','top2_prob', 'top2_label','top3_prob', 'top3_label'])
train_df = train_df.rename(columns={'top1_label': 'insurance_label'})
final_df=pd.concat([train_df,test_df],ignore_index=True)
final_df.to_csv("final_labels.csv", index=False)


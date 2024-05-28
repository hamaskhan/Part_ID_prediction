import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn.ensemble._forest", lineno=589)
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn.ensemble._forest", lineno=601)
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn.ensemble._forest", lineno=577)

# Load training data from CSV
df = pd.read_csv('training_data.csv')

# Encode the target variable (part_id)
le_part_id = LabelEncoder()
df['part_id_encoded'] = le_part_id.fit_transform(df['part_id'])

# Encode the organization IDs
le_org = LabelEncoder()
df['organization_encoded'] = le_org.fit_transform(df['organization'])

# Prepare the data for Word2Vec
sentences = [desc.split() for desc in df['description']]

# Train a Word2Vec model
print("Training Word2Vec model...")
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, seed=42)
word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)

def get_embedding(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros(100)  # Assuming 100-dimensional embeddings

def embed_description(description, model):
    words = description.split()
    embeddings = np.array([get_embedding(word, model) for word in words])
    return np.mean(embeddings, axis=0)

# Apply to the dataset with progress display
print("Embedding descriptions...")
df['description_embedded'] = list(tqdm(df['description'].apply(lambda x: embed_description(x, word2vec_model))))

# TF-IDF Vectorizer with n-grams
print("Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # unigrams and bigrams
tfidf_vectorizer.fit(df['description'])

# Combine TF-IDF and Word2Vec features
def combine_features(df, tfidf_vectorizer):
    tfidf_features = tfidf_vectorizer.transform(df['description'])
    embedded_features = np.vstack(df['description_embedded'].values)
    combined_features = hstack([df[['organization_encoded']], tfidf_features, embedded_features])
    return combined_features

X_combined = combine_features(df, tfidf_vectorizer)
y = df['part_id_encoded']

# Duplicate samples for classes with only one sample
print("Duplicating samples for classes with only one sample...")
class_counts = y.value_counts()
single_sample_classes = class_counts[class_counts == 1].index

# Append duplicated samples to the dataframe
duplicated_samples = df[df['part_id_encoded'].isin(single_sample_classes)]
df = pd.concat([df, duplicated_samples], ignore_index=True)

# Recompute X_combined and y after duplication
X_combined = combine_features(df, tfidf_vectorizer)
y = df['part_id_encoded']

# Apply SMOTE for minority classes
print("Applying SMOTE for minority classes...")
smote = SMOTE(sampling_strategy='minority', k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y)

# Split the resampled data into training and test sets
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Implementing BalancedRandomForestClassifier
print("Training Balanced Random Forest model...")
start_time = time.time()
balanced_rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, replacement=True, bootstrap=True, sampling_strategy='all', class_weight='balanced_subsample')
balanced_rf.fit(X_train_res, y_train_res)
end_time = time.time()
print(f"Training completed in {(end_time - start_time):.2f} seconds")

# Making predictions with Balanced Random Forest
print("Making predictions on test set with Balanced Random Forest...")
y_pred_rf = balanced_rf.predict(X_test_res)

# Evaluate the Balanced Random Forest model on the test set
accuracy_rf = accuracy_score(y_test_res, y_pred_rf)
print(f'Balanced Random Forest model accuracy on test data: {accuracy_rf:.2f}')
#print("Balanced Random Forest Classification report:\n", classification_report(y_test_res, y_pred_rf, target_names=le_part_id.classes_))

# Implementing AdaBoost with Balanced Random Forest as the base estimator
print("Training AdaBoost with Balanced Random Forest base estimator...")
start_time = time.time()
boosted_rf = AdaBoostClassifier(base_estimator=balanced_rf, n_estimators=20, random_state=42)
boosted_rf.fit(X_train_res, y_train_res)
end_time = time.time()
print(f"Training completed in {(end_time - start_time):.2f} seconds")

# Making predictions with AdaBoosted Balanced Random Forest
print("Making predictions on test set with AdaBoosted Balanced Random Forest...")
y_pred_boosted_rf = boosted_rf.predict(X_test_res)

# Evaluate the AdaBoosted Balanced Random Forest model on the test set
accuracy_boosted_rf = accuracy_score(y_test_res, y_pred_boosted_rf)
print(f'AdaBoosted Balanced Random Forest model accuracy on test data: {accuracy_boosted_rf:.2f}')
#print("AdaBoosted Balanced Random Forest Classification report:\n", classification_report(y_test_res, y_pred_boosted_rf, target_names=le_part_id.classes_))

# Function to predict part_id
def predict_part_id(org, desc, org_encoder, desc_vectorizer, word_model, trained_model):
    org_encoded = org_encoder.transform([org])
    desc_embedded = embed_description(desc, word_model)
    desc_tfidf = desc_vectorizer.transform([desc])
    combined_features = hstack([org_encoded.reshape(1, -1), desc_tfidf, desc_embedded.reshape(1, -1)])
    part_id_encoded = trained_model.predict(combined_features)
    return le_part_id.inverse_transform(part_id_encoded)[0]

# Load evaluation data from CSV
evaluation_df = pd.read_csv('evaluation_data.csv')

# Add columns to save results with progress display
print("Predicting on evaluation data...")
evaluation_df['predicted_part_id'] = list(tqdm(evaluation_df.apply(
    lambda row: predict_part_id(row['organization'], row['description'], le_org, tfidf_vectorizer, word2vec_model, boosted_rf), axis=1)))
evaluation_df['correct/incorrect'] = evaluation_df.apply(
    lambda row: 'correct' if row['part_id'] == row['predicted_part_id'] else 'incorrect', axis=1)

# Save results to CSV
evaluation_df.to_csv('result.csv', index=False)

# Calculate evaluation score
accuracy_evaluation = accuracy_score(evaluation_df['part_id'], evaluation_df['predicted_part_id'])
print(f'Model accuracy on evaluation data: {accuracy_evaluation:.2f}')

# End timing the prediction process
total_time = time.time() - start_time
print(f'Total time taken for prediction on the evaluation dataset: {total_time:.2f} seconds')
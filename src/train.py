import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from scipy.sparse import hstack
from tqdm import tqdm
import pickle
from gensim.models import Word2Vec
from utils.preprocess import DataProcessor

class ModelPipeline:
    """
    A class to handle the entire machine learning pipeline including data preprocessing,
    model training, and saving of models.
    """
    
    def __init__(self, dataset_path, model_dir='models'):
        """
        Initialize the ModelPipeline with the dataset path and model directory.
        
        Parameters:
        dataset_path (str): Path to the dataset CSV file.
        model_dir (str): Directory where the models will be saved.
        """
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.data_processor = DataProcessor(data_path=self.dataset_path)
        self.le_part_id = LabelEncoder()
        self.le_org = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # unigrams and bigrams
        self.word2vec_model = None
        self.svm_model = SVC(kernel='linear', random_state=42)
        
    def preprocess_data(self):
        """
        Preprocess the dataset with SMOTE and embeddings.
        """
        print('\nPreprocessing dataset with SMOTE and Embeddings...')
        self.data_processor.duplicate_rare_classes()
        self.data_processor.apply_smote_and_generate()
        self.data_processor.save_counts()  # optional
        self.data_processor.save_generated_data()  # optional
        self.data_processor.save_training_data()
        
    def load_processed_data(self):
        """
        Load the processed data after SMOTE and embeddings.
        """
        df = self.data_processor.generated_data.drop(columns=['generated'])
        df['part_id_encoded'] = self.le_part_id.fit_transform(df['part_id'])
        df['organization_encoded'] = self.le_org.fit_transform(df['organization'])
        return df
    
    def split_data(self, df):
        """
        Split the data into training and test sets.
        
        Parameters:
        df (pd.DataFrame): The preprocessed DataFrame.
        
        Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        """
        X = df[['organization_encoded', 'description']]
        y = df['part_id_encoded']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_word2vec(self, sentences):
        """
        Train a Word2Vec model.
        
        Parameters:
        sentences (list of list of str): Tokenized sentences for Word2Vec training.
        """
        print("\nTraining Word2Vec model...")
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, seed=42)
        self.word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)
        
    def get_embedding(self, word):
        """
        Get the Word2Vec embedding for a given word.
        
        Parameters:
        word (str): The word to get the embedding for.
        
        Returns:
        np.ndarray: The embedding vector for the word.
        """
        try:
            return self.word2vec_model.wv[word]
        except KeyError:
            return np.zeros(100)  # Assuming 100-dimensional embeddings
        
    def embed_description(self, description):
        """
        Embed a description using the Word2Vec model.
        
        Parameters:
        description (str): The description text to embed.
        
        Returns:
        np.ndarray: The embedding vector for the description.
        """
        words = description.split()
        embeddings = np.array([self.get_embedding(word) for word in words])
        return np.mean(embeddings, axis=0)
    
    def embed_descriptions(self, X):
        """
        Embed descriptions for a dataset.
        
        Parameters:
        X (pd.DataFrame): The dataset containing descriptions.
        
        Returns:
        pd.Series: A series of embedded descriptions.
        """
        return X['description'].apply(lambda x: self.embed_description(x))
    
    def vectorize_and_combine_features(self, X_train, X_test):
        """
        Vectorize descriptions with TF-IDF and combine with other features.
        
        Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        
        Returns:
        X_train_combined (scipy.sparse.csr.csr_matrix): Combined training features.
        X_test_combined (scipy.sparse.csr.csr_matrix): Combined test features.
        """
        print("Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train['description'])
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test['description'])
        
        X_train_embedded = np.vstack(X_train['description_embedded'].values)
        X_test_embedded = np.vstack(X_test['description_embedded'].values)
        
        X_train_combined = hstack([X_train[['organization_encoded']], X_train_tfidf, X_train_embedded])
        X_test_combined = hstack([X_test[['organization_encoded']], X_test_tfidf, X_test_embedded])
        
        return X_train_combined, X_test_combined
    
    def train_svm(self, X_train_combined, y_train):
        """
        Train an SVM model.
        
        Parameters:
        X_train_combined (scipy.sparse.csr.csr_matrix): Combined training features.
        y_train (pd.Series): Training labels.
        """
        print("Training SVM model...")
        self.svm_model.fit(X_train_combined, y_train)
        
    def save_models(self):
        """
        Save the trained models and encoders.
        """
        with open(os.path.join(self.model_dir, 'svm_model.pkl'), 'wb') as f:
            pickle.dump(self.svm_model, f)
        with open(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(os.path.join(self.model_dir, 'label_encoder_part_id.pkl'), 'wb') as f:
            pickle.dump(self.le_part_id, f)
        with open(os.path.join(self.model_dir, 'label_encoder_org.pkl'), 'wb') as f:
            pickle.dump(self.le_org, f)
        self.word2vec_model.save(os.path.join(self.model_dir, 'word2vec_model.model'))
        print('Saved models (SVM, Word2Vec) and TFIDF_Vectorizer, Label Encoders')
        
def main():
    # Define the dataset path relative to the project directory
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print('Project directory: ', project_directory)
    
    dataset_path = os.path.join(project_directory, 'data', 'raw', 'dataset.csv')
    
    # Initialize the ModelPipeline
    pipeline = ModelPipeline(dataset_path)
    
    # Preprocess the data
    pipeline.preprocess_data()
    
    # Load the processed data
    df = pipeline.load_processed_data()
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = pipeline.split_data(df)
    
    # Prepare the data for Word2Vec
    sentences = [desc.split() for desc in df['description']]
    
    # Train Word2Vec model
    pipeline.train_word2vec(sentences)
    
    # Embed descriptions
    print("Embedding descriptions...")
    X_train['description_embedded'] = list(tqdm(pipeline.embed_descriptions(X_train)))
    X_test['description_embedded'] = list(tqdm(pipeline.embed_descriptions(X_test)))
    
    # Vectorize descriptions and combine with other features
    X_train_combined, X_test_combined = pipeline.vectorize_and_combine_features(X_train, X_test)
    
    # Train SVM model
    pipeline.train_svm(X_train_combined, y_train)
    
    # Save the models and encoders
    pipeline.save_models()
    
if __name__ == "__main__":
    main()

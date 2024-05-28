import os
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from gensim.models import Word2Vec
import pickle
from sklearn.metrics import accuracy_score
from utils.prediction_analysis import ResultProcessor

class PartIdPredictor:
    """
    A class to predict part IDs based on organization and description.
    """

    def __init__(self):
        """
        Initialize the PartIdPredictor by loading necessary models and encoders.
        """
        # Load the models
        with open('models/svm_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open('models/label_encoder_part_id.pkl', 'rb') as f:
            self.le_part_id = pickle.load(f)
        with open('models/label_encoder_org.pkl', 'rb') as f:
            self.le_org = pickle.load(f)
        self.word2vec_model = Word2Vec.load('models/word2vec_model.model')

    def get_embedding(self, word):
        """
        Get the word embedding vector from the Word2Vec model.

        Args:
            word (str): The word to get the embedding for.

        Returns:
            np.array: The embedding vector for the word.
        """
        try:
            return self.word2vec_model.wv[word]
        except KeyError:
            return np.zeros(100)  # Assuming 100-dimensional embeddings

    def embed_description(self, description):
        """
        Embed a description text using Word2Vec word embeddings.

        Args:
            description (str): The description text to embed.

        Returns:
            np.array: The embedded representation of the description.
        """
        words = description.split()
        embeddings = np.array([self.get_embedding(word) for word in words])
        return np.mean(embeddings, axis=0)

    def predict_part_id(self, org, desc):
        """
        Predict the part ID for a given organization and description.

        Args:
            org (str): The organization.
            desc (str): The description.

        Returns:
            str: The predicted part ID.
        """
        org_encoded = self.le_org.transform([org])
        desc_embedded = self.embed_description(desc)
        desc_tfidf = self.tfidf_vectorizer.transform([desc])
        combined_features = hstack([org_encoded.reshape(1, -1), desc_tfidf, desc_embedded.reshape(1, -1)])
        part_id_encoded = self.model.predict(combined_features)
        return self.le_part_id.inverse_transform(part_id_encoded)[0]

def evaluate_model():
    """
    Evaluate the PartIdPredictor model on evaluation data.
    """
    # Load evaluation data from CSV
    evaluation_df = pd.read_csv('data/evaluation/dataset.csv')

    # Ensure the output directory exists
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start timing the prediction process
    import time
    start_time = time.time()

    # Instantiate the predictor
    predictor = PartIdPredictor()

    # Add columns to save results with progress display
    print("Predicting on evaluation data...")
    evaluation_df['predicted_part_id'] = evaluation_df.apply(
        lambda row: predictor.predict_part_id(row['organization'], row['description']), axis=1)
    evaluation_df['correct/incorrect'] = np.where(evaluation_df['part_id'] == evaluation_df['predicted_part_id'], 'correct', 'incorrect')

    # Save output with correct/incorrect tags and results to CSV
    results_path = os.path.join(output_dir, 'output.csv')
    evaluation_df.to_csv(results_path, index=False)

    # Save predicted column to CSV with header "prediction"
    prediction_path = os.path.join(output_dir, 'results.csv')
    evaluation_df[['predicted_part_id']].rename(columns={'predicted_part_id': 'prediction'}).to_csv(prediction_path, index=False)

    # Calculate evaluation score
    accuracy_evaluation = accuracy_score(evaluation_df['part_id'], evaluation_df['predicted_part_id'])
    print(f'Model accuracy on evaluation data: {accuracy_evaluation:.2f}')

    # End timing the prediction process
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for prediction on the evaluation dataset: {total_time:.2f} seconds')

    # Call results_analysis method of ResultProcessor
    processor = ResultProcessor('output/output.csv', 'output/processed_results.csv')
    processor.results_analysis()

if __name__ == "__main__":
    evaluate_model()

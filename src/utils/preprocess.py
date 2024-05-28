import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE

class DataProcessor:
    """
    Class to handle the data processing pipeline including loading data, training Word2Vec model,
    applying SMOTE, and generating synthetic data.

    Attributes:
        data (pd.DataFrame): The dataset.
        word2vec_model (Word2Vec): The Word2Vec model.
        smote (SMOTE): The SMOTE instance.
        generated_data (pd.DataFrame): The generated dataset with synthetic samples.
    """

    def __init__(self, data_path, vector_size=100, window=5, min_count=1, workers=4, smote_strategy='not majority'):
        """
        Initialize the DataProcessor with dataset and model parameters.

        Parameters:
            data_path (str): Path to the dataset CSV file.
            vector_size (int): Dimensionality of the Word2Vec embeddings.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads to train the model.
            smote_strategy (str): SMOTE sampling strategy.
        """
        self.data = pd.read_csv(data_path)
        self.word2vec_model = None
        self.smote = SMOTE(random_state=42, k_neighbors=1, sampling_strategy=smote_strategy)
        self.generated_data = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self._train_word2vec_model()

    def _train_word2vec_model(self):
        """Train the Word2Vec model on the dataset descriptions."""
        sentences = [desc.split() for desc in self.data['description']]
        self.word2vec_model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                                       min_count=self.min_count, workers=self.workers, seed=42)
        self.word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)

    def get_embedding(self, word):
        """
        Get the embedding of a word using the Word2Vec model.

        Parameters:
            word (str): The word to get the embedding for.

        Returns:
            np.ndarray: The embedding of the word.
        """
        try:
            return self.word2vec_model.wv[word]
        except KeyError:
            return np.zeros(self.vector_size)  # Assuming vector_size-dimensional embeddings

    def embed_description(self, description):
        """
        Embed a description using Word2Vec.

        Parameters:
            description (str): The description to embed.

        Returns:
            np.ndarray: The embedding of the description.
        """
        words = description.split()
        embeddings = np.array([self.get_embedding(word) for word in words])
        return np.mean(embeddings, axis=0)

    def reverse_embedding(self, embedding):
        """
        Reverse an embedding back to a description using Word2Vec.

        Parameters:
            embedding (np.ndarray): The embedding to reverse.

        Returns:
            str: The description corresponding to the embedding.
        """
        words = self.word2vec_model.wv.most_similar(positive=[embedding], topn=10)
        return ' '.join([word for word, _ in words])

    def duplicate_rare_classes(self, threshold=3):
        """
        Duplicate classes with less than or equal to a threshold number of samples.

        Parameters:
            threshold (int): The maximum number of samples a class can have to be duplicated.
        """
        class_counts = self.data['part_id'].value_counts()
        duplicate_classes = class_counts[class_counts <= threshold].index

        for part_id in duplicate_classes:
            duplicate_samples = self.data[self.data['part_id'] == part_id]
            self.data = pd.concat([self.data, duplicate_samples], ignore_index=True)

    def apply_smote_and_generate(self):
        """
        Apply Word2Vec embeddings, SMOTE, and reverse operations separately for each organization,
        generating synthetic data.
        """
        generated_data = []
        for organization, org_data in self.data.groupby('organization'):
            X_org = org_data['description']
            y_org = org_data['part_id']

            X_embeddings = np.array([self.embed_description(desc) for desc in X_org])
            X_resampled, y_resampled = self.smote.fit_resample(X_embeddings, y_org)

            original_len = len(X_org)
            synthetic_indices = list(range(original_len, len(X_resampled)))
            synthetic_descriptions = [self.reverse_embedding(embedding) for embedding in X_resampled[synthetic_indices]]
            synthetic_organizations = [organization] * len(synthetic_indices)

            synthetic_data = pd.DataFrame({
                'description': synthetic_descriptions,
                'organization': synthetic_organizations,
                'part_id': y_resampled[synthetic_indices],
                'generated': 'Generated'
            })

            org_data['generated'] = 'Original'
            generated_data.append(pd.concat([org_data, synthetic_data], ignore_index=True))

        self.generated_data = pd.concat(generated_data, ignore_index=True)

    def save_counts(self, before_path='data/processed/before.csv', after_path='data/processed/after.csv'):
        """
        Save the counts of samples per class before and after applying SMOTE.

        Parameters:
            before_path (str): Path to save the counts before SMOTE.
            after_path (str): Path to save the counts after SMOTE.
        """
        before_counts = self.data.groupby(['organization', 'part_id']).size().reset_index(name='count')
        before_counts.to_csv(before_path, index=False)

        after_counts = self.generated_data.groupby(['organization', 'part_id']).size().reset_index(name='count')
        after_counts.to_csv(after_path, index=False)

    def save_generated_data(self, path='data/processed/generated.csv'):
        """
        Save the generated dataset with synthetic samples to a CSV file.

        Parameters:
            path (str): Path to save the generated dataset.
        """
        self.generated_data.to_csv(path, index=False)

    def save_training_data(self, path='data/processed/dataset_smote.csv'):
        """
        Save the training dataset without the 'generated' column to a CSV file.

        Parameters:
            path (str): Path to save the training dataset.
        """
        training_data = self.generated_data.drop(columns=['generated'])
        training_data.to_csv(path, index=False)

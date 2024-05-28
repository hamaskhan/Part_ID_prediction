# Part_ID_prediction
Predict the part_id from the organization and description data

Here are some key points:

1. Preprocessing: To address the class imbalance, I used SMOTE (Synthetic Minority Over-sampling Technique) for the minority class. Specifically, I utilized embeddings of the descriptions to resample the nearest neighbors using SMOTE. The mapping between synthetic descriptions and their corresponding organization and part IDs is maintained, after the reversal of embeddings back to descriptions.

2. Prediction Algorithm: I combined Word2Vec embeddings and TFIDF features to train a Support Vector Classifier (SVC).

3. Evaluation: The problem description specifies using `dataset.csv` for prediction. The average prediction accuracy for all classes is approximately 95% on `dataset.csv`. However, due to the imbalance in part IDs across different organizations, a more informative metric is the per part ID accuracy for each organization, which can be found in `output/processed_results.csv`.

Using `dataset.csv` directly for prediction would inflate accuracy due to overfitting the training data. Therefore, I used `train_test_split.py` to set aside about 20% of the data as a holdout/test set to assess per-part ID accuracy for each organization, which is crucial for minority classes. This can be further used to calculate the Precision-Recall AUC (PR-AUC) scores.

On the unseen holdout/test set, the initial SVM without SMOTE achieved an average class accuracy of ~75%. With SMOTE applied to description embeddings, the average class accuracy improved to ~93% on the unseen holdout/test set in subsequent experiments.

Additional Experiments: I also experimented with RandomForest and Bagging/Boosting, but their average per-class accuracy was lower compared to the SVM. I attempted further optimization of the SVM parameters using GridSearch, but this was quite time-consuming on my MacBook Air hardware.

The ‘src/predict.py’ has the code to run the prediction and show the time taken on the dataset.csv. The prediction time on individual instances is very fast (~1 second) as the model is light. 

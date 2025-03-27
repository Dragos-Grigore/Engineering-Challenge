# Engineering-Challenge
This project implements a hybrid unsupervised + supervised machine learning pipeline to automatically classify company descriptions into a predefined insurance taxonomy. It uses a combination of text similarity, TF-IDF vectorization, data augmentation, and a Random Forest classifier.

The workflow includes the following key steps:

Preprocessing:

-Cleans and lemmatizes text from multiple metadata fields (e.g., description, tags, category).

-Removes punctuation, lowercases text and lemmatizes the text.

-Considered using other preprocessing techniques, like eliminating stopwords but got better results without eliminating them.

-Created a new column named combined_text, where all features where put in a single string

Weak Labeling (Unsupervised):

Used cosine similarity between the new combined_text column and the taxonomy labels.

Assigns top-3 candidate labels and probabilities per company.

Confidence-Based Split:

-Splits the data into:

-Training set: where cosine similarity is confident (top1_prob > 0.10)

-Test set: where predictions are weak (used for final labeling)

-Verified where would be the baseline and observed that the first label is associated correctly when probability>0.10.

Data Augmentation:

-Applies WordNet-based synonym replacement to enrich the training data.

Supervised Classification:

-Trains a Random Forest classifier on the augmented training data.

-Used other models but observed that Random Forest classifier give best results.

-Used hyperparameter tuning to see which parameters give best results.

-Evaluates using 5-fold cross-validation (accuracy + std).

-Splits your training data into 5 folds (because cv=5)

-Trains the model on 4 folds and tests on the remaining 1

-Repeats this process 5 times (each fold gets a turn as the test set)

-Returns an array of 5 accuracy scores (1 per fold)

-The results are Mean Accuracy = 0.8594, Std = 0.0148.

-Got better results for Accuracy like 0.9742 but it definitly overfits.

Predicts labels for the unlabeled test set.

Export Final Results:

Combines both labeled and predicted data into one final dataset.

The solutions excels because the cosine similarity got good results and transformed an unsupervised problem into a uspervised one.
The solution needs improvement to baseline because >0.10 seems too low.
Cosine similarity doesn't find an occurence in dataset for every label.
The solution would get better results for a bigger dataset as it will have the chance to cover more labels and get more examples for the covered ones.
An assumption which was made is that all the labels with the probability above 0.10 are correct.
This solution was used because the dataset has many keywords in description and category which also appear in the labels, so using cosine similarity to label a part of the dataset seemed a good idea.

Saves to final_labels.csv.

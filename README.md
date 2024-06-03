# Duplicate Questions Detection
## Description
Imagine a platform where a physicist can help a chef with a math problem and get cooking tips in return. Such a place exists, enabling people to share knowledge on anything. With millions of users, it's common to see similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer and make contributors feel they need to answer multiple versions.

This platform values canonical questions to enhance the user experience for seekers and writers. In this project, we tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. This will make it easier to find high-quality answers, improving the experience for all users.
## Project Overview
In this project, we aim to classify pairs of questions as duplicates or non-duplicates using natural language processing (NLP) techniques, feature engineering, and machine learning models. The steps involved include text preprocessing, feature engineering, and applying models like Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) for classification.

## Notebooks

- Bag of Words Model: [Notebook implementing the BoW model.](https://www.kaggle.com/code/gyanbardhan/bow-00)
- TF-IDF Model: [Notebook implementing the TF-IDF model](https://github.com/Gyanbardhan/Duplicate-Question/blob/main/TF-IDF.ipynb).

## Datasets
We use the Quora Question Pairs dataset, which contains pairs of questions and a label indicating whether the questions are duplicates or not.

Dataset:- [Quora Question Pairs.](https://www.kaggle.com/datasets/gyanbardhan/quora-duplicate-questions-copy)

## Text Preprocessing
Text preprocessing is a crucial step in NLP tasks to ensure the text data is clean and suitable for model training. The steps include:

- Tokenization: Splitting text into individual words or tokens.
- Lowercasing: Converting all text to lowercase to maintain uniformity.
- Stop Words Removal: Removing common words that do not contribute to the meaning.
- Stemming/Lemmatization: Reducing words to their root form.
- Removing Special Characters: Eliminating non-alphanumeric characters.
## Feature Engineering
Feature engineering involves creating features that help machine learning models understand the data better. The techniques used include:
### Basic Features
- q1_len: Char Length of Question1
- q2_len: Char Length of Question2
- q1_words: No. of words in Question1
- q2_words: No. of words in Question2
- words_common: No. of unique common words
- words_total: q1_words + q2_words
- word_share: (word_common)/(word_total)
### Advanced Features
1. Token Features
- cwc_min: This is the ratio of the number of common words to the length of the smaller question
- cwc_max: This is the ratio of the number of common words to the length of the larger question
- csc_min: This is the ratio of the number of common stop words to the smaller stop word count among the two questions
- csc_max: This is the ratio of the number of common stop words to the larger stop word count among the two questions
- ctc_min: This is the ratio of the number of common tokens to the smaller token count among the two questions
- ctc_max: This is the ratio of the number of common tokens to the larger token count among the two questions
- last_word_eq: 1 if the last word in the two questions is same, 0 otherwise
- first_word_eq: 1 if the first word in the two questions is same, 0 otherwise
2. Length Based Features
- mean_len: Mean of the length of the two questions (number of words)
- abs_len_diff: Absolute difference between the length of the two questions (number of words)
- longest_substr_ratio: Ratio of the length of the longest substring among the two questions to the length of the smaller question
3. Fuzzy Features
- fuzz_ratio: fuzz_ratio score from fuzzywuzzy
- fuzz_partial_ratio: fuzz_partial_ratio from fuzzywuzzy
- token_sort_ratio: token_sort_ratio from fuzzywuzzy
- token_set_ratio: token_set_ratio from fuzzywuzzy

## Models 
We employ different models to classify question pairs with [<b>Random Forest Classifier<b>](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html):

### Bag of Words (BoW)
BoW is a simple and effective method to represent text data as a matrix of token counts. Each row represents a question pair, and each column represents a token from the vocabulary.

### Term Frequency-Inverse Document Frequency (TF-IDF)
TF-IDF is an advanced technique that adjusts the token counts by considering the frequency of tokens in the entire corpus. It highlights important tokens while reducing the weight of common ones.

## Model Evaluation
The final notebook compares different models based on their performance metrics, such as accuracy,confusion_matrix. The best-performing model is selected for deployment.


## Join Us
Join us in our quest to improve the experience by identifying duplicate questions efficiently. Together, we can help users find high-quality answers quickly, enhancing their overall experience on the platform.

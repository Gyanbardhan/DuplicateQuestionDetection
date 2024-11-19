# Duplicate Question Detection Using NLP and Transformers

[Detailed Report](https://github.com/Gyanbardhan/DuplicateQuestionDetection/blob/main/Report.pdf)

## Description
This project addresses the challenge of detecting duplicate questions on online platforms using advanced NLP techniques and machine learning models. By identifying similar questions, we enhance the user experience by reducing repetitive content and making it easier to find high-quality answers quickly. The solution integrates two distinct approaches for classification: Bag of Words (BoW) with Random Forest and DistilBERT, each offering unique insights into the semantic relationships between questions.
## Project Overview
In this project, we classify question pairs as duplicates or non-duplicates using a combination of traditional feature engineering methods and advanced deep learning techniques. The key steps include data preprocessing, feature engineering, model training with scikit-learn and Hugging Face's Transformers library, and deployment for efficient evaluation.

## Models
### Approach 1: Bag of Words (BoW) with Random Forest Classifier
#### Vectorization: 
- BoW and TF-IDF are used to represent questions as numerical features.
#### Feature Engineering: 
- Includes basic, token-based, length-based, and fuzzy features.
#### Modeling: 
- Random Forest Classifier is trained on engineered features to classify question pairs as duplicates or non-duplicates.
#### Performance: 
- Achieved 81.67% accuracy in detecting duplicate questions.
### Approach 2: DistilBERT Transformer (Deep Learning Model)
#### Tokenization: 
- The DistilBERT tokenizer splits questions into subword units, which are converted into embeddings. These embeddings capture the meaning of each question in context.
#### Self-Attention Mechanism: 
- DistilBERT utilizes a self-attention mechanism, which allows the model to weigh the relevance of each word in the context of the others, helping to understand deeper semantic relationships between words and sentences.
#### Modeling: 
- DistilBERT is fine-tuned on the Quora Question Pairs dataset for the task of duplicate question classification. This fine-tuning involves adjusting the weights of the model's parameters based on the specific dataset to improve accuracy in predicting whether two questions are duplicates or not.
#### Performance: 
- DistilBERT significantly outperformed the traditional methods, achieving 89.89% accuracy, which is a substantial improvement over BoW-based models. It effectively understands the contextual relationships between questions and handles variations in phrasing, making it highly suitable for detecting semantic duplicates.
#### Key Benefit: 
- Unlike traditional models, DistilBERT benefits from deep learning's ability to understand nuances in sentence structure and context, which is particularly useful for detecting semantically similar questions even if they are phrased differently.


## Notebooks
#### Approach 1 :
- Bag of Words Model: [Notebook implementing the BoW model.](https://www.kaggle.com/code/gyanbardhan/bow-00)
- TF-IDF Model: [Notebook implementing the TF-IDF model](https://github.com/Gyanbardhan/Duplicate-Question/blob/main/TF-IDF.ipynb)
#### Approach 2 :
- DisilBERT : [Notebook implementing the DistilBERT.](https://huggingface.co/spaces/gyanbardhan123/Bert_DuplicateQuestionDetection/blob/main/Bert%20Duplicate%20Question%20Detection.ipynb)

## Datasets
We use the Quora Question Pairs dataset, which contains pairs of questions and a label indicating whether the questions are duplicates or not.

- Dataset:- [Quora Question Pairs.](https://www.kaggle.com/datasets/gyanbardhan/quora-duplicate-questions-copy)

## Text Preprocessing
Text preprocessing is an essential step for cleaning and preparing the text data for modeling:

- Tokenization: Splitting text into individual words or subwords.
- Lowercasing: Ensuring uniformity by converting text to lowercase.
- Stop Words Removal: Eliminating non-contributory words like "is", "the", etc.
- Stemming/Lemmatization: Reducing words to their base form.
- Special Character Removal: Cleaning the text by removing unwanted characters. 
## Feature Engineering
#### Basic Features:
- q1_len, q2_len: Character lengths of the two questions.
- q1_words, q2_words: Word count in each question.
- words_common: Common words between both questions.
- words_total: Total number of words in both questions combined.
- word_share: Ratio of common words to total words.
#### Token Features:
- cwc_min, cwc_max: Ratios of common words to smaller and larger question lengths.
- csc_min, csc_max: Ratios of common stop words to smaller and larger stop word counts.
- first_word_eq, last_word_eq: Binary features indicating if the first or last words of the questions are the same.
#### Length-Based Features:
- mean_len: Average length of the two questions.
- abs_len_diff: Absolute difference in word count between the two questions.
- longest_substr_ratio: Ratio of the longest common substring to the smaller question length.
#### Fuzzy Features:
- fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio: Various fuzzy similarity scores between the questions.
## Model Evaluation
The models are evaluated using accuracy and confusion matrices to assess performance. The best-performing model (DistilBERT) achieved a significant improvement in accuracy compared to traditional models (BoW with Random Forest).
- Approach 1: Random Forest (81.67%) and
- Approach2: DistilBERT (89.89%)

## Deployment
Both models are deployed on Hugging Face Spaces using a 16 GB CPU environment, optimized for efficient evaluation.

- Approach 1 (Random Forest with BoW): [Hugging Face Space - Duplicate Question Detection (Random Forest)](https://huggingface.co/spaces/gyanbardhan123/Duplicate_Question_Detection  )
- Approach 2 (DistilBERT): Hugging Face Space - [Duplicate Question Detection (DistilBERT)](https://huggingface.co/spaces/gyanbardhan123/Bert_DuplicateQuestionDetection  )
## Key Takeaways
- BoW with Random Forest: Provides a traditional, efficient solution with 81.67% accuracy.
- DistilBERT: Leverages advanced transformer-based architecture, achieving 89.89% accuracy with superior understanding of question context.
## Join Us
Join us in our quest to improve the experience by identifying duplicate questions efficiently. Together, we can help users find high-quality answers quickly, enhancing their overall experience on the platform.
## Keywords
- NLP, Duplicate Question Detection, Machine Learning, DistilBERT, Random Forest, BoW, TF-IDF, Feature Engineering, Hugging Face, Transformer Models, Quora Dataset

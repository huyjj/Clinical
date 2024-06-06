from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk import ngrams
from rouge import Rouge

class EvaluationMetrics:
    def __init__(self):
        self.rouge = Rouge()

    def calculate_conciseness(self, generated_text, reference_texts):
        """
        Calculate the conciseness score of the generated text.
        
        Parameters:
        generated_text (str): The generated text.
        reference_texts (list): List of reference texts.
        
        Returns:
        float: Conciseness score.
        """
        texts = reference_texts + [generated_text]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        generated_vector = tfidf_matrix[-1].toarray()[0]
        reference_vectors = tfidf_matrix[:-1].toarray()
        reference_mean_vector = np.mean(reference_vectors, axis=0)
        cosine_similarity = np.dot(generated_vector, reference_mean_vector) / (np.linalg.norm(generated_vector) * np.linalg.norm(reference_mean_vector))
        conciseness_score = cosine_similarity / len(str(generated_text).split())
        return conciseness_score

    def calculate_redundancy(self, generated_text, n):
        """
        Calculate the redundancy of the generated text based on n-grams.
        
        Parameters:
        generated_text (str): The generated text.
        n (int): The size of n-grams (e.g., 1 for unigram, 2 for bigram).
        
        Returns:
        float: Redundancy score.
        """
        words = generated_text.split()
        n_grams = list(ngrams(words, n))
        total_n_grams = len(n_grams)
        if total_n_grams == 0:
            return 0.0
        n_gram_counts = Counter(n_grams)
        unique_n_grams = sum(1 for count in n_gram_counts.values() if count == 1)
        redundancy = 1 - (unique_n_grams / total_n_grams)
        return redundancy

    def calculate_repetition_rate(self, generated_text, n):
        """
        Calculate the repetition rate of the generated text based on n-grams.
        
        Parameters:
        generated_text (str): The generated text.
        n (int): The size of n-grams (e.g., 1 for unigram, 2 for bigram).
        
        Returns:
        float: Repetition rate.
        """
        words = generated_text.split()
        n_grams = list(ngrams(words, n))
        total_n_grams = len(n_grams)
        if total_n_grams == 0:
            return 0.0
        n_gram_counts = Counter(n_grams)
        repeated_n_grams = sum(1 for count in n_gram_counts.values() if count > 1)
        repetition_rate = repeated_n_grams / total_n_grams
        return repetition_rate

    def informativeness_score(self, generated_text, reference_texts):
        """
        Calculate the informativeness score of the generated text.
        
        Parameters:
        generated_text (str): The generated text.
        reference_texts (list): List of reference texts.
        
        Returns:
        float: Informativeness score.
        """
        texts = reference_texts + [generated_text]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        generated_vector = tfidf_matrix[-1].toarray()[0]
        reference_vectors = tfidf_matrix[:-1].toarray()
        reference_mean_vector = np.mean(reference_vectors, axis=0)
        cosine_similarity = np.dot(generated_vector, reference_mean_vector) / (np.linalg.norm(generated_vector) * np.linalg.norm(reference_mean_vector))
        informativeness_score = 1 - cosine_similarity
        return informativeness_score

    def record_rouge(self, rouge_score, results_df, index):
        """
        Record the ROUGE scores in the results dataframe.
        
        Parameters:
        rouge_score (dict): The ROUGE score dictionary.
        results_df (pd.DataFrame): The results dataframe.
        index (int): The index of the current row in the dataframe.
        
        Returns:
        pd.DataFrame: Updated results dataframe.
        """
        results_df.loc[index, 'rouge-1_f'] = rouge_score[0]["rouge-1"]['f']
        results_df.loc[index, 'rouge-1_p'] = rouge_score[0]["rouge-1"]['p']
        results_df.loc[index, 'rouge-1_r'] = rouge_score[0]["rouge-1"]['r']
        results_df.loc[index, 'rouge-2_f'] = rouge_score[0]["rouge-2"]['f']
        results_df.loc[index, 'rouge-2_p'] = rouge_score[0]["rouge-2"]['p']
        results_df.loc[index, 'rouge-2_r'] = rouge_score[0]["rouge-2"]['r']
        results_df.loc[index, 'rouge-l_f'] = rouge_score[0]["rouge-l"]['f']
        results_df.loc[index, 'rouge-l_p'] = rouge_score[0]["rouge-l"]['p']
        results_df.loc[index, 'rouge-l_r'] = rouge_score[0]["rouge-l"]['r']
        return results_df

    def cosine_similarity(self, vec1, vec2):
        """
        Calculate the cosine similarity between two vectors.
        
        Parameters:
        vec1 (array-like): The first vector.
        vec2 (array-like): The second vector.
        
        Returns:
        float: Cosine similarity.
        """
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)
        return cosine_sim


### zero-shot spurious mitigation

# loop over tokens in the sentence
# mask/delete (all) the token and get new sentence
# get the prediction confidence of the new sentence
# if new pred != old pred:
#     search the nearest k neighbors of the new sentence in the test set
#     if label distribution of neighbors are not even, then we consider the token as spurious
#     solving the spurious correlation by taking the new prediction.

#TODO: variation: token embedding not using the last layer, but the embedding layer.


import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig
from collections import Counter
import time
from functools import lru_cache
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from datetime import datetime
import re

import random
seed = 42
random.seed(seed)

import faiss

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu or pip install faiss-gpu")


class SpuriousMitigation:
    """
    spurious correlation mitigation with GPU acceleration and FAISS.
    """
    
    def __init__(self, device: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer,
                 test_sentences: List[str], test_labels: List[int] = None,         
                 k_neighbors: int = 5, eveness_threshold: float = 0.7,
                 use_cache: bool = True, batch_size: int = 8,
                 use_roberta_base_for_embeddings: bool = False,
                 use_faiss_index: bool = True,
                 use_ivf: bool = False, nlist: int = 100, nprobe: int = 10,
                 use_ivfpq: bool = False, nbits: int = 8, m: int = 16,
                 use_hnsw: bool = False, hnsw_m: int = 16, hnsw_ef_construction: int = 100, hnsw_ef: int = 100):
        """
        Initialize the spurious mitigation system.
        
        Args:
            test_sentences: List of test sentences for neighbor search
            test_labels: Corresponding labels for test sentences
            test_predictions: Corresponding predictions for test sentences
            k_neighbors: Number of neighbors to consider for label distribution analysis
            eveness_threshold: Threshold for considering prediction changes significant
            use_cache: Whether to use caching for neighbor search
            batch_size: Batch size for model inference
            use_gpu: Whether to use GPU acceleration
            checkpoint_path: Path to RoBERTa fine-tuned model checkpoint
            use_ivf: Whether to use IVF index for FAISS (default False)
            nlist: Number of clusters for IVF (default 100)
            use_ivfpq: Whether to use IVFPQ index for FAISS (default False)
            nbits: Number of bits per sub-vector for PQ (default 8)
            use_hnsw: Whether to use HNSW index for FAISS (default False)
            hnsw_m: HNSW M parameter (default 16)
            hnsw_ef_construction: HNSW efConstruction parameter (default 100)
            hnsw_ef: HNSW efSearch parameter (default 100)
        """
        
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        print(f"Using device: {self.device}")

        self.use_roberta_base_for_embeddings = use_roberta_base_for_embeddings
        if self.use_roberta_base_for_embeddings:
            print("Loading base RoBERTa model")
            self.roberta_base_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.roberta_base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
            self.roberta_base_model.to(self.device)
            self.roberta_base_model.eval()
        
        self.test_sentences = test_sentences
        self.test_labels = test_labels
        self.k_neighbors = k_neighbors
        self.eveness_threshold = eveness_threshold
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_ivfpq = use_ivfpq
        self.nbits = nbits
        self.use_hnsw = use_hnsw
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef = hnsw_ef
        
        # Define pronouns to exclude from masking and grouping
        self.excluded_pronouns = {
            'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'theirs',
            'he', 'him', 'his', 'she', 'her', 'hers', 'we', 'us', 'our', 'ours',
            'you', 'your', 'yours', 'i', 'me', 'my', 'mine', 'myself', 'yourself',
            'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves',
            'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'none', 'neither', 'either', 'every', 'everyone', 'everything',
            'somebody', 'someone', 'something', 'nobody', 'noone', 'nothing',
            'anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything'
        }
        
        # Initialize neighbor search: always use checkpoint model embeddings
        self._initialize_neighbor_search()
        
        # Calculate all test set predictions
        print("Getting test set predictions...")
        self.test_predictions = self.get_batch_predictions(self.test_sentences)
    
    def _initialize_neighbor_search(self):
        """
        Initialize neighbor search: always use checkpoint model embeddings, 
        but choose between FAISS or sklearn for nearest neighbor search.
        """
        print("Setting up RoBERTa embeddings for semantic neighbor search...")
        # Always use checkpoint model for embeddings
        self.test_embeddings = self._get_roberta_embeddings(self.test_sentences)
        
        if FAISS_AVAILABLE:
            print("Using FAISS for fast nearest neighbor search...")
            self._setup_faiss_search()
        else:
            print("FAISS not available, falling back to sklearn for nearest neighbor search")
            self._setup_sklearn_search()
    
    def _setup_faiss_search(self):
        """
        Setup FAISS index for fast nearest neighbor search with RoBERTa embeddings.
        
        Strategy: Use L2-normalized embeddings with IndexFlatIP for cosine similarity.
        - IndexFlatIP computes inner product: <a, b>
        - When vectors are L2-normalized: <a, b> = ||a|| * ||b|| * cos(θ) = cos(θ)
        - This gives us cosine similarity directly

        Priority: HNSW > IVFPQ > IVF > Flat  TODO:
        """
        # Normalize RoBERTa embeddings for cosine similarity
        embeddings_normalized = self.test_embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        dimension = embeddings_normalized.shape[1]
        try:
            print("Trying to use GPU FAISS index...")
            res = faiss.StandardGpuResources()
            if self.use_hnsw:
                print(f"Using HNSW index (M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}, efSearch={self.hnsw_ef})...")
                cpu_index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
                cpu_index.hnsw.efConstruction = self.hnsw_ef_construction
                cpu_index.hnsw.efSearch = self.hnsw_ef
                cpu_index.add(embeddings_normalized)
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                print(f"GPU FAISS HNSW index created with {len(self.test_sentences)} vectors of dim {dimension}")
            elif self.use_ivfpq:
                print(f"Using IVFPQ index (nlist={self.nlist}, nbits={self.nbits})...")
                quantizer = faiss.IndexFlatIP(dimension)
                cpu_index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, self.m, self.nbits)
                cpu_index.train(embeddings_normalized)
                cpu_index.add(embeddings_normalized)
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self.faiss_index.nprobe = self.nprobe
                print(f"GPU FAISS IVFPQ index created with {len(self.test_sentences)} vectors of dim {dimension}")
            elif self.use_ivf:
                print(f"Using IVF index (nlist={self.nlist})...")
                quantizer = faiss.IndexFlatIP(dimension)
                cpu_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                cpu_index.train(embeddings_normalized)  
                cpu_index.add(embeddings_normalized)
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self.faiss_index.nprobe = self.nprobe
                print(f"GPU FAISS IVF index created with {len(self.test_sentences)} vectors of dim {dimension}")
            else: 
                print("Using FlatIP index...")
                cpu_index = faiss.IndexFlatIP(dimension)
                cpu_index.add(embeddings_normalized)
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                print(f"GPU FAISS FlatIP index created with {len(self.test_sentences)} vectors of dim {dimension}")
        except Exception as e:
            print(f"[Warning] GPU FAISS failed: {e}. Falling back to CPU FAISS.")
            if self.use_hnsw:
                print(f"Using CPU HNSW index (M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}, efSearch={self.hnsw_ef})...")
                cpu_index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
                cpu_index.hnsw.efConstruction = self.hnsw_ef_construction
                cpu_index.hnsw.efSearch = self.hnsw_ef
                cpu_index.add(embeddings_normalized)
                self.faiss_index = cpu_index
                print(f"CPU FAISS HNSW index created with {len(self.test_sentences)} vectors of dim {dimension}")
            elif self.use_ivfpq:
                print(f"Using CPU IVFPQ index (nlist={self.nlist}, nbits={self.nbits})...")
                quantizer = faiss.IndexFlatIP(dimension)
                m = 16
                cpu_index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, m, self.nbits)
                cpu_index.train(embeddings_normalized)
                cpu_index.add(embeddings_normalized)
                self.faiss_index = cpu_index
                self.faiss_index.nprobe = self.nprobe
                print(f"CPU FAISS IVFPQ index created with {len(self.test_sentences)} vectors of dim {dimension}")
            elif self.use_ivf:
                print(f"Using CPU IVF index (nlist={self.nlist})...")
                quantizer = faiss.IndexFlatIP(dimension)
                cpu_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                cpu_index.train(embeddings_normalized)
                cpu_index.add(embeddings_normalized)
                self.faiss_index = cpu_index
                self.faiss_index.nprobe = self.nprobe
                print(f"CPU FAISS IVF index created with {len(self.test_sentences)} vectors of dim {dimension}")
            else:
                print("Using CPU FlatIP index...")
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(embeddings_normalized)
                print(f"CPU FAISS FlatIP index created with {len(self.test_sentences)} vectors of dim {dimension}")
        print("Note: Using L2-normalized embeddings with FlatIP, IVFFlat, IVFPQ, or HNSW for cosine similarity")
    
    def _setup_sklearn_search(self):
        """Setup sklearn for nearest neighbor search with RoBERTa embeddings."""
        self.neighbor_model = NearestNeighbors(n_neighbors=self.k_neighbors, metric='cosine')
        self.neighbor_model.fit(self.test_embeddings)
        print(f"Sklearn index created with {len(self.test_sentences)} RoBERTa vectors")
    
    def _get_roberta_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Get RoBERTa embeddings for a list of sentences.
        
        Args:
            sentences: List of sentences to embed
            
        Returns:
            numpy array of embeddings
        """
        embeddings = []

        if self.use_roberta_base_for_embeddings:
            model = self.roberta_base_model
        else:
            model = self.model
        
        # Process in batches
        # for i in tqdm(range(0, len(sentences), self.batch_size), desc="Getting RoBERTa embeddings"):
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            
            inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Use the last hidden state and take mean pooling
                last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
                
                # Mean pooling (excluding padding tokens)
                attention_mask = inputs['attention_mask']
                masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
                summed = torch.sum(masked_hidden, dim=1)
                counts = torch.sum(attention_mask, dim=1, keepdim=True)
                mean_pooled = summed / counts
                
                embeddings.append(mean_pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_batch_predictions(self, sentences: List[str]) -> List[Tuple[int, float]]:
        """
        Get predictions for multiple sentences in batch with GPU acceleration.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            List of (predicted_label, confidence_score) tuples
        """

        # in batch
        if not sentences:
            return []
        
        predictions = []
        # for i in tqdm(range(0, len(sentences), self.batch_size), desc="Getting predictions"):
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1).cpu().tolist()
                confidences = torch.max(probabilities, dim=1).values.cpu().tolist()
                predictions.extend(list(zip(predicted_labels, confidences)))
        return predictions
        
        # # Move inputs to GPU
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     probabilities = torch.softmax(outputs.logits, dim=1)
        #     predicted_labels = torch.argmax(probabilities, dim=1).cpu().tolist()
        #     confidences = torch.max(probabilities, dim=1).values.cpu().tolist()
        
        # return list(zip(predicted_labels, confidences))
    
    def _search_neighbors(self, sentences: List[str], original_sentences: List[str]) -> List[List[int]]:
        """
        sentences: list of sentences to search neighbors for
        original_sentences: list of original sentences
        Returns: list of neighbor indices for each sentence
        """
        sentence_embeddings = self._get_roberta_embeddings(sentences)  # shape: (batch, dim)
        if FAISS_AVAILABLE:
            sentence_embeddings_normalized = sentence_embeddings.copy()
            faiss.normalize_L2(sentence_embeddings_normalized)
            distances, indices = self.faiss_index.search(sentence_embeddings_normalized, self.k_neighbors+1)   
        else:
            distances, indices = self.neighbor_model.kneighbors(sentence_embeddings, n_neighbors=self.k_neighbors+1)

        # exclude the original sentence itself
        neighbor_indices = []
        for i in range(len(sentences)):
            if original_sentences[i] in self.test_sentences:
                self_idx = self.test_sentences.index(original_sentences[i])
                idxs = [int(j) for j in indices[i] if int(j) != self_idx]
            else:
                idxs = [int(j) for j in indices[i]]
            neighbor_indices.append(idxs[:self.k_neighbors])
        return neighbor_indices
    
    def get_neighbor_label_distribution(self, sentence_dicts: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Get label distribution of k-nearest neighbors using RoBERTa embeddings.
        
        Args:
            sentence_dict_batch: list of sentence dictionaries
        Returns:
            Dictionary mapping label to count
        """
        # sentence = sentence_dict['sentence']
        # original_sentence = sentence_dict['original_sentence']
        
        # neighbor_indices = self._search_neighbors(sentence, original_sentence)  
        # neighbor_sentences = [self.test_sentences[int(i)] for i in neighbor_indices]
        # neighbor_predictions = [self.test_predictions[i][0] for i in neighbor_indices]
        
        # label_counts = Counter(neighbor_predictions)
        # result = dict(label_counts) # key: label, value: count


        # sentences_batch = [sentence_dict['sentence'] for sentence_dict in sentence_dict_batch]
        # original_sentences_batch = [sentence_dict['original_sentence'] for sentence_dict in sentence_dict_batch]

        # neighbor_indices_batch = self._search_neighbors(sentences_batch, original_sentences_batch)
        # neighbor_predictions_batch = [
        #     [self.test_predictions[j][0] for j in neighbor_indices]
        #     for neighbor_indices in neighbor_indices_batch
        # ]

        # label_counts_batch = [Counter(neighbor_predictions) for neighbor_predictions in neighbor_predictions_batch]
        # result_batch = [dict(label_counts) for label_counts in label_counts_batch]

        # in batch pass into _search_neighbors by self.batch_size
        result_batch = []
        for i in range(0, len(sentence_dicts), self.batch_size):
            sentence_dicts_batch = sentence_dicts[i:i+self.batch_size]
            sentences_batch = [sentence_dict['sentence'] for sentence_dict in sentence_dicts_batch]
            original_sentences_batch = [sentence_dict['original_sentence'] for sentence_dict in sentence_dicts_batch]
            neighbor_indices_batch = self._search_neighbors(sentences_batch, original_sentences_batch)  
            neighbor_predictions_batch = [
                [self.test_predictions[j][0] for j in neighbor_indices]
                for neighbor_indices in neighbor_indices_batch
            ]
            label_counts_batch = [dict(Counter(neighbor_predictions)) for neighbor_predictions in neighbor_predictions_batch]
            result_batch.extend(label_counts_batch)
        
        return result_batch #List[Dict[int, int]]
    
    def is_label_distribution_skewed(self, label_distribution: Dict[int, int]) -> bool:
        """
        Check if label distribution is skewed (indicating potential spurious correlation).
        
        Args:
            label_distribution: Dictionary mapping label to count
            
        Returns:
            True if distribution is skewed, False otherwise
        """
        if len(label_distribution) <= 1:
            return True  # If there's only one label, it's skewed
        
        total_count = sum(label_distribution.values())
        max_count = max(label_distribution.values())
        
        return max_count / total_count > self.eveness_threshold
    
    def generate_modified_sentences(self, sentence: str) -> List[Tuple[str, int, str]]:
        """
        Generate modified sentences by masking tokens at the input_ids level.
        This ensures consistent tokenization and avoids issues with different tokenizations.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of (modified_sentence, token_index, modification_type) tuples
        """

        sentence = ' '.join(sentence.split())
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'][0]  # Remove batch dimension
        
        # Get token positions (skip special tokens like [CLS], [SEP] and pronouns)
        token_positions = []
        token_texts = []
        token_ids = []  # Store the actual input_ids for each token
        for i, token_id in enumerate(input_ids):
            if token_id not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                token_text = self.tokenizer.decode([token_id])
            
                # skip pronouns, pure punctuation, and empty tokens
                if (token_text.strip().lower() not in self.excluded_pronouns and 
                    not re.match(r'^[^\w\s]+$', token_text) and
                    token_text.strip()):
                    token_positions.append(i)
                    token_texts.append(token_text)
                    token_ids.append(token_id.item())  

                
        
        modified_sentences = []
        
        mask_position_list = []
        for i, token_idx in enumerate(token_positions): 
            
            masked_input_ids = input_ids.clone()
            
            token_id_to_mask = [token_ids[i]] 

            # ====== Find tokens with same word root ======
            current_token = token_texts[i].strip()  # Remove leading/trailing spaces for comparison
            current_lemma = get_lemma(current_token)
            
            # Check for tokens that contain the current token or its lemma as substring
            for j, other_token in enumerate(token_texts):
                if j == i: continue
                other_token_clean = other_token.strip()
                other_lemma = get_lemma(other_token_clean)

                # Match if:
                if current_lemma in other_lemma:
                    token_id_to_mask.append(token_ids[j])
        
            mask_positions = [j for j, id in enumerate(input_ids) if id in token_id_to_mask]
            if sorted(mask_positions) in mask_position_list:
                continue
            mask_position_list.append(sorted(mask_positions))

            for pos in mask_positions:
                masked_input_ids[pos] = self.tokenizer.mask_token_id
        
            # Decode back to text
            masked_sentence = self.tokenizer.decode(masked_input_ids, skip_special_tokens=True)
            
            # Get the original token text for reference
            original_token = self.tokenizer.decode([input_ids[token_idx]])
            
            modified_sentences.append((masked_sentence, token_idx, "masked", original_token, current_lemma))
        
        return modified_sentences
    
    def mitigate_spurious_correlations_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        batch process a list of sentences for spurious mitigation.
        Args:
            indices: list of sentence indices to process
        Returns:
            list of mitigation results for each sentence
        """
        batch_sentences = [self.test_sentences[i] for i in indices]
        batch_original_predictions = [self.test_predictions[i] for i in indices]

        all_modified_sentences = []  # List[List[Tuple[str, ...]]]
        all_modified_texts = []      # List[str]
        modified_map = []            # record the start and end of each sentence's modified_sentences in all_modified_texts
        for sent in batch_sentences:
            modified = self.generate_modified_sentences(sent)
            all_modified_sentences.append(modified)
            modified_map.append((len(all_modified_texts), len(all_modified_texts) + len(modified)))
            all_modified_texts.extend([m[0] for m in modified])

        all_predictions = self.get_batch_predictions(all_modified_texts) # List[Tuple[int, float]]

        all_sentence_dicts = []  # List[Dict[str, Any]], length = len(all_modified_texts)
        for idx_in_batch, idx in enumerate(indices):  
            sentence = self.test_sentences[idx]
            original_prediction, original_confidence = batch_original_predictions[idx_in_batch]
            modified_sentences = all_modified_sentences[idx_in_batch]
            for i, modified_sentence_data in enumerate(modified_sentences):
                pred, conf = all_predictions[modified_map[idx_in_batch][0] + i]
                modified_sentence_dict = {
                    'sentence': modified_sentence_data[0],
                    'position': modified_sentence_data[1],
                    'modification_type': modified_sentence_data[2],
                    'token': modified_sentence_data[3],
                    'token_lemma': modified_sentence_data[4],
                    'original_sentence': sentence,
                    'original_prediction': original_prediction,
                    'original_confidence': original_confidence,
                    'new_prediction': pred,
                    'new_confidence': conf,
                    'prediction_changed': pred != original_prediction,
                }
                all_sentence_dicts.append(modified_sentence_dict)

        all_neighbor_dists = self.get_neighbor_label_distribution(all_sentence_dicts)

        results = []
        ptr = 0
        for idx_in_batch, idx in enumerate(indices):
            start, end = modified_map[idx_in_batch]
            modified_sentences = all_modified_sentences[idx_in_batch]
            predictions = all_predictions[start:end]
            original_prediction, original_confidence = batch_original_predictions[idx_in_batch]
            sentence = self.test_sentences[idx]

            spurious_tokens = []
            mitigated_predictions = []
            all_token_analyses = []

            for i, modified_sentence_data in enumerate(modified_sentences):
                pred, conf = predictions[i]
                modified_sentence_dict = all_sentence_dicts[ptr]
                neighbor_dist = all_neighbor_dists[ptr]
                ptr += 1
                token_analysis = modified_sentence_dict.copy()
                token_analysis['neighbor_distribution'] = neighbor_dist
                all_token_analyses.append(token_analysis)
                if pred != original_prediction:
                    if self.is_label_distribution_skewed(neighbor_dist):
                        spurious_tokens.append(token_analysis)
                        majority_pred = Counter(neighbor_dist).most_common(1)[0][0]
                        mitigated_predictions.append(majority_pred)

            # Determine final prediction
            if mitigated_predictions:
                if mitigated_predictions != original_prediction:
                    final_prediction = Counter(mitigated_predictions).most_common(1)[0][0]
                else:
                    final_prediction = original_prediction
            else:
                final_prediction = original_prediction

            result = {
                'original_sentence': sentence,
                'original_prediction': original_prediction,
                'original_confidence': original_confidence,
                'spurious_tokens': spurious_tokens,
                'mitigated_prediction': mitigated_predictions,
                'final_prediction': final_prediction,
                'num_spurious_tokens': len(spurious_tokens),
            }
            results.append(result)
        return results
    
    def find_semantically_similar_tokens(self, target_token: str, top_k: int = 5, similarity_threshold: float = 0.99) -> List[Tuple[str, float]]:
        """
        Find semantically similar tokens in the vocabulary using model embeddings.
        
        Args:
            target_token: The token to find similar tokens for
            top_k: Number of similar tokens to return
            similarity_threshold: Minimum similarity threshold 
            
        Returns:
            List of (token, similarity_score) tuples
        """
        # Get embedding for the target token
        target_embedding = self._get_token_embedding(target_token)
        
        if target_embedding is None:
            return []
        
        # Get embeddings for all tokens in vocabulary
        vocab_embeddings = self._get_vocab_embeddings()
        
        # Calculate similarities
        similarities = []
        for token, embedding in vocab_embeddings.items():
            if token != target_token:  # Exclude the target token itself
                similarity = self._cosine_similarity(target_embedding, embedding)
                # Only include tokens with similarity above threshold
                if similarity > similarity_threshold:
                    similarities.append((token, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _get_token_embedding(self, token: str) -> np.ndarray:
        """
        Get embedding for a single token.
        
        Args:
            token: The token to get embedding for
            
        Returns:
            Token embedding as numpy array
        """
        try:
            # Create a simple sentence with just the token
            sentence = f" {token} "  # Add spaces to ensure proper tokenization
            
            inputs = self.tokenizer(
                sentence, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get the last hidden state
                last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
                
                # Find the position of our target token
                input_ids = inputs['input_ids'][0]
                token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                
                # Find the token position (skip special tokens)
                token_positions = []
                for i, tid in enumerate(input_ids):
                    if tid not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                        token_positions.append(i)
                
                if not token_positions:
                    return None
                
                # Use the first non-special token position
                token_position = token_positions[0]
                token_embedding = last_hidden_state[0, token_position].cpu().numpy()
                
                return token_embedding
                
        except Exception as e:
            print(f"Error getting embedding for token '{token}': {e}")
            return None
    
    def _get_vocab_embeddings(self, max_vocab_size: int = 10000) -> Dict[str, np.ndarray]:
        """
        Get embeddings for a subset of vocabulary tokens.
        
        Args:
            max_vocab_size: Maximum number of vocabulary tokens to process
            
        Returns:
            Dictionary mapping token to embedding
        """
        # Cache vocab embeddings to avoid recomputing
        if hasattr(self, '_vocab_embeddings_cache'):
            return self._vocab_embeddings_cache
        
        print("Computing vocabulary embeddings (this may take a while)...")
        
        vocab_embeddings = {}
        
        # Get a subset of vocabulary tokens (most common ones)
        vocab_items = list(self.tokenizer.get_vocab().items())
        vocab_items.sort(key=lambda x: x[1])  # Sort by token ID
        
        # Take the first max_vocab_size tokens
        selected_tokens = vocab_items[:max_vocab_size]
        
        # Process tokens in batches
        for i in tqdm(range(0, len(selected_tokens), self.batch_size), desc="Computing vocab embeddings"):
            batch_tokens = selected_tokens[i:i + self.batch_size]
            
            for token, token_id in batch_tokens:
                # Skip special tokens
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, 
                           self.tokenizer.pad_token, self.tokenizer.mask_token]:
                    continue
                
                # Get embedding for this token
                embedding = self._get_token_embedding(token)
                if embedding is not None:
                    vocab_embeddings[token] = embedding
        
        # Cache the results
        self._vocab_embeddings_cache = vocab_embeddings
        print(f"Computed embeddings for {len(vocab_embeddings)} vocabulary tokens")
        
        return vocab_embeddings
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_tokens_in_sentence(self, sentence: str, target_token: str, top_k: int = 5, similarity_threshold: float = 0.99) -> List[Tuple[str, float, int]]:
        """
        Find tokens in a sentence that are semantically similar to a target token.
        
        Args:
            sentence: The sentence to search in
            target_token: The token to find similar tokens for
            top_k: Number of similar tokens to return
            similarity_threshold: Minimum similarity threshold (default: 0.99)
            
        Returns:
            List of (token, similarity_score, position) tuples
        """
        # Get target token embedding
        target_embedding = self._get_token_embedding(target_token)
        
        if target_embedding is None:
            return []
        
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'][0]
        
        # Get embeddings for all tokens in the sentence
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
        
        # Calculate similarities for each token
        similarities = []
        for i, token_id in enumerate(input_ids):
            # Skip special tokens
            if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                continue
            
            # Get token text
            token_text = self.tokenizer.decode([token_id])
            
            # Skip if it's the target token itself
            if token_text == target_token:
                continue
            
            # Get token embedding
            token_embedding = last_hidden_state[i].cpu().numpy()
            
            # Calculate similarity
            similarity = self._cosine_similarity(target_embedding, token_embedding)
            
            # Only include tokens with similarity above threshold
            if similarity > similarity_threshold:
                similarities.append((token_text, similarity, i))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _remove_token_from_sentence(self, sentence: str, token: str) -> str:
        """
        Remove the first occurrence of token from the sentence (simple whitespace split-join).
        """
        words = sentence.split()
        for i, w in enumerate(words):
            if w.strip() == token.strip():
                words.pop(i)
                break
        return ' '.join(words)

def get_lemma(token):
    """Get the base form of a token using NLTK lemmatizer"""
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        # Try different parts of speech
        for pos in ['n', 'v', 'a', 'r']:  # noun, verb, adjective, adverb
            lemma = lemmatizer.lemmatize(token.lower(), pos=pos)
            if lemma != token.lower():  # If we found a different lemma, use it
                return lemma
        return token.lower()  # If no different lemma found, return original
    except ImportError:
        print("Warning: NLTK not installed. Install with: pip install nltk")
        print("Then download WordNet: python -c 'import nltk; nltk.download(\"wordnet\")'")
        return token.lower()
    except Exception:
        return token.lower()
                
def load_test_data(test_data_path):
    test_sentences = []
    test_labels = []
    df = pd.read_csv(test_data_path) # column 1 is sentence, column 2 is label
    for index, row in df.iterrows():
        test_sentences.append(row['sentence'])
        test_labels.append(row['label'])

    return test_sentences, test_labels

def load_model(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer

def mitigate_from_file(test_data_path, checkpoint_path):
    test_sentences, test_labels = load_test_data(test_data_path)
    test_sentences = test_sentences
    test_labels = test_labels

    print(f"Number of test sentences: {len(test_sentences)}")

    model, tokenizer = load_model(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    
    batch_size = 512
    print(f"Batch size: {batch_size}")
    mitigator = SpuriousMitigation(
        device=device,
        model=model,
        tokenizer=tokenizer,
        test_sentences=test_sentences,
        batch_size=batch_size, 
        k_neighbors=20,
        use_roberta_base_for_embeddings=False,
        use_faiss_index=True,
        use_hnsw=True,
    )

    print("Starting to mitigate spurious correlations...")
    results = []
    total_num = len(test_sentences)
    for i in tqdm(range(0, total_num, batch_size), desc="Mitigating spurious correlations"):
        batch_indices = list(range(i, min(i + batch_size, total_num)))
        batch_results = mitigator.mitigate_spurious_correlations_batch(batch_indices)
        results.extend(batch_results)

    return results

def mitigation_results_evaluation(results, data_path):
    print(f"Number of results: {len(results)}")
    new_predictions = [result['final_prediction'] for result in results]
    if 'ground_truth' in results[0]:
        ground_truths = [result['ground_truth'] if 'ground_truth' in result else result['original_prediction'] for result in results]
    else:
        df = pd.read_csv(data_path)
        ground_truths = [df.iloc[i]['label'] for i in range(len(df))]

    # Calculate accuracy
    correct = sum(1 for new, truth in zip(new_predictions, ground_truths) if new == truth)
    accuracy = correct / len(new_predictions)
    print(f"Accuracy: {accuracy}")
    
    # robust accuracy from the filtered data
    filtered_indices = extract_filtered_data(results)
    print(f"Number of filtered sentences: {len(filtered_indices)}")
    filtered_results = [results[i] for i in filtered_indices]
    filtered_new_predictions = [result['final_prediction'] for result in filtered_results]
    filtered_ground_truths = [ground_truths[i] for i in filtered_indices]
    filtered_accuracy = sum(1 for new, truth in zip(filtered_new_predictions, filtered_ground_truths) if new == truth) / len(filtered_new_predictions)
    print(f"Filtered accuracy: {filtered_accuracy}")

    # normal accuracy
    normal_indices = extract_normal_data(results)
    print(f"Number of normal sentences: {len(normal_indices)}")
    normal_results = [results[i] for i in normal_indices]
    normal_predictions = [result['final_prediction'] for result in normal_results]
    normal_ground_truths = [ground_truths[i] for i in normal_indices]
    normal_accuracy = sum(1 for new, truth in zip(normal_predictions, normal_ground_truths) if new == truth) / len(normal_predictions)
    print(f"Normal accuracy: {normal_accuracy}")

    # original accuracy
    # compare the original prediction and the ground truth
    original_predictions = [result['original_prediction'] for result in results]
    filtered_original_predictions = [result['original_prediction'] for result in filtered_results]
    original_accuracy = sum(1 for new, truth in zip(original_predictions, ground_truths) if new == truth) / len(original_predictions)
    filtered_original_accuracy = sum(1 for new, truth in zip(filtered_original_predictions, filtered_ground_truths) if new == truth) / len(filtered_original_predictions)
    # print("\n")
    print("---original results---")
    print(f"Original accuracy: {original_accuracy}")
    print(f"Original filtered  accuracy: {filtered_original_accuracy}")

    # original normal accuracy
    original_normal_predictions = [result['original_prediction'] for result in normal_results]
    original_normal_ground_truths = [ground_truths[i] for i in normal_indices]
    original_normal_accuracy = sum(1 for new, truth in zip(original_normal_predictions, original_normal_ground_truths) if new == truth) / len(original_normal_predictions)
    print(f"Original normal accuracy: {original_normal_accuracy}")

    # filtered normal accuracy

def extract_filtered_data(results):
    """
    return the index of the sentences that are contrain the word "book" or "movie"
    """
    output_list = []
    for i, result in enumerate(results):
        sentence = result['original_sentence'].lower()
        has_book = "book" in sentence
        has_movie = "movie" in sentence
        if has_book != has_movie:  
            output_list.append(i)
    return output_list

def extract_normal_data(results):
    """
    return the index of the sentences that do not contain the word "book" or "movie"
    """
    output_list = []
    for i, result in enumerate(results):
        sentence = result['original_sentence'].lower()
        has_book = "book" in sentence
        has_movie = "movie" in sentence
        if not has_book and not has_movie:
            output_list.append(i)
    return output_list



if __name__ == "__main__":
    test_data_path="/mnt/disk21/user/jiayili/doNt-Forget-your-Language/data/test_downsampled/unbiased_amazon_test.csv"
    checkpoint_path = "/mnt/disk21/user/jiayili/peer_learning/results/biased_amazon_output/checkpoint-5835"

    results = mitigate_from_file(test_data_path, checkpoint_path)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/mitigation'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'{save_path}/{current_time}.json'
    with open(file_name, 'w') as f:
        json.dump(results, f)
    print("Saved results to ", file_name)
    mitigation_results_evaluation(results, test_data_path)
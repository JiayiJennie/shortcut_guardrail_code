from numpy.ma import masked
from utils.model_utils import TokenScore
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from utils.stop_tokens import EXCLUDED_TOKENS
import re
from captum.attr import IntegratedGradients, Saliency
import torch.nn.functional as F
from collections import defaultdict
# from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter
# from transformers import pipeline
import nltk
from nltk.corpus import wordnet as wn
import spacy
from lemminflect import getInflection
import requests
from sentence_transformers import SentenceTransformer
import copy
import random

random.seed(42)


try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu or pip install faiss-gpu")

nlp = spacy.load("en_core_web_sm")

def get_lemma_spacy(word: str):
    doc = nlp(word)
    return doc[0].lemma_

def get_batch_predictions(sentences: List[str], model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, 
                        device: str, batch_size: int, _tqdm: bool = True) -> List[Tuple[int, float]]:
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

        model.eval()
        iterator = range(0, len(sentences), batch_size)
        if _tqdm:
            iterator = tqdm(iterator, desc="Getting predictions", total=len(sentences) // batch_size + 1)

        with torch.no_grad():
            for i in iterator:
                batch_sentences = sentences[i:i + batch_size]
                inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1).cpu().tolist()
                confidences = torch.max(probabilities, dim=1).values.cpu().tolist()
                predictions.extend(list(zip(predicted_labels, confidences)))
        return predictions

        
class ShortcutTokenFinder:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification, 
    batch_size: int = 8, _tqdm: bool = True, use_ig: bool = False, use_saliency: bool = False, ig_n_steps: int = 5,
    mask_sensitivity_threshold: float = 0.1, use_bert_base_for_embeddings: bool = False, k_neighbors: int = 5,
    use_hnsw: bool = False, hnsw_m: int = 16, hnsw_ef_construction: int = 100, hnsw_ef: int = 100,
    use_ivfpq: bool = False, nlist: int = 100, m: int = 16, nbits: int = 8, nprobe: int = 10,
    use_ivf: bool = False, _lambda: float = 0.5,
    majority_label_percentage_threshold: float = 0.8,
     consistency_ratio_threshold: float = 0.95,
     min_num_flips: int = 2,
     min_prevalence: float = 0.01,
     use_excluded_tokens: bool = True,
     whitelist_tokens: set | None = None):
        self.tokenizer = tokenizer
        self.model = model
        # Number of labels/classes. Used by some heuristics; default to 2 if not present.
        try:
            self.num_labels = int(getattr(getattr(model, "config", None), "num_labels", 2) or 2)
        except Exception:
            self.num_labels = 2
        self.batch_size = batch_size
        self._tqdm = _tqdm
        self.use_ig = use_ig
        self.use_saliency = use_saliency
        self.ig_n_steps = ig_n_steps
        self.mask_sensitivity_threshold = mask_sensitivity_threshold
        # Stage 2 thresholds (configurable via __init__, like mask_sensitivity_threshold)
        self.majority_label_percentage_threshold = float(majority_label_percentage_threshold) / max(1, (self.num_labels - 1))
        self.consistency_ratio_threshold = float(consistency_ratio_threshold) / max(1, (self.num_labels - 1))
        # Stage 2: minimum number of label changes required when ablating a token.
        # Historically this was `num_flips > 1` => at least 2 flips.
        self.min_num_flips = int(min_num_flips)
        # Stage 2: minimum fraction of sentences for which we can actually ablate the token.
        # Historically hard-coded to prevalence > 0.01.
        self.min_prevalence = float(min_prevalence)
        # Stage 1 token filtering: if False, do NOT filter tokens by utils.stop_tokens.EXCLUDED_TOKENS.
        self.use_excluded_tokens = bool(use_excluded_tokens)
        self.whitelist_tokens = whitelist_tokens or set()
        self.use_bert_base_for_embeddings = use_bert_base_for_embeddings
        self.k_neighbors = k_neighbors
        self.device = model.device
        self.use_hnsw = use_hnsw
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef = hnsw_ef
        self.use_ivfpq = use_ivfpq
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self._lambda = _lambda
        self.bert_base_model = None
    
    def compute_entropy(self, token_scores: torch.Tensor, normalization='softmax'):
        """
        Input: token_scores (batch, seq_len)
        normalize: 'softmax' or 'minmax'
        Output: entropy (batch)
        """
        if normalization == 'softmax':
            p = torch.softmax(token_scores, dim=-1)
        elif normalization == 'minmax':
            x = token_scores - token_scores.min(dim=-1, keepdim=True).values
            p = x/(x.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            raise ValueError("normalize must be 'softmax' or 'minmax'")
        
        entropy = p / torch.log(torch.tensor(p.shape[-1], device = token_scores.device))  #normalize to [0, 1]
        return entropy
    
    def compute_top_k_mass(self, token_scores: torch.Tensor, top_k_token: list[int], normalization='softmax'):
        """
        token_scores: (batch, seq_len)
        normalization: 'softmax' or 'minmax'
        return: top-k mass (batch,)
        """
        if normalization == 'softmax':
            p = torch.softmax(token_scores, dim=-1)
        elif normalization == 'minmax':
            p = token_scores - token_scores.min(dim=-1, keepdim=True).values
            p = p/(p.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            raise ValueError("normalization must be 'softmax' or 'minmax'")
        topk_vals = [torch.topk(p, k, dim=-1).values.sum(dim=-1) for k in top_k_token]
        return topk_vals # list of (batch,)
    
    def stage1_find_important_tokens(self, sentences: list[str], top_k_token: int = 10, return_scores: bool = False,
        return_entropy_only: bool = False):
        """Stage 1: Find high attribution tokens based on IG, Saliency, or Attn scores

        Args:
            sentences: list of input texts
            top_k_token: number of top tokens to return per sentence
            return_scores: if True, return list[list[{"token": str, "score": float}]] instead of list[list[str]]
        """
        self.sentences = sentences
        self.top_k = top_k_token
        
        if self.use_ig:
            token_scores = self.get_ig_scores(sentences)
        elif self.use_saliency:
            token_scores = self.get_saliency_scores(sentences)
        else:   
            token_scores = self.get_attn_scores(sentences)  
        entropy = self.compute_entropy(token_scores, normalization='minmax').mean()
        top_k_mass = self.compute_top_k_mass(token_scores, [1, self.top_k], normalization='minmax')
        top_k_mass = [m.mean() for m in top_k_mass]
        # gini_index = self.compute_gini_index(token_scores).mean()
        if return_entropy_only:
            return entropy, top_k_mass
        ranked_tokens_idx = self.rank_tokens(token_scores)
        if return_scores:
            important_tokens = self.get_top_tokens_with_scores(ranked_tokens_idx, token_scores)
        else:
            important_tokens = self.get_top_tokens(ranked_tokens_idx)
        return important_tokens, entropy, top_k_mass
    
    def get_top_tokens(self, ranked_tokens_idx: torch.Tensor):  # (batch, 512)
        """Get top-k tokens from ranked tokens
        Args:
            ranked_tokens_idx: tensor of shape 
        Returns:
            important_tokens: list of list of top-k tokens
        """
        selected_tokens_idx = ranked_tokens_idx[:, :self.top_k*2]  ## for speed up
        input_ids = self.tokenizer(
            self.sentences, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=512
        )['input_ids']  #(batch, seq_len)
        important_tokens = []
        if self._tqdm:
            pbar = tqdm(range(len(input_ids)), desc="Getting top-k high attribution tokens")
        else:
            pbar = range(len(input_ids))
        
        for i in pbar:
            ids = input_ids[i][selected_tokens_idx[i]].tolist()
            # IMPORTANT (BERT): do NOT use tokenizer.decode(id) here.
            # `decode()` can drop WordPiece markers like "##", which breaks later matching.
            wordpieces = self.tokenizer.convert_ids_to_tokens(ids)
            tokens = [tok for tok in wordpieces if self._is_valid_token(tok)]
            tokens = tokens[:self.top_k]
            important_tokens.append(tokens)
        
        return important_tokens

    def get_top_tokens_with_scores(self, ranked_tokens_idx: torch.Tensor, token_scores: torch.Tensor):
        """Get top-k tokens with their attribution scores (descending)."""
        selected_tokens_idx = ranked_tokens_idx[:, : self.top_k * 2]  ## for speed up
        input_ids = self.tokenizer(
            self.sentences,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )["input_ids"]  # (batch, seq_len)

        # Ensure scores are on CPU for .item()
        if token_scores.is_cuda:
            token_scores_cpu = token_scores.detach().cpu()
        else:
            token_scores_cpu = token_scores.detach()

        important_tokens = []
        if self._tqdm:
            pbar = tqdm(range(len(input_ids)), desc="Getting top-k high attribution tokens (with scores)")
        else:
            pbar = range(len(input_ids))

        for i in pbar:
            idxs = selected_tokens_idx[i].tolist()
            ids = input_ids[i][selected_tokens_idx[i]].tolist()
            wordpieces = self.tokenizer.convert_ids_to_tokens(ids)

            out = []
            for pos, tok in zip(idxs, wordpieces):
                if not self._is_valid_token(tok):
                    continue
                out.append({"token": tok, "score": float(token_scores_cpu[i, pos].item())})
                if len(out) >= self.top_k:
                    break
            important_tokens.append(out)

        return important_tokens
    
    def _is_valid_token(self, tok):
        """Check if a token is valid"""
        tok = tok.strip()
        # BERT WordPiece: keep "##" in the returned token, but validate on surface form.
        surface = tok[2:] if tok.startswith("##") else tok
        surface = surface.strip()
        if len(surface) == 0:
            return False
        if self.use_excluded_tokens and surface.lower() in EXCLUDED_TOKENS and surface.lower() not in self.whitelist_tokens:
            return False
        return (
            tok not in self.tokenizer.all_special_tokens
            and not re.fullmatch(r"[^\w\s]+", surface)
            and len(tok) > 0
        )

    @ torch.no_grad()
    def get_attn_scores(self, sentences: list[str], _tqdm: bool = None):
        """
        Get the attention scores for batch of sentences.
        Args:
            sentences: list of sentences
        Returns:
            attn_scores: tensor of shape (batch, seq_len)
        """
        cls_index = 0
        attn_scores = []
        if _tqdm is None:
            _tqdm = self._tqdm
        if _tqdm:
            pbar = tqdm(range(0, len(sentences), self.batch_size), desc="Getting attention scores")
        else:
            pbar = range(0, len(sentences), self.batch_size)
        for i in pbar:
            start_idx = i
            end_idx = start_idx + self.batch_size
            batch_sentences = sentences[start_idx:end_idx]
            if not batch_sentences:
                continue
            inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            attentions = outputs.attentions[-1] # (batch, num_heads, seq_len, seq_len)
            attentions = attentions[:, :, cls_index, :] #(batch, heads, seq_len)
            attentions = attentions.mean(dim=1) #(batch, seq_len) (8, 512)
            attn_scores.append(attentions.cpu())
        return torch.cat(attn_scores) # (n, seq_len)

    @ torch.no_grad()
    def get_ig_scores(self, sentences: list[str]):
        ig = IntegratedGradients(self._forward_func_logits)
        all_scores = []      
        
        if self._tqdm:
            pbar = tqdm(range(0, len(sentences), self.batch_size), desc="Getting IG scores")
        else:
            pbar = range(0, len(sentences), self.batch_size)

        for i in pbar:
            start_idx = i
            end_idx = start_idx + self.batch_size
            batch_sentences = sentences[start_idx:end_idx]
            if not batch_sentences:
                continue

            inputs = self.tokenizer(
                batch_sentences, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=512
            ).to(self.model.device)
            input_ids = inputs["input_ids"]

            # Get embeddings - BERT uses model.bert instead of model.roberta
            embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
            baseline_embeds = torch.zeros_like(embeddings)  # or: use pad embeddings if you prefer

            logits = self.model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"]).logits
            predicted_labels = logits.argmax(dim=-1)

            attributions = ig.attribute(
                inputs=embeddings,
                baselines=baseline_embeds,
                target=predicted_labels,  
                n_steps=self.ig_n_steps,
                internal_batch_size=embeddings.shape[0],
                additional_forward_args=(inputs["attention_mask"],),
            )
            scores = attributions.sum(dim=-1)  # (batch, seq_len)
            all_scores.append(scores.cpu())

        return torch.cat(all_scores)
    
    def get_saliency_scores(self, sentences: list[str]):
        """Calculate saliency scores for each token in the sentences."""
        saliency = Saliency(self._forward_func_logits)
        all_scores = []
        max_len_in_dataset = 0

        if self._tqdm:
            pbar = tqdm(range(0, len(sentences), self.batch_size), desc="Getting Saliency scores")
        else:
            pbar = range(0, len(sentences), self.batch_size)

        for i in pbar:
            start_idx = i
            end_idx = start_idx + self.batch_size
            batch_sentences = sentences[start_idx:end_idx]
            if not batch_sentences:
                continue

            inputs = self.tokenizer(
                batch_sentences, 
                return_tensors="pt", 
                padding=True, ## avoid padding to max_len and speed up saliency.attribute()
                truncation=True, 
                max_length=512,
            )

            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            seq_lens = attention_mask.sum(dim=-1).tolist()
            max_len_in_batch = max(seq_lens)
            max_len_in_dataset = max(max_len_in_dataset, max_len_in_batch)

            # Get embeddings - BERT uses model.bert instead of model.roberta
            embeddings = self.model.bert.embeddings.word_embeddings(input_ids).detach()
            embeddings.requires_grad = True

            # Get predicted labels
            with torch.no_grad():
                logits = self.model(inputs_embeds=embeddings, attention_mask=attention_mask).logits
            predicted_labels = logits.argmax(dim=-1)

            # Compute saliency
            grads = saliency.attribute(
                embeddings, 
                target=predicted_labels,
                additional_forward_args=(attention_mask,), 
            )
        
            scores = grads.abs().sum(dim=-1)  # (batch, seq_len)
            scores = scores * attention_mask # padding tokens are ignored
            all_scores.append(scores.cpu())
        
        # === Padding to unify length ===
        padded_scores = []
        for s in all_scores:
            pad_len = max_len_in_dataset - s.shape[1]
            if pad_len > 0:
                s = F.pad(s, (0, pad_len), value=0)
            padded_scores.append(s)
        return torch.cat(padded_scores)

    def _forward_func_logits(self, inputs_embeds, attention_mask=None):
        if attention_mask is None:
            attention_mask = (inputs_embeds.abs().sum(dim=-1) != 0).long()
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return output.logits # (batch, num_classes)

    def rank_tokens(self, attn_scores: torch.Tensor):
        sorted_scores, sorted_idx = attn_scores.sort(descending=True)
        return sorted_idx
    
    def stage2_validate_shortcut(self,
                                predictions: list[tuple[int, int]],
                                neighbor_smoothing: bool = True,
                                sim_threshold_data: float = 0.8,
                                sim_threshold_ablation: float = 0.2,
                                method: str = "mask",
                                candidate_tokens_in_each_sentence: list[list[str]] = None,
                                token_occurrence: dict = None,
                                debug_tokens: list[str] | None = None):
        """Stage 2: Validate the candidate tokens as shortcut tokens by LOO masking and optional neighbor smoothing"""
        if neighbor_smoothing:
            self._initialize_neighbor_search()
        self.predictions = predictions
        self.sim_threshold_data = sim_threshold_data             # for finding similar tokens when validating shortcut tokens
        self.sim_threshold_ablation = sim_threshold_ablation     # for finding similar tokens when ablating tokens
        if token_occurrence is None and candidate_tokens_in_each_sentence is not None:
            token_occurrence = self.get_token_occurrence(candidate_tokens_in_each_sentence)
        self.token_occurrence = token_occurrence
        token_similarity_matrix = self.compute_token_similarity_matrix(token_occurrence)
        self.token_similarity_matrix = token_similarity_matrix

        filtered_shortcuts = []
        if self._tqdm:
            pbar = tqdm(token_occurrence.items(), desc="Validating shortcut tokens")
        else:
            pbar = token_occurrence.items()
        
        token_idx = 0
        token_occurrence_new = defaultdict(list)
        debug_set = set(t.strip() for t in (debug_tokens or []) if isinstance(t, str) and t.strip())
        for token, sent_indices in pbar: 
            token_group = self.get_similar_tokens(token_idx)
            is_debug = False
            if debug_set:
                if token.strip() in debug_set:
                    is_debug = True
                else:
                    for tg in token_group:
                        if isinstance(tg, str) and tg.strip() in debug_set:
                            is_debug = True
                            break
            # print(f"len(token_group): {len(token_group)}")
            sent_indices_containing_tokens = []
            for t in token_group:
                sent_indices_containing_tokens.extend(token_occurrence[t])
            sent_indices_containing_tokens = list(set(sent_indices_containing_tokens))
            

            ablated_sentences, ablated_indices = self.engineer_token(
                token_group, sent_indices_containing_tokens, _tqdm=False, method=method
            )
            # With high `sim_threshold_ablation` (e.g., 0.9) it is common that no token positions
            # are matched in any sentence, so nothing is ablated. Skip such tokens gracefully.
            if len(ablated_indices) == 0 or len(ablated_sentences) == 0:
                if is_debug:
                    print(
                        f"[DEBUG stage2] token='{token}' group_size={len(token_group)} "
                        f"n_candidates={len(sent_indices_containing_tokens)} -> NO ABLATION MATCHES "
                        f"(sim_threshold_ablation={self.sim_threshold_ablation}, method={method})"
                    )
                token_idx += 1
                continue
            ablated_pred = get_batch_predictions(ablated_sentences, self.model, self.tokenizer, 
                                                self.model.device, self.batch_size, False)
                    
            # For multi-class: track "label changes" as (from_label, to_label) pairs
            flip_scores = []
            flip_directions = []  # kept for backward-compat (stores "to" labels)
            change_pairs = []
            original_labels_list = []
            neighbor_predictions_list = []
            neighbor_indices_list = []
            # for j, sent_idx in enumerate(sent_indices):
            for j, sent_idx in enumerate(ablated_indices):
                original_label = self.predictions[sent_idx][0]
                original_labels_list.append(original_label)
                masked_pred = ablated_pred[j][0]
                if masked_pred != original_label:  # Label changed => candidate token matters
                    flip_directions.append(masked_pred)
                    change_pairs.append((original_label, masked_pred))
                    score = 1
                    if neighbor_smoothing:
                        neighbor_distribution, neighbor_indices = self.get_neighbor_label_distribution([ablated_sentences[j]], [self.sentences[sent_idx]])
                        score = self.is_label_distribution_skewed(neighbor_distribution, masked_pred) 
                        neighbor_predictions_list.append(neighbor_distribution)
                        neighbor_indices_list.append((sent_idx, neighbor_indices))
                else:
                    score = 0
                flip_scores.append(score)
            mask_sensitivity = self.calculate_ablation_sensitivity(flip_scores, len(ablated_indices))
            prevalence = len(ablated_indices) / len(self.sentences)
            num_flips = sum(1 for s in flip_scores if s > 0)
            if num_flips == 0:
                flip_confidence = 0.0
            else:
                flip_confidence = sum(flip_scores) / num_flips
            # Consistency: for multiclass, prefer consistency of the *change direction* (from->to)
            most_common_change = None
            change_counts_str = {}
            if len(change_pairs) > 0:
                change_counts = Counter(change_pairs)
                change_counts_str = {f"{a}->{b}": int(c) for (a, b), c in change_counts.items()}
                most_common_change, count = change_counts.most_common(1)[0]
                consistency_ratio = count / len(change_pairs) if len(change_pairs) > 0 else 0.0
            else:
                consistency_ratio = 1.0
            majority_label_percentage = Counter(original_labels_list).most_common(1)[0][1]/len(original_labels_list) 
            # sent_indices_containing_tokens, raw prediction, masked prediction, neighbor prediction
            raw_predictions = Counter([self.predictions[i][0] for i in sent_indices_containing_tokens])
            if neighbor_smoothing:
                neighbor_predictions = [dist for dist in neighbor_predictions_list]
            else:
                neighbor_predictions = []
            passes = {
                "mask_sensitivity": mask_sensitivity > self.mask_sensitivity_threshold,
                "prevalence": prevalence > self.min_prevalence,
                "consistency_ratio": consistency_ratio > self.consistency_ratio_threshold,
                "majority_label_percentage": majority_label_percentage > self.majority_label_percentage_threshold,
                "num_flips": num_flips >= self.min_num_flips,
            }
            if is_debug:
                top_changes = []
                if len(change_pairs) > 0:
                    top_changes = Counter(change_pairs).most_common(5)
                print(
                    f"[DEBUG stage2] token='{token}' group_size={len(token_group)} "
                    f"n_candidates={len(sent_indices_containing_tokens)} n_ablated={len(ablated_indices)} "
                    f"mask_sensitivity={mask_sensitivity:.4f} (thr>{self.mask_sensitivity_threshold}) "
                    f"prevalence={prevalence:.4f} (thr>{self.min_prevalence}) "
                    f"consistency_ratio={consistency_ratio:.4f} (thr>{self.consistency_ratio_threshold:.4f}) "
                    f"majority_label_pct={majority_label_percentage:.4f} (thr>{self.majority_label_percentage_threshold:.4f}) "
                    f"num_changes={num_flips} (thr>={self.min_num_flips}) "
                    f"passes={passes} "
                    f"top_changes={[(int(a), int(b), int(c)) for (a,b), c in top_changes]}"
                )

            if all(passes.values()):
                # print(f"mask_sensitivity: {mask_sensitivity}, prevalence: {prevalence}, consistency_ratio: {consistency_ratio}, majority_label_percentage: {majority_label_percentage}, num_flips: {num_flips}")
                filtered_shortcuts.append({"token": token, "sensitivity": mask_sensitivity, "prevalence": prevalence, "num_occur": len(sent_indices),
                    "consistency_ratio": consistency_ratio, "flip_confidence": flip_confidence, "num_flips": num_flips, "majority_label_percentage": majority_label_percentage,
                    "raw_predictions": raw_predictions, "neighbor_predictions": neighbor_predictions, "token_group": token_group,
                    # New (multiclass-friendly) metadata:
                    "change_counts": change_counts_str,
                    "most_common_change": (int(most_common_change[0]), int(most_common_change[1])) if most_common_change is not None else None,
                })
                token_occurrence_new[token] = sent_indices
            token_idx += 1        
        filtered_shortcuts.sort(key=lambda x: x["sensitivity"], reverse=True)

        return filtered_shortcuts, token_occurrence_new    

    def get_similar_tokens(self, token_idx: int):
        token_similarity_matrix = self.token_similarity_matrix
        similar_tokens_idx = (token_similarity_matrix[token_idx] > self.sim_threshold_data).nonzero()[0].tolist() # For NumPy arrays, nonzero() doesn't take as_tuple parameter
        similar_tokens = [list(self.token_occurrence.keys())[i] for i in similar_tokens_idx]
        return similar_tokens


    def compute_token_similarity_matrix(self, token_occurrence: dict):
        token_similarity_matrix = np.zeros((len(token_occurrence), len(token_occurrence)))
        tokens = list(token_occurrence.keys())
        token_embeddings_tensor = self.get_word_embeddings(tokens)  #(token_num, 768)
        
        token_embeddings_tensor = F.normalize(token_embeddings_tensor, p=2, dim=-1) # (token_num, 768)
        token_similarity_matrix = torch.matmul(token_embeddings_tensor, token_embeddings_tensor.T) # (token_num, token_num)

        return token_similarity_matrix.cpu().numpy()

    @ torch.no_grad()
    def get_word_embeddings(self, tokens: List[str]):
        encoding = self.tokenizer(tokens, add_special_tokens=False, return_tensors="pt", padding=True)
        token_ids = encoding["input_ids"].to(self.model.device)
        if self.use_bert_base_for_embeddings:
            if self.bert_base_model is None:
                self.bert_base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
                self.bert_base_model.to(self.model.device)
            token_embs = self.bert_base_model.bert.embeddings.word_embeddings(token_ids)  
        else:
            token_embs = self.model.bert.embeddings.word_embeddings(token_ids)  

        attention_mask = encoding["attention_mask"].to(self.model.device)
        embeds_sum = (token_embs * attention_mask.unsqueeze(-1)).sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        token_embs = embeds_sum / lengths
        return token_embs


    def _bert_token_starts_with_space(self, token: str) -> bool:
        """
        BERT-specific: Check if a token starts with a space.
        BERT uses WordPiece tokenization where tokens starting with ## are subword tokens.
        Tokens that don't start with ## typically start a new word (and thus have a space before them).
        However, the first token after [CLS] doesn't have a space prefix.
        """
        # BERT tokens starting with ## are subword tokens (no space)
        # Other tokens (except special tokens) typically start a new word (have space)
        # But this is not always reliable, so we use a heuristic
        return not token.startswith('##') and token not in self.tokenizer.all_special_tokens

    def _decode_preserving_sep(self, input_ids, keep_mask: bool = False) -> str:
        """Decode token IDs while preserving internal [SEP] for NLI-style inputs.

        Removes [CLS], [PAD], and the trailing [SEP] (added by the tokenizer),
        but keeps any internal [SEP] that separates premise from hypothesis.
        Optionally keeps [MASK] tokens (for the "mask" ablation method).
        """
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        for st in (self.tokenizer.cls_token, self.tokenizer.pad_token):
            if st:
                text = text.replace(st, '')
        if not keep_mask and self.tokenizer.mask_token:
            text = text.replace(self.tokenizer.mask_token, '')
        if self.tokenizer.unk_token:
            text = text.replace(self.tokenizer.unk_token, '')
        text = text.strip()
        if self.tokenizer.sep_token and text.endswith(self.tokenizer.sep_token):
            text = text[:-len(self.tokenizer.sep_token)].strip()
        return text

    @ torch.no_grad()
    def engineer_token(self, tokens: List[str], sentences: List[int], _tqdm: bool = None, method: str = "mask", 
    blacklist_similarity_threshold: float = 0.9, dictionary: str = "datamuse", num_outsentences_for_each_sentence: int = 1, topics=None):
        """
        ablate the token positions along the sequence length whose static embeddings are most similar to the candidate token
        Args:
            tokens: the candidate tokens
            sentences: the sentence indices in the cluster
            method: "delete" to remove tokens completely, "mask" to replace with [MASK] token, 
                   "mask_fill" to replace with contextually appropriate tokens using fill-mask
            blacklist_similarity_threshold: threshold for adding similar tokens to blacklist (default: 0.9)

        Returns:
            ablated_sentences: the ablated sentences
            ablated_indices: the indices of the ablated tokens
        """
        #1) get embedding of given token
        self.suspicious_tokens = tokens
        # Pre-compute suspicious lemmas once (for rewrite_by_dict)
        if method == "rewrite_by_dict":
            self._suspicious_lemmas = set(get_lemma_spacy(s.lower().strip()) for s in tokens)
            nltk.download('wordnet', quiet=True)
            nlp = spacy.load("en_core_web_sm")
        # IMPORTANT (BERT): tokens are expected to be WordPieces (possibly with "##").
        # Use vocab lookup instead of encode/decode roundtrips to avoid losing "##" and changing ids.
        vocab = self.tokenizer.get_vocab()

        def _token_to_id(tok: str) -> int | None:
            t = tok.strip()
            if t in vocab:
                return int(vocab[t])
            # If a caller passed surface form without "##", also try continuation form.
            if not t.startswith("##") and ("##" + t) in vocab:
                return int(vocab["##" + t])
            # Fallback: try encode (may produce multiple ids); take the first.
            enc = self.tokenizer.encode(t, add_special_tokens=False)
            if len(enc) == 0:
                return None
            return int(enc[0])

        token_ids = []
        for t in tokens:
            tid = _token_to_id(t)
            if tid is not None:
                token_ids.append(tid)
        token_embs = [self.model.bert.embeddings.word_embeddings.weight[token_id] for token_id in token_ids]

        if method == "rewrite_by_sim":
            resources = self.prepare_token_rewriting_resources(shortcut_token_ids=token_ids, blacklist_similarity_threshold=blacklist_similarity_threshold)

        ablated_sentences = []
        ablated_indices = []
        if _tqdm is None:
            _tqdm = self._tqdm
        if _tqdm:
            pbar = tqdm(sentences, desc="Processing tokens")
        else:
            pbar = sentences
        
        for sent_idx in pbar:
            
            #2) get sentence embeddings of the sentence
            sent_text = self.sentences[sent_idx]
            encoded = self.tokenizer(sent_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = encoded["input_ids"][0]
            embeddings = self.model.bert.embeddings.word_embeddings.weight[input_ids]

            #3) compute similarity between token and the target token   #!
            sims = [torch.cosine_similarity(token_emb, embeddings, dim=1) for token_emb in token_embs]
            
            #4) find the position of the token with the highest similarity
            matched_position_all = []
            for token_emb, token, sim in zip(token_embs, tokens, sims):
                matched_indices = torch.where(sim > self.sim_threshold_ablation)[0].tolist()
                matched_position_all.extend(matched_indices)
            
            matched_position_all = list(set(matched_position_all)) # remove the duplicated indices
            special_ids_set = set(int(sid) for sid in self.tokenizer.all_special_ids)
            matched_position_all = [
                pos for pos in matched_position_all
                if int(input_ids[pos]) not in special_ids_set
            ]
            
            if len(matched_position_all) > 0:
                ablated_indices.append(sent_idx)

                #5) ablate matched tokens
                input_ids_ablated = input_ids.clone()
                
                if method == "delete":
                    matched_indices_sorted = sorted(matched_position_all, reverse=True) # avoid index shifting when deleting
                    input_ids_list = input_ids_ablated.tolist()
                    for idx in matched_indices_sorted:
                        if 0 <= idx < len(input_ids_list):
                            input_ids_list.pop(idx)

                    input_ids_ablated = torch.tensor(input_ids_list)
                    ablated_sentence = self._decode_preserving_sep(input_ids_ablated)
                
                elif method == "mask":
                    for idx in matched_position_all:
                        input_ids_ablated[idx] = self.tokenizer.mask_token_id
                    
                    ablated_sentence = self._decode_preserving_sep(input_ids_ablated, keep_mask=True)
                
                elif method == "mask_fill":  # use mlm mask
                    # Initialize mask filler if not exists
                    if not hasattr(self, 'mask_filler'):
                        from transformers import pipeline
                        self.mask_filler = pipeline("fill-mask", model="bert-base-uncased", tokenizer="bert-base-uncased", device=self.model.device)
                    
                    # Process each matched position sequentially
                    current_ids = input_ids_ablated.clone()
                    for idx in matched_position_all:
                        before_token = self.tokenizer.decode(current_ids[idx], skip_special_tokens=True)

                        # print(f"before_token: {before_token}")
                        
                        # Create masked version for this position
                        temp_ids = current_ids.clone()
                        temp_ids[idx] = self.tokenizer.mask_token_id
               
                        masked_sentence = self._decode_preserving_sep(temp_ids, keep_mask=True)
    
                        # Ensure we have exactly one mask token
                        if masked_sentence.count(self.tokenizer.mask_token) != 1:
                            print(f"Warning: Incorrect mask count in '{masked_sentence}', skipping...")
                            continue
                        
                        # Get fill-mask predictions with error handling
                        filled_tokens = self.mask_filler(masked_sentence, top_k=10)
                        print(f"filled_tokens: {filled_tokens}")
                        print("--------------------------------")
                        
                        # Choose replacement token (avoid original token)
                        replacement_token_str = filled_tokens[0]["token_str"]
                        if replacement_token_str.strip() == before_token.strip() and len(filled_tokens) > 1:
                            replacement_token_str = filled_tokens[1]["token_str"]
                        
                        # Update current_ids with the replacement
                        try:
                            replacement_ids = self.tokenizer.encode(replacement_token_str, add_special_tokens=False)
                            if replacement_ids:
                                current_ids[idx] = replacement_ids[0]
                        except Exception as e:
                            print(f"Failed to encode replacement token '{replacement_token_str}': {e}")
                            continue
                    
                    ablated_sentence = self._decode_preserving_sep(current_ids)
                
                elif method == "rewrite_by_sim":
                    # Use FAISS index over L2-normalized token embeddings to find nearest tokens
                    faiss_index = resources.get("faiss_index")
                    token_metadata = resources.get("token_metadata")
                    normalized_embeddings = resources.get("normalized_embeddings")
                    special_ids = resources.get("special_ids", set())
                    blacklist_ids = resources.get("blacklist_ids", set())  # Get blacklist
                    # Caches persisted in resources across all sentences
                    nn_cache = resources["nn_cache"]                 # token_id -> List[int]
                    replacement_cache = resources["replacement_cache"] # token_id -> Optional[int]
                    neighbor_k = 32

                    for pos_idx in matched_position_all:
                        src_inpid = int(input_ids_ablated[pos_idx])  
                        if src_inpid in special_ids:
                            continue

                        # Check cache or run search
                        if src_inpid in replacement_cache:
                            cached_repl = replacement_cache[src_inpid]
                            if cached_repl is not None:
                                input_ids_ablated[pos_idx] = int(cached_repl)
                                continue
                            # if None cached, fall through to try again (in case constraints changed)

                        if src_inpid in nn_cache:
                            nn_ids = nn_cache[src_inpid]
                        else:
                            # Prepare normalized query vector (1, dim)
                            query = normalized_embeddings[src_inpid:src_inpid+1].astype('float32')
                            # Search nearest neighbors
                            _, nn = faiss_index.search(query, neighbor_k)
                            nn_ids = nn[0].tolist()  
                            nn_cache[src_inpid] = nn_ids

                        # Filter and select a valid replacement
                        src_meta = token_metadata.get(src_inpid, None)
                        replacement_inpid = None
                        for cand_inpid in nn_ids:
                            cand_inpid = int(cand_inpid)
                            if cand_inpid == src_inpid:  # a)avoid the original token
                                continue
                            if cand_inpid in blacklist_ids:  # b)avoid blacklisted tokens (shortcut tokens and their similar ones)
                                continue
                            meta = token_metadata.get(cand_inpid)
                            if meta is None or meta["is_special"]:  # c)avoid the special token
                                continue
                            # d) Keep whitespace prefix property aligned (BERT uses ## for subwords)
                            if src_meta is not None and meta["starts_with_space"] != src_meta["starts_with_space"]:  
                                continue
                            replacement_inpid = cand_inpid
                            break  # get the first valid replacement token  #TODO: score the replacement tokens and select the best one

                        if replacement_inpid is not None:
                            input_ids_ablated[pos_idx] = replacement_inpid
                        # update replacement cache (store even None to skip re-search next time)
                        replacement_cache[src_inpid] = replacement_inpid

                    ablated_sentence = self._decode_preserving_sep(input_ids_ablated)
                
                elif method == "rewrite_by_dict":
                    special_ids = set(self.tokenizer.all_special_ids)
                    # save candidates for each position
                    candidates_at_each_position = {}
                    for pos_idx in matched_position_all:
                        src_inpid = int(input_ids_ablated[pos_idx])
                        if src_inpid in special_ids:  # skip special tokens
                            continue
                        # IMPORTANT: for BERT WordPiece we must use token strings (with "##") via convert_ids_to_tokens.
                        # tokenizer.decode([id]) often drops the "##" marker, which breaks prefix-space logic.
                        orig_token_str = self.tokenizer.convert_ids_to_tokens(src_inpid)
                        # BERT: tokens starting with ## are subwords (no whitespace before), others are word-start.
                        orig_has_prefix_space = self._bert_token_starts_with_space(orig_token_str)
                        orig_surface = orig_token_str[2:] if orig_token_str.startswith("##") else orig_token_str
                        orig_surface = orig_surface.strip()
                        if dictionary == "datamuse":
                            candidates = self.get_synonyms_from_datamuse(orig_surface, topics=topics)  # get the synonyms from datamuse
                            # print(f"candidates: {candidates}")
                        elif dictionary == "wordnet":
                            candidates = self.get_synonyms_from_wordnet(orig_surface)  # get the synonyms from wordnet
                        else:
                            raise ValueError(f"Unknown dictionary: {dictionary}. Use 'datamuse' or 'wordnet'.")

                        def _encode_candidate_as_wordpiece(word: str, as_continuation: bool) -> list[int]:
                            """
                            Return a list of token ids (possibly length>1) for `word`.
                            If as_continuation=True, try to make the first piece a continuation ("##...") when possible.
                            """
                            toks = self.tokenizer.tokenize(word)
                            if not toks:
                                return []
                            if as_continuation and not toks[0].startswith("##"):
                                pref = "##" + toks[0]
                                # Only swap if the prefixed form exists in vocab; otherwise keep original tokens.
                                if pref in self.tokenizer.get_vocab():
                                    toks[0] = pref
                            ids = self.tokenizer.convert_tokens_to_ids(toks)
                            # convert_tokens_to_ids may return None for unknown; filter those out.
                            return [int(i) for i in ids if i is not None]

                        # IMPORTANT: keep a consistent representation:
                        # candidate_token_id is ALWAYS a List[int] (never an int).
                        candidate_token_ids = [[int(self.tokenizer.mask_token_id)]]
                        if candidates:
                            candidate_token_ids = []
                            for cand in candidates:
                                cand_ids = _encode_candidate_as_wordpiece(cand, as_continuation=(not orig_has_prefix_space))
                                if cand_ids:
                                    candidate_token_ids.append(cand_ids)
                        if len(candidate_token_ids) > 0:
                            candidates_at_each_position[pos_idx] = candidate_token_ids
                        else:
                            raise ValueError(f"No synonyms found for '{orig_surface}'")

                    # ablated_sentence = self.tokenizer.decode(input_ids_ablated, skip_special_tokens=True)
                    
                    # each position has a list of candidate token ids
                    # randomly select a candidate token id for each position
                    # and we want to output 1 or 5 new sentences
                    output_sentences = []
                    candidates_at_each_position_copy = copy.deepcopy(candidates_at_each_position)
                    positions_to_replace = list(candidates_at_each_position.keys())
                    if len(positions_to_replace) == 0:
                        # Nothing replaceable (e.g., only special tokens matched) -> skip this sentence.
                        continue
                    for i in range(num_outsentences_for_each_sentence): #*10
                        input_ids_ablated_copy = input_ids_ablated.clone()
                        replaced_ids = {}
                        for pos_idx in positions_to_replace:
                            # candidate_token_id = random.choice(candidates_at_each_position_copy[pos_idx])
                            if len(candidates_at_each_position_copy[pos_idx]) == 0:
                                candidates_at_each_position_copy[pos_idx] = copy.deepcopy(candidates_at_each_position[pos_idx])
                            candidate_token_id = random.choice(candidates_at_each_position_copy[pos_idx])
                            candidates_at_each_position_copy[pos_idx].remove(candidate_token_id)
                            replaced_ids[pos_idx] = candidate_token_id
                        for pos_idx in sorted(positions_to_replace, reverse=True):
                            candidate_token_id = replaced_ids[pos_idx]
                            # Robust normalization: candidate_token_id should be List[int], but handle int defensively.
                            if isinstance(candidate_token_id, (int, np.integer)):
                                candidate_token_id = [int(candidate_token_id)]
                            if len(candidate_token_id) == 0:
                                continue
                            token_id_tensor = torch.tensor(candidate_token_id, dtype=torch.long)
                            input_ids_ablated_copy = torch.cat(
                                [input_ids_ablated_copy[:pos_idx], token_id_tensor, input_ids_ablated_copy[pos_idx+1:]]
                            )
                        output_sentence = self._decode_preserving_sep(input_ids_ablated_copy)
                        output_sentences.append(output_sentence)
                    ablated_sentence = output_sentences
                else:
                    raise ValueError(f"Unknown ablation method: {method}. Use 'delete', 'mask', 'mask_fill', 'rewrite_by_sim', or 'rewrite_by_dict'.")
                
                if method == "rewrite_by_dict":
                    ablated_sentences.extend(output_sentences)
                else:
                    ablated_sentences.append(ablated_sentence)

        return ablated_sentences, ablated_indices
    
    def get_synonyms_from_datamuse(self, word, pos=None, max_results=60, max_retries=3, topics=None):
        """
        get the synonyms from datamuse API
        word: the word to query
        max_results: the maximum number of results to return
        max_retries: maximum number of retry attempts
        """
        # Initialize cache if not exists
        if not hasattr(self, '_datamuse_cache'):
            self._datamuse_cache = {}
        
        # Check cache first
        cache_key = (word.lower(), pos, max_results)
        if cache_key in self._datamuse_cache:
            return self._datamuse_cache[cache_key]
        
        # API call with increasing timeout
        url = f"https://api.datamuse.com/words?ml={word}&max={max_results}&topics={topics}"
        
        # Start with 5 seconds, then increase by 10 seconds each retry
        timeout_values = [5, 15, 25, 35]
        
        for attempt in range(max_retries):
            timeout = timeout_values[attempt]
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                synonyms = []
                      
                for item in data:
                    w = item["word"]
                    tags = item.get("tags", [])
                    if " " in w or "_" in w or "-" in w:
                        continue
                    # if len(self.tokenizer.tokenize(w)) > 1:
                    #     print(f"skipping {w} because it is longer than 1 token")
                    #     continue
                    if any(get_lemma_spacy(w) in suspicious_lemma for suspicious_lemma in self._suspicious_lemmas) or \
                       any(suspicious_lemma in get_lemma_spacy(w) for suspicious_lemma in self._suspicious_lemmas):
                        continue
                    synonyms.append(w)
                
                # Cache the result
                self._datamuse_cache[cache_key] = synonyms
                return synonyms
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout ({timeout}s) fetching synonyms for '{word}', retrying with longer timeout ({timeout_values[attempt + 1]}s)...")
                else:
                    print(f"Timeout fetching synonyms for '{word}' after {max_retries} attempts with timeouts {timeout_values[:max_retries]}")
                    self._datamuse_cache[cache_key] = []
                    return []
                    
            except Exception as e:
                raise Exception(f"Error fetching synonyms for '{word}': {e}")
        
        # Should not reach here, but just in case
        return []

    def get_synonyms_from_wordnet(self, word, pos=None, min_freq=4.0):
        """
        word: string, e.g. "study"
        pos: pos filter, e.g. 'n' (noun), 'v' (verb), 'a' (adj), 'r' (adv)
            if not provided, return all synonyms
        """
        # synset_list = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
        if not hasattr(self, '_wordnet_cache'):
            self._wordnet_cache = {}
        
        cache_key = (word.lower(), pos, min_freq)
        if cache_key in self._wordnet_cache:
            return self._wordnet_cache[cache_key]
        
        syn_set = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
        candidates = set()

        for syn in syn_set:
            for lemma in syn.lemmas():
                if lemma.count() < min_freq:
                    continue
                lemma_name = lemma.name().replace('_', ' ')
                # if len(self.tokenizer.tokenize(lemma_name)) > 1:
                #     continue
                if ' ' in lemma_name:
                    continue
                if get_lemma_spacy(lemma_name) in self._suspicious_lemmas:
                    continue
                candidates.add(lemma_name.lower())

        # avoid the original word itself
        candidates.discard(word.lower())
        self._wordnet_cache[cache_key] = sorted(candidates)
        return self._wordnet_cache[cache_key]

    def prepare_token_rewriting_resources(self, shortcut_token_ids: List[int] = None, blacklist_similarity_threshold: float = 0.9):
        """Prepare and cache resources for token rewriting (persist across sentences).
        
        Args:
            shortcut_token_ids: List of token IDs identified as shortcut tokens
            blacklist_similarity_threshold: Threshold for adding similar tokens to blacklist (default: 0.9)
        """
        if hasattr(self, "_rewrite_resources"):
            return self._rewrite_resources
        
        token_embeddings = self.model.bert.embeddings.word_embeddings.weight.data  # (vocab_size, embedding_dim)
        print(f"token_embeddings shape: {token_embeddings.shape}")
        
        vocab = self.tokenizer.get_vocab()
        special_ids = set(self.tokenizer.all_special_ids)
        token_metadata = {}
        
        for token, token_id in vocab.items():
            token_metadata[token_id] = {
                'token': token,
                'starts_with_space': self._bert_token_starts_with_space(token),  # BERT-specific space detection
                'is_special': token_id in special_ids,
                'embedding': token_embeddings[token_id]
            }
        
        embeddings_np = token_embeddings.detach().cpu().numpy().astype('float32')
        # faiss.normalize_L2(embeddings_np)
        index = self._build_faiss_index_for_matrix(embeddings_np)

        # Build blacklist: shortcut tokens + their similar tokens (>= blacklist_similarity_threshold)
        blacklist_ids = set()
        if shortcut_token_ids is not None and len(shortcut_token_ids) > 0:
            for token_id in shortcut_token_ids:
                blacklist_ids.add(token_id)
                query_emb = embeddings_np[token_id:token_id+1].astype('float32')
                _, nn_indices = index.search(query_emb, 1000)  # Get top 1000 neighbors
                
                for nn_id in nn_indices[0]:
                    nn_id = int(nn_id)
                    # Compute cosine similarity using torch
                    vec1 = torch.tensor(embeddings_np[token_id]).unsqueeze(0)  # (1, 768)
                    vec2 = torch.tensor(embeddings_np[nn_id]).unsqueeze(0)    # (1, 768)
                    similarity = float(torch.cosine_similarity(vec1, vec2, dim=1).item())
                    if similarity >= blacklist_similarity_threshold:
                        blacklist_ids.add(nn_id)
            
            print(f"Blacklist size: {len(blacklist_ids)} tokens")  
            # Print some examples
            example_tokens = []
            for bid in list(blacklist_ids)[:20]:
                if bid in token_metadata:
                    example_tokens.append(token_metadata[bid]['token'])
            print(f"Example blacklisted tokens (first 20): {example_tokens}")
            
            blacklist_info = []
            for token_id in shortcut_token_ids:
                print("--------------------------------")
                token_text = token_metadata[token_id]['token']
                print(f"token_text: {token_text}")
                # Find similar tokens for this shortcut token
                similar_tokens = []
                query_emb = embeddings_np[token_id:token_id+1].astype('float32')
                _, nn_indices = index.search(query_emb, 100)
                for nn_id in nn_indices[0]:
                    nn_id = int(nn_id)
                    if nn_id not in blacklist_ids:  #!
                        vec1 = torch.tensor(embeddings_np[token_id]).unsqueeze(0)  # (1, 768)
                        vec2 = torch.tensor(embeddings_np[nn_id]).unsqueeze(0)    # (1, 768)
                        similarity = float(torch.cosine_similarity(vec1, vec2, dim=1).item())
                        # if similarity >= blacklist_similarity_threshold:
                        if similarity >= 0.2 and similarity <= 0.4:
                            similar_tokens.append({
                                'token_id': nn_id,
                                'token': token_metadata[nn_id]['token'],
                                'similarity': similarity
                            })
                            # print(f"similarity: {similarity}")
                            # print(f"nn_token_text: {token_metadata[nn_id]['token']}")
                
                blacklist_info.append({
                    'shortcut_token_id': token_id,
                    'shortcut_token': token_text,
                    'num_similar_tokens': len(similar_tokens),
                    'similar_tokens': similar_tokens[:10]  # Keep top 10 for each
                })
                print(similar_tokens[:10])
            
    
            resources_blacklist_info = blacklist_info
        else:
            resources_blacklist_info = []

        resources = {
            'token_embeddings': token_embeddings,
            'token_metadata': token_metadata,
            'normalized_embeddings': embeddings_np,
            'faiss_index': index,
            'special_ids': special_ids,
            'blacklist_ids': blacklist_ids,  # Add blacklist to resources
            'blacklist_info': resources_blacklist_info,  # Detailed info about blacklisted tokens
            'nn_cache': {},                 # token_id -> List[int]
            'replacement_cache': {},        # token_id -> Optional[int]
        }
        self._rewrite_resources = resources
        return resources

    def save_blacklist_info(self, output_path: str = "results/blacklist_info.json"):
        """Save blacklist information to a JSON file for inspection.
        
        Args:
            output_path: Path to save the blacklist information
        """
        import json
        import os
        
        if not hasattr(self, "_rewrite_resources"):
            print("No blacklist information available. Run engineer_token with method='rewrite' first.")
            return
        
        blacklist_info = self._rewrite_resources.get("blacklist_info", [])
        blacklist_ids = self._rewrite_resources.get("blacklist_ids", set())
        
        if len(blacklist_info) == 0:
            print("No blacklist information to save.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for JSON serialization
        output_data = {
            "total_blacklisted_tokens": len(blacklist_ids),
            "num_shortcut_tokens": len(blacklist_info),
            "shortcut_tokens": blacklist_info
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Blacklist information saved to {output_path}")
        print(f"Total blacklisted tokens: {len(blacklist_ids)}")
        print(f"Number of shortcut tokens: {len(blacklist_info)}")

    def get_token_occurrence(self, candidate_tokens: list[list[str]]):
        token_occurrence = defaultdict(list)
        for i, tokens in enumerate(candidate_tokens):
            for t in tokens:
                token_occurrence[t].append(i)
        return token_occurrence

    def _initialize_neighbor_search(self):
        """
        Initialize neighbor search: always use checkpoint model embeddings, 
        but choose between FAISS or sklearn for nearest neighbor search.
        """
        self.test_embeddings = self._get_sentence_embeddings(self.sentences)
        
        if FAISS_AVAILABLE:
            print("Using FAISS for fast nearest neighbor search...")
            self._setup_faiss_search()
        else:
            raise ValueError("FAISS not available")
            # print("FAISS not available, falling back to sklearn for nearest neighbor search")
            # self._setup_sklearn_search()

    def _setup_faiss_search(self):
        """
        Setup FAISS index for fast nearest neighbor search with BERT embeddings.
        
        Strategy: Use L2-normalized embeddings with IndexFlatIP for cosine similarity.
        - IndexFlatIP computes inner product: <a, b>
        - When vectors are L2-normalized: <a, b> = ||a|| * ||b|| * cos(θ) = cos(θ)
        - This gives us cosine similarity directly

        Priority: HNSW > IVFPQ > IVF > Flat 
        """
        self.faiss_index = self._build_faiss_index_for_matrix(self.test_embeddings)
        print("Note: Using L2-normalized embeddings with FlatIP, IVFFlat, IVFPQ, or HNSW for cosine similarity")
    
    # def _setup_sklearn_search(self):
    #     """Setup sklearn for nearest neighbor search with BERT embeddings."""
    #     self.neighbor_model = NearestNeighbors(n_neighbors=self.k_neighbors, metric='cosine')
    #     self.neighbor_model.fit(self.test_embeddings)
    #     print(f"Sklearn index created with {len(self.sentences)} BERT vectors")

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Get last hidden state embeddings given input list of sentences.
        Args:
            sentences: List of sentences to embed
            
        Returns:
            numpy array of embeddings
        """
        embeddings = []

        if self.use_bert_base_for_embeddings:
            model = self.bert_base_model
        else:
            model = self.model
        
        # Process in batches
        # for i in tqdm(range(0, len(sentences), self.batch_size), desc="Getting BERT embeddings"):
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

    def _build_faiss_index_for_matrix(self, embeddings_np: np.ndarray):
        """
        Build a FAISS index for a given matrix of embeddings using the same
        strategy as sentence FAISS setup. Returns a FAISS index that performs
        cosine similarity search via L2-normalized vectors and inner product.
        """
        # Ensure float32 copy and L2 normalization
        embeddings_np = embeddings_np.astype('float32').copy()
        faiss.normalize_L2(embeddings_np)
        dimension = embeddings_np.shape[1]
        try:
            # Try GPU first if available
            print("Trying to use GPU FAISS index...")
            res = faiss.StandardGpuResources()
            if self.use_hnsw:
                print(f"Using HNSW index (M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}, efSearch={self.hnsw_ef})...")
                cpu_index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
                cpu_index.hnsw.efConstruction = self.hnsw_ef_construction
                cpu_index.hnsw.efSearch = self.hnsw_ef
                cpu_index.add(embeddings_np)
                return faiss.index_cpu_to_gpu(res, 0, cpu_index)
            if self.use_ivfpq:
                print(f"Using IVFPQ index (nlist={self.nlist}, nbits={self.nbits})...")
                quantizer = faiss.IndexFlatIP(dimension)
                cpu_index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, self.m, self.nbits)
                cpu_index.train(embeddings_np)
                cpu_index.add(embeddings_np)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                gpu_index.nprobe = self.nprobe
                return gpu_index
            if self.use_ivf:
                print(f"Using IVF index (nlist={self.nlist})...")
                quantizer = faiss.IndexFlatIP(dimension)
                cpu_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                cpu_index.train(embeddings_np)
                cpu_index.add(embeddings_np)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                gpu_index.nprobe = self.nprobe
                return gpu_index
            print("Using FlatIP index...")
            cpu_index = faiss.IndexFlatIP(dimension)
            cpu_index.add(embeddings_np)
            return faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as e:
            print(f"[Warning] GPU FAISS failed: {e}. Falling back to CPU FAISS.")
            if self.use_hnsw:
                print(f"Using CPU HNSW index (M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}, efSearch={self.hnsw_ef})...")
                cpu_index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
                cpu_index.hnsw.efConstruction = self.hnsw_ef_construction
                cpu_index.hnsw.efSearch = self.hnsw_ef
                cpu_index.add(embeddings_np)
                return cpu_index
            if self.use_ivfpq:
                print(f"Using CPU IVFPQ index (nlist={self.nlist}, nbits={self.nbits})...")
                quantizer = faiss.IndexFlatIP(dimension)
                cpu_index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, self.m, self.nbits)
                cpu_index.train(embeddings_np)
                cpu_index.add(embeddings_np)
                cpu_index.nprobe = self.nprobe
                return cpu_index
            if self.use_ivf:
                print(f"Using CPU IVF index (nlist={self.nlist})...")
                quantizer = faiss.IndexFlatIP(dimension)
                cpu_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                cpu_index.train(embeddings_np)
                cpu_index.add(embeddings_np)
                cpu_index.nprobe = self.nprobe
                return cpu_index
            print("Using CPU FlatIP index...")
            cpu_index = faiss.IndexFlatIP(dimension)
            cpu_index.add(embeddings_np)
            return cpu_index
    
    def _search_neighbors(self, sentences: List[str], original_sentences: List[str]) -> List[List[int]]:
        """
        sentences: list of sentences to search neighbors for
        original_sentences: list of original sentences
        Returns: list of neighbor indices for each sentence
        """
        sentence_embeddings = self._get_sentence_embeddings(sentences)  # shape: (batch, dim)
        if FAISS_AVAILABLE:
            sentence_embeddings_normalized = sentence_embeddings.copy()
            faiss.normalize_L2(sentence_embeddings_normalized)
            distances, indices = self.faiss_index.search(sentence_embeddings_normalized, self.k_neighbors+1)   
        else:
            distances, indices = self.neighbor_model.kneighbors(sentence_embeddings, n_neighbors=self.k_neighbors+1)

        # exclude the original sentence itself
        neighbor_indices = []
        for i in range(len(sentences)):
            if original_sentences[i] in self.sentences:
                self_idx = self.sentences.index(original_sentences[i])
                idxs = [int(j) for j in indices[i] if int(j) != self_idx]
            else:
                idxs = [int(j) for j in indices[i]]
            neighbor_indices.append(idxs[:self.k_neighbors])
        return neighbor_indices

    def is_label_distribution_skewed(self, label_distribution: Dict[int, int], masked_pred: int) -> bool:
        """
        Check if label distribution is skewed (indicating potential spurious correlation).    
        Args:
            label_distribution: Dictionary mapping label to count
            
        Returns:
            True if distribution is skewed, False otherwise
        """
        if len(label_distribution) <= 1:
            return 1  # If there's only one label, it's skewed
        
        total_count = sum(label_distribution.values())
        target_count = label_distribution.get(masked_pred, 0)
        return (target_count / total_count) if total_count else 0.0
    
    def get_neighbor_label_distribution(self, ablated_sentences: List[str], original_sentences: List[str]) -> Dict[int, int]:
        """ Get label distribution of neighbors"""
        neighbor_indices = self._search_neighbors(ablated_sentences, original_sentences)[0]
        neighbor_predictions = [self.predictions[i][0] for i in neighbor_indices]
        label_distribution = Counter(neighbor_predictions)
        return dict(label_distribution), neighbor_indices

    def calculate_ablation_sensitivity(self, score_list, len_occur):
        """Calculate the sensitivity of the model to the ablation of a token."""
        if len_occur == 0:
            return 0.0
        return sum(score_list) / len_occur


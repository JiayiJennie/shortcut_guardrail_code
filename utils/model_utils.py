import torch, re #umap
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
from tqdm import tqdm
# from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import torch.nn.functional as F
from utils.stop_tokens import EXCLUDED_TOKENS

@dataclass
class TokenScore:
    text: str
    sim: float
    margin: float




def get_accuracy(predictions, labels):
    """
    args:
        predictions: List[Tuple[int, float]]
        labels: List[int]
    return:
        accuracy: float
    """
    correct = 0
    total = len(predictions)
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred[0] == label:
            correct += 1
    return correct / total

def load_model(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def load_test_data(test_data_path):
    test_sentences = []
    test_labels = []
    df = pd.read_csv(test_data_path) # column 1 is sentence, column 2 is label
    for index, row in df.iterrows():
        test_sentences.append(row['sentence'])
        test_labels.append(row['label'])

    return test_sentences, test_labels


@ torch.no_grad()
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

        iterator = range(0, len(sentences), batch_size)
        if _tqdm:
            iterator = tqdm(iterator, desc="Getting predictions", total=len(sentences) // batch_size + 1)

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

@ torch.no_grad()
def get_roberta_embeddings(sentences: List[str], model: AutoModelForSequenceClassification, 
                           tokenizer: AutoTokenizer, device: str, batch_size: int, _tqdm: bool = True) -> np.ndarray:
    """
    Get RoBERTa embeddings for a list of sentences.
    
    Args:
        sentences: List of sentences to embed
        
    Returns:
        numpy array of embeddings
    """
    embeddings = []
    
    iterator = range(0, len(sentences), batch_size)
    if _tqdm:
        iterator = tqdm(iterator, desc="Getting sentence embeddings", total=len(sentences) // batch_size + 1)
    
    for i in iterator:
        batch_sentences = sentences[i:i + batch_size] 
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
     
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


def get_layer_embedding(hidden_states, layers, method="avg"):
    """
    Get aggregated embeddings
    """
    if method == "avg":
        embs = torch.stack([hidden_states[i] for i in layers]).mean(dim=0)
    elif method == "concat":
        embs = torch.cat([hidden_states[i] for i in layers], dim=-1)
    # elif method == "concat_pca":
    #     embs = concat_and_reduce(hidden_states, layers[0], layers[1], target_dim=hidden_states[0].size(-1))
    return embs


# def concat_and_reduce(hs, layer_a, layer_b, target_dim=768):
#     """concat two layers and reduce to target_dim"""
#     emb_a = hs[layer_a]  # [seq_len, dim]
#     emb_b = hs[layer_b]  # [seq_len, dim]
#     concat_emb = torch.cat([emb_a, emb_b], dim=-1)  # [seq_len, 2*dim]

#     # PCA 
#     pca = PCA(n_components=target_dim)
#     concat_np = concat_emb.cpu().numpy()
#     reduced_np = pca.fit_transform(concat_np)
#     reduced_emb = torch.tensor(reduced_np, dtype=torch.float32)
#     return reduced_emb


def debias_embeddings(X, n_pcs=2, strength=1.0):
    """'All-but-the-Top' embedding post-processing (Mu & Viswanath, 2018).
    Removes mean and top `n_pcs` principal components to reduce anisotropy."""
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    
    # remove mean
    X = X - np.mean(X, axis=0, keepdims=True)
    # remove first n_pcs principal components
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T[:, :n_pcs]
    X = X - strength * (X @ V) @ V.T   # strength: the strength of the debiasing
    return X


@ torch.no_grad()
def extract_token_embeddings(sentences: List[str], model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str, 
                            layers_idxs: List[int] = [12], embeddings_method: str = "avg", 
                            debias: bool = True, debias_n_pcs: int = 2, debias_strength: float = 1.0,
                             _tqdm: bool = True) -> np.ndarray:
    """
    Extract contextual embeddings for all tokens in the given sentences.
    """
    tokens_text: List[str] = []
    tokens_embs: List[torch.Tensor] = []
    tokens_origin: List[Tuple[int, str]] = []
    
    iterator = range(len(sentences))
    if _tqdm:
        iterator = tqdm(iterator, desc="Getting token embeddings", total=len(sentences))
    for i in iterator:
        sentence = sentences[i]
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  
        outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = get_layer_embedding(outputs.hidden_states, layers_idxs, method=embeddings_method).squeeze(0) # [seq_len, hidden_dim]

        # tok_strs = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])# list of strings
        tok_strs = [tokenizer.decode(id) for id in inputs['input_ids'][0]]
        seen = set()
        keep = []
        for idx, tok in enumerate(tok_strs):
            tok_lower = tok.strip().strip("Ġ").lower()
            if (
                tok_lower not in tokenizer.all_special_tokens 
                and bool(tok_lower)
                and tok_lower not in EXCLUDED_TOKENS
                and tok_lower not in seen
                and not re.fullmatch(r'[^\w\s]+', tok_lower) 
            ):
                keep.append((idx, tok))
                seen.add(tok_lower)

        idxs, toks = zip(*keep)
        tok_embs = hidden_states[list(idxs)].cpu() # [num_toks, hidden_dim]

        for j, (tok, emb) in enumerate(zip(toks, tok_embs)):
            tokens_text.append(tok) #! tok.strip()
            tokens_embs.append(emb)
            tokens_origin.append((idxs[j], toks[j]))
    

    tokens_embs = torch.stack(tokens_embs)
    if debias:
        tokens_embs = debias_embeddings(tokens_embs, n_pcs=debias_n_pcs, strength=debias_strength)
    return tokens_text, tokens_embs, tokens_origin


@ torch.no_grad()
def get_token_embedding(token: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str) -> np.ndarray:
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
        
        inputs = tokenizer(
            sentence, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        
        # Find the position of our target token
        input_ids = inputs['input_ids'][0]
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        
        # Find the token position (skip special tokens)
        token_positions = []
        for i, tid in enumerate(input_ids):
            if tid not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
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

def find_cluster_representatives_sentences(agglomerative_labels, test_predictions, test_embeddings_umap, top_k_sentence, homogeneity_threshold):
    """
    Find the cluster representatives and the filtered clusters
    Args:
    agglomerative_labels: the labels of the clusters
    test_predictions: the predictions of the sentences
    test_embeddings_umap: the embeddings of the sentences
    top_k_sentence: the number of sentences to be selected as representatives
    homogeneity_threshold: the threshold of the homogeneity

    Returns:
    cluster_data: the data of the clusters
    cluster_centroids: the centroids of the clusters
    cluster_representatives: the top k representatives sentences of the clusters
    filtered_clusters_idxs: the indices of the filtered clusters by homogeneity
    """
    cluster_data = {}
    cluster_centroids = {}
    cluster_representatives = {}
    filtered_clusters_idxs = []
    
    for label in np.unique(agglomerative_labels):
        if label != -1:
            cluster_indices = np.where(agglomerative_labels == label)[0]
            cluster_data[label] = cluster_indices

            cluster_predictions = [test_predictions[i][0] for i in cluster_indices]
            majority_class = max(set(cluster_predictions), key=cluster_predictions.count)
            homogeneity = cluster_predictions.count(majority_class) / len(cluster_predictions)
            print(f"Cluster {label} homogeneity: {homogeneity}")
            if homogeneity > homogeneity_threshold:
                filtered_clusters_idxs.append(label)

            cluster_embeddings = test_embeddings_umap[cluster_indices]
            # cluster_embeddings = test_embeddings[cluster_indices]
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            cluster_centroids[label] = cluster_centroid
            distances = np.linalg.norm(cluster_embeddings - cluster_centroid, axis=1)
            nearest_indices = np.argsort(distances)[:top_k_sentence]
            cluster_representatives[label] = [cluster_indices[i] for i in nearest_indices]
            
    return cluster_data, cluster_centroids, cluster_representatives, filtered_clusters_idxs

# find representative token in each cluster
# for c in each cluster
# for token in each sentence
# get token embeddings by extracting the token dimension from sentence embedding
# calculate the disance between the token embedding and the cluster centroid
# get top k tokens that are closest to the centroid
class RepresentativeTokenFinder:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str, data: List[str],
                sentence_embeddings: np.ndarray, cluster_data: Dict[int, List[int]], cluster_representatives: Dict[int, List[int]], filtered_clusters: List[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device  
        self.data = data
        self.sentence_embeddings = sentence_embeddings
        self.cluster_data = cluster_data
        self.cluster_representatives = cluster_representatives
        self.filtered_clusters = filtered_clusters

    @ torch.no_grad()
    def rank_tokens_for_cluster(self, embeddings_umap, agglomerative_labels, umap_model, cluster_centroids_embs,
                            top_k, debias=True, n_pcs=2, 
                            strength=0.1, margin_type="min"):
        out: Dict[int, List[Dict[str, float]]] = {}
        # for cluster_index in np.unique(agglomerative_labels):
        #     cluster_indices = np.where(agglomerative_labels == cluster_index)[0]  #
        #     scores = self.rank_tokens_per_cluster(embeddings_umap, agglomerative_labels, cluster_index, cluster_indices, umap_model, top_k)
        #     out[cluster_index] = [{"token": s.text, "sim_self": s.sim_self, "sim_margin": s.sim_margin} for s in scores]
        
        cluster_centroids_raw = {}
        for cluster_index in np.unique(agglomerative_labels):
            # cluster_indices = self.cluster_representatives[cluster_index]
            cluster_indices = self.cluster_data[cluster_index]
            cluster_embeddings_raw = self.sentence_embeddings[cluster_indices]
            if debias:
                cluster_embeddings = debias_embeddings(cluster_embeddings_raw, n_pcs=n_pcs, strength=strength)
            else:
                cluster_embeddings = cluster_embeddings_raw
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            cluster_centroids_raw[cluster_index] = cluster_centroid
        
        for cluster_index in self.filtered_clusters:
            scores = self.rank_tokens_per_cluster(embeddings_umap, agglomerative_labels, cluster_index, cluster_centroids_embs, umap_model, 
                                                top_k, debias, n_pcs, strength, margin_type)
            out[cluster_index] = [{"token": s.text, "sim": s.sim, "margin": s.margin} for s in scores]
        return out

    @ torch.no_grad()
    def rank_tokens_per_cluster(self, embeddings_umap, agglomerative_labels, cluster_index, cluster_centroids_embs, umap_model, 
                                top_k, debias, n_pcs, strength, margin_type):
        # 1) sentence embeddings and cluster center
        # cluster_centroid = cluster_centroids[cluster_index]
        # cluster_centroid_norm = cluster_centroid / np.linalg.norm(cluster_centroid)
        # cluster_centroid_umap = umap_model.transform(cluster_centroid_norm.reshape(1, -1)).squeeze(0)
        # cluster_centroid_umap = cluster_centroid_umap / np.linalg.norm(cluster_centroid_umap)

        centroid_umap = cluster_centroids_embs[cluster_index]
        centroid_umap_norm = centroid_umap / np.linalg.norm(centroid_umap)

        
        # 2) token embeddings from these sentences
        tok_texts, tok_embs, _ = extract_token_embeddings([self.data[i] for i in self.cluster_representatives[cluster_index]], 
                                                          self.model, self.tokenizer, self.device, [-2, -1], debias=debias, debias_n_pcs=n_pcs, debias_strength=strength, )
        

        # 3) calculate a margin score which defined as score(token) = cos(token_emb, centroid_self) - avg_{other_clusters} cos(token_emb, centroid_other)
        # L2 normalize tok_embs and cluster_centroid
        tok_embs_norm = tok_embs / np.linalg.norm(tok_embs, axis=1, keepdims=True)
        tok_embs_umap = umap_model.transform(tok_embs_norm) # (tok_num, 2)
        tok_embs_umap_norm = tok_embs_umap / np.linalg.norm(tok_embs_umap, axis=1, keepdims=True)
        
        # Calculate similarity with current cluster centroid
        self_sims = tok_embs_umap_norm @ centroid_umap_norm  # Cleaner syntax
        # self_sims = torch.tensor(tok_embs_umap @ cluster_centroid_umap)
        
        other_sims_list = []
        for other_cluster_index in np.unique(agglomerative_labels):
            if other_cluster_index == cluster_index:
                continue
            # other_cluster_centroid = cluster_centroids[other_cluster_index]
            # other_cluster_centroid_norm = other_cluster_centroid / np.linalg.norm(other_cluster_centroid)
            # other_cluster_centroid_umap = umap_model.transform(other_cluster_centroid_norm.reshape(1, -1)).squeeze(0)
            # other_cluster_centroid_umap = other_cluster_centroid_umap / np.linalg.norm(other_cluster_centroid_umap)

            other_cluster_centroid = cluster_centroids_embs[other_cluster_index]
            other_cluster_centroid_norm = other_cluster_centroid / np.linalg.norm(other_cluster_centroid)
            
            # Calculate cosine similarity between token embeddings and other cluster centroid
            other_sims = tok_embs_umap_norm @ other_cluster_centroid_norm  # Cleaner syntax
            # other_sims = torch.tensor(tok_embs_umap @ other_cluster_centroid_umap)
            other_sims_list.append(other_sims)
        other_sims_list = np.array(other_sims_list)
        if margin_type == "max":
            out_other_sims = np.max(other_sims_list, axis=0)
        elif margin_type == "avg":
            out_other_sims = np.mean(other_sims_list, axis=0)
        elif margin_type == "min":
            out_other_sims = np.min(other_sims_list, axis=0)
        margin = self_sims - out_other_sims

        # 4) initial ranking
        idxs = np.argsort(margin)[::-1]
        # idxs = np.argsort(self_sims)[::-1]
        # for i in idxs:
        #     if 'book' in tok_texts[i]:
        #         print(tok_texts[i], self_sims[i].item(), margin[i].item())
        print('--------------------------------')

        return [TokenScore(text=tok_texts[i], sim=self_sims[i].item(), margin=margin[i].item()) for i in idxs[:top_k]]


class RepTokenFinder:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str, data: List[str], cluster_components: Dict[int, List[int]]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data = data
        self.cluster_components = cluster_components
    
    
    def rank_tokens_for_cluster(self, top_k: int = 100):
        """
        Rank tokens for each cluster by calculating a score: 
        score = the avg{L2_norm(Key(token) - Key(other_token))}
        where other_token is the token in other clusters
        """
        toks_each_cluster = {}
        keys_emb_each_cluster = {}

        # 1）: Collect tokens and embeddings per cluster
        for c, sentence_indices in self.cluster_components.items():
            all_toks, all_embs = [], []
            for sent_idx in sentence_indices:
                sent_text = self.data[sent_idx]
                key_embs, key_toks = self.get_token_keys_from_sent(sent_text)
                all_toks.extend(key_toks)
                all_embs.append(key_embs)
            toks_each_cluster[c] = all_toks
            keys_emb_each_cluster[c] = torch.cat(all_embs, dim=0)  # (tok_num, hidden_dim)
            print(keys_emb_each_cluster[c].shape)
        # assert False

       # 2） calculate the score for each token
        scores_each_cluster = {}
        for c, tokens in toks_each_cluster.items():
            current_embs = keys_emb_each_cluster[c]  
            other_embs = torch.cat([keys_emb_each_cluster[i] for i in range(len(keys_emb_each_cluster)) if i != c], dim=0)

            def avg_l2_distance(a, b, chunk_size=1024):
                n = a.size(0)
                total = torch.zeros(n, dtype=torch.float32) # cpu
                for start in tqdm(range(0, b.size(0), chunk_size), desc=f"Calculating average L2 distance for cluster {c}"):
                    end = start + chunk_size
                    d = torch.cdist(a, b[start:end], p=2)  # (n, chunk)
                    total += d.cpu().sum(dim=1)
                return total / b.size(0)  # average distance
            # distances = torch.cdist(current_embs, other_embs, p=2)
            # scores = distances.mean(dim=1)
            scores = avg_l2_distance(current_embs, other_embs)

            token_scores_pairs = [
                (tokens[idx], score) for idx, score in enumerate(scores.tolist())
            ]
            scores_each_cluster[c] = sorted(token_scores_pairs, key=lambda x: x[1], reverse=True)[:top_k]
        return scores_each_cluster

    @ torch.no_grad()
    def get_token_keys_from_sent(self, sent_text):
        """From a given sentence, get the token embeddings from the Key of last attention block
        Args:
            sent_text: the text of the sentence

        Returns:
            key_embs: the embeddings of the tokens in the Key of last attention block
            key_toks: the tokens in the Key of last attention block
        """
        # 1）tokenize
        encoding = self.tokenizer(
            sent_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 2) get last attention weights
        outputs = self.model(**encoding, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hs_last_input = hidden_states[-2]

        key_proj = self.model.roberta.encoder.layer[-1].attention.self.key
        
        # 3) get the token embeddings from the Key of last attention block
        key_embs = F.linear(hs_last_input, key_proj.weight, key_proj.bias)[0][1:-1,:]
        
        # Reshape to multi-head format and average over attention heads
        seq_len, hidden_dim = key_embs.shape  # [seq_len, hidden_dim]
        num_heads = self.model.roberta.encoder.layer[-1].attention.self.num_attention_heads
        head_dim = hidden_dim // num_heads
        
        # Reshape to [num_heads, seq_len, head_dim] and average over heads
        key_embs = key_embs.view(seq_len, num_heads, head_dim).transpose(0, 1)  # [num_heads, seq_len, head_dim]
        key_embs = key_embs.mean(dim=0)  # [seq_len, head_dim]
        
        tok_strs = [self.tokenizer.decode(id) for id in encoding["input_ids"][0][1:-1]]

        #4) remove tokens that are in exlude_list, punctuations, special tokens and empty tokens
        # Create a mask for valid tokens
        valid_mask = [
            tok.strip().lower() not in EXCLUDED_TOKENS 
            and tok.strip() not in self.tokenizer.all_special_tokens 
            and not re.fullmatch(r'[^\w\s]+', tok.strip())
            # and tok.strip() not in tok.strip()
            for tok in tok_strs
        ]
        # Filter tokens and embeddings using the mask
        tok_strs = [tok for i, tok in enumerate(tok_strs) if valid_mask[i]]
        key_embs = key_embs[valid_mask]
      
        return key_embs, tok_strs

     

class ShortcutSeedFinder:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str, data: List[str],
            cluster_representatives: Dict[int, List[int]], filtered_clusters_idxs: List[int],
            sim_threshold: float = 0.2, sensitivity_threshold: float = 0.5, alpha: float = 0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device  
        self.data = data
        self.cluster_representatives = cluster_representatives
        self.filtered_clusters_idxs = filtered_clusters_idxs
        self.sim_threshold = sim_threshold
        self.sensitivity_threshold = sensitivity_threshold
        self.alpha = alpha
    
    @ torch.no_grad()
    def ablate_token(self, tokens: List[str], sentences: List[int]):
        """
        ablate the token positions along the sequence length whose static embeddings are most similar to the candidate token
        Args:
            tokens: the candidate tokens
            sentences: the sentence indices in the cluster

        Returns:
            ablated_sentences: the ablated sentences
            ablated_indices: the indices of the ablated tokens
        """
        #1) get embedding of given token
        # print(f"token: {token}")
        token_ids = [self.tokenizer.encode(token, add_special_tokens=False)[0] for token in tokens]
        token_embs = [self.model.roberta.embeddings.word_embeddings.weight[token_id] for token_id in token_ids]
        #todo: change to mean embeddings of all subwords

        ablated_sentences = []
        ablated_indices = []
        for sent_idx in sentences:
            
            #2) get sentence embeddings of the sentence
            sent_text = self.data[sent_idx]
            encoded = self.tokenizer(sent_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = encoded["input_ids"][0]
            # if token_id not in input_ids:
            #     continue
            # print(token in sent_text)
            embeddings = self.model.roberta.embeddings.word_embeddings.weight[input_ids]

            #3) compute similarity between token and the target token
            sims = [torch.cosine_similarity(token_emb, embeddings, dim=1) for token_emb in token_embs]
            
            # if "book" in token or "movie" in token:
            #     print("--------------------------------")
            #     print(f"token: {token}")
            #     print(f"sent_idx: {sent_idx}")
            #     for i, sim in enumerate(sims):
            #         if sim > self.sim_threshold:
            #             print(f"token: {self.tokenizer.decode(input_ids[i])}, similarity: {sim}")

            #4) find the position of the token with the highest similarity
            matched_indices_all = []
            for token_emb, token, sim in zip(token_embs, tokens, sims):
                matched_indices = (sim > self.sim_threshold).nonzero(as_tuple=True)[0].tolist() 
                matched_indices_all.extend(matched_indices)
            # remove the duplicated indices
            matched_indices_all = list(set(matched_indices_all))
            # matched_indices = (sims > self.sim_threshold).nonzero(as_tuple=True)[0].tolist()
            # print(f"this sentence is: {sent_text}")
            # print(f"matched_indices: {matched_indices}, this token is: {self.tokenizer.decode(input_ids[matched_indices])}")
            if len(matched_indices_all) > 0:
                ablated_indices.append(sent_idx)

                #5) replace matched token with <mask>
                input_ids_ablated = input_ids.clone()
                for idx in matched_indices_all:
                    input_ids_ablated[idx] = self.tokenizer.mask_token_id

                ablated_sentence = self.tokenizer.decode(input_ids_ablated, skip_special_tokens=False)
                special_tokens_to_remove = [t for t in self.tokenizer.all_special_tokens if t != self.tokenizer.mask_token]
                for special_token in special_tokens_to_remove:
                    ablated_sentence = ablated_sentence.replace(special_token, '')
                
                # print(f"ablated_sentence: {ablated_sentence}")
                ablated_sentences.append(ablated_sentence)
        # assert False
        return ablated_sentences, ablated_indices

        

    def calculate_ablation_sensitivity(self, predictions, new_predictions):
        label_distribution = Counter([pred[0] for pred in predictions])
        new_label_distribution = Counter([pred[0] for pred in new_predictions])
        return np.abs(label_distribution[0] - new_label_distribution[0]) / sum(label_distribution.values())


    def calculate_prevalence(self, len_occur, num_sentences):
        return len_occur / num_sentences


    def find_shortcut_seed(self, token_candidates, cluster_sentences, predictions):
        seed_list = {}
        for cluster_idx in self.filtered_clusters_idxs:
            # tokens = [x["token"] for x in token_candidates[cluster_idx]]
            tokens = [x[0] for x in token_candidates[cluster_idx]]
            sentences = cluster_sentences[cluster_idx]
            for tok in tqdm(sorted(set(tokens)), desc=f"Finding shortcut seed for cluster {cluster_idx}"):
                tok_list = [tok]
                ablated_sentences, ablated_indices = self.ablate_token(tok_list, sentences)
                if len(ablated_sentences) == 0:
                    raise ValueError(f"No ablated sentences for token: {tok}")
                old_predictions = [predictions[i] for i in ablated_indices]
                new_predictions = get_batch_predictions(ablated_sentences, self.model, self.tokenizer, self.device, batch_size=128, _tqdm=False)
                assert len(old_predictions) == len(new_predictions)
                # option: check the new embeddings fall into which cluster
                sensitivity = self.calculate_ablation_sensitivity(old_predictions, new_predictions)
                prevalence = self.calculate_prevalence(len(ablated_sentences), len(sentences))
                uncertainty = self.alpha * sensitivity + (1 - self.alpha) * prevalence
                # print(f"Token: {tok}, ablated sentences: {len(ablated_sentences)},Sensitivity: {sensitivity}, Prevalence: {prevalence}, Uncertainty: {uncertainty}")
                if sensitivity > self.sensitivity_threshold:
                    seed_list[tok] = {
                        "sensitivity": sensitivity,
                        "prevalence": prevalence,
                        "uncertainty": uncertainty,
                        "cluster_idx": cluster_idx,
                    }
        return seed_list




            
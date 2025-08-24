import os, sys, argparse
curdir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(curdir)
from utils import *
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from datasets import load_dataset

# set random seeds
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/mnt/disk21/user/jiayili/peer_learning/results/biased_amazon_output/checkpoint-5835")
    parser.add_argument("--test_data_path", type=str, default="/mnt/disk21/user/jiayili/doNt-Forget-your-Language/data/test_downsampled/unbiased_amazon_test.csv")
    parser.add_argument("--use_hf_dataset", action="store_true", help="use huggingface dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_pcs", type=int, default=2)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--distance_threshold", type=float, default=0.5, help="distance threshold for agglomerative clustering")
    parser.add_argument("--top_k_sentence", type=int, default=20, help="top k sentences that near the centroid in each clusters")
    parser.add_argument("--top_k_token", type=int, default=100, help="top k tokens that are representative in each clusters")
    parser.add_argument("--homogeneity_threshold", type=float, default=0.7, help="homogeneity threshold for filtering clusters")
    parser.add_argument("--sensitivity_threshold", type=float, default=0.5, help="sensitivity threshold for finding shortcut seed")
    return parser.parse_args()


def main(args):
    model, tokenizer = load_model(args.checkpoint_path)
    model.to(args.device)

    if args.use_hf_dataset:
        dataset = load_dataset(args.test_data_path)
        test_sentences = dataset["test"]["text"]
        test_labels = dataset["test"]["label"]
    else:
        test_sentences, test_labels = load_test_data(args.test_data_path)
    test_predictions = get_batch_predictions(test_sentences, model, tokenizer, args.device, args.batch_size)

    test_embeddings = get_roberta_embeddings(test_sentences, model, tokenizer, args.device, args.batch_size)
    test_embeddings_norm = normalize(test_embeddings, axis=1)

    # apply umap for dimensionality reduction
    umap_model = umap.UMAP(metric='cosine', random_state=42, n_jobs=1) 
    test_embeddings_umap = umap_model.fit_transform(test_embeddings_norm)

    # apply agglomerative clustering
    print("Applying agglomerative clustering...")
    agglomerative_model = AgglomerativeClustering(n_clusters=None, distance_threshold=args.distance_threshold, linkage='single', metric='l2')
    agglomerative_clusters = agglomerative_model.fit(test_embeddings_umap)
    agglomerative_labels = agglomerative_clusters.labels_

    # if unique agglomerative_labels
    if len(set(agglomerative_labels)) == 1:
        raise ValueError("All sentences are in the same cluster, try to decrease the distance threshold")
    
    # find k=20 sentences that near the centroid in each clusters
    cluster_sent_idx, cluster_centroids_embs, cluster_representatives_sents, filtered_cluster_idxs = find_cluster_representatives_sentences(agglomerative_labels, 
                                                                                        test_predictions, test_embeddings_umap, 
                                                                                 args.top_k_sentence, args.homogeneity_threshold)                                                                            
    # print("--------------Sentences------------------")
    # for label, idxs in cluster_sent_idx.items():
    #     print("label", label)
    #     idx_count = 0
    #     movie_count = 0
    #     book_count = 0
    #     for idx in idxs:
    #         idx_count += 1
    #         if 'movie' in test_sentences[idx]:
    #             movie_count += 1
    #         if 'book' in test_sentences[idx]:
    #             book_count += 1
    #     print(f"idx_count: {idx_count}, movie_count: {movie_count}, book_count: {book_count}")
    # print("--------------Representatives------------------")
    # for label, idxs in cluster_representatives_sents.items():
    #     print("label", label)
    #     idx_count = 0
    #     movie_count = 0
    #     book_count = 0
    #     film_count = 0
    #     books_count = 0
    #     for idx in idxs:
    #         idx_count += 1
    #         if 'movie' in test_sentences[idx]:
    #             movie_count += 1
    #         if 'book' in test_sentences[idx]:
    #             book_count += 1
    #         if 'film' in test_sentences[idx]:
    #             film_count += 1
    #         if 'books' in test_sentences[idx]:
    #             books_count += 1
    #     print(f"idx_count: {idx_count}, movie_count: {movie_count}, film_count: {film_count}, book_count: {book_count}, books_count: {books_count}")
    
    # finder = RepresentativeTokenFinder(model, tokenizer, args.device, test_sentences, 
    #                                 test_embeddings, cluster_data, cluster_representatives, filtered_clusters_idxs)
    # token_candidates = finder.rank_tokens_for_cluster(test_embeddings_umap, agglomerative_labels, umap_model, cluster_centroids_embs,
    #                                                   top_k = args.top_k_token, debias=False, n_pcs=2, strength=0.1, margin_type="max")
    # finder = RepTokenFinder(model, tokenizer, args.device, test_sentences, cluster_sent_idx)
    finder = RepTokenFinder(model, tokenizer, args.device, test_sentences, cluster_representatives_sents)
    token_candidates = finder.rank_tokens_for_cluster()
  
    s_finder = ShortcutSeedFinder(model, tokenizer, args.device, test_sentences, cluster_representatives_sents, filtered_cluster_idxs, 
                            sensitivity_threshold = args.sensitivity_threshold)
    shortcut_seed = s_finder.find_shortcut_seed(token_candidates, cluster_representatives_sents, test_predictions)
    print("-----------------Results---------------")
    for s in shortcut_seed:
        print(s, shortcut_seed[s])

    # old accuracy
    old_accuracy = get_accuracy(test_predictions, test_labels)
    print(f"old accuracy: {old_accuracy}")

    new_predictions = test_predictions.copy()
    for cluster_idx in filtered_cluster_idxs:
        sentence_indices_in_cluster = cluster_sent_idx[cluster_idx]
        seeds = [s[0] for s in shortcut_seed.items() if s[1]['cluster_idx'] == cluster_idx]
        ablated_sentences, ablated_indices = s_finder.ablate_token(seeds, sentence_indices_in_cluster)
        ablated_predictions = get_batch_predictions(ablated_sentences, model, tokenizer, args.device, args.batch_size)
        for i, idx in enumerate(ablated_indices):
            new_predictions[idx] = ablated_predictions[i]

    new_accuracy = get_accuracy(new_predictions, test_labels)
    print(f"new accuracy: {new_accuracy}")



if __name__ == "__main__":
    args = get_args()
    main(args)
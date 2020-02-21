import argparse
import os
import h5py
import json
import random
import numpy as np
import itertools
import csv
from typing import Dict, Tuple, Sequence, List

from scipy.spatial.distance import cosine
from allennlp.common.tqdm import Tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
import pandas as pd

import spacy
nlp = spacy.load("en_core_web_sm")

# TODO(mm): Replace words by tokens


# TODO(mm): Calculate average similarity of a token t_0 to a particular token t_1 for different layers
#   - To which token is t_0 most similar? On which layer?

# TODO(mm): Calculate average similarit of a token t_0 to itself for different layers BUT
#   - From different samples, i.e compare average Obama from sentences that talk about A vs. average Obama from sentences that talk about B
#       - A and B could be relations?


def calculate_token_similarity_across_sentences(embedding_fn: str, token2sent_indexer: Dict[str, List[Tuple[int, int]]], out_fn: str, most_frequent=10, num_samples=1000) -> None:
    """
        Each word in token2sent_indexer appears in multiple sentences. Thus each occurrence of the token
        will have a unique embedding at each layer. For each layer, calculate the average cosine
        similarity between embeddings of the same token across its different occurrences. This captures
        how sentence-specific each token representation is at each layer.

        # most_frequent: consider only most_frequent most frequent tokens. If -1 consider all tokens

        Create a table of size (#tokens x #layers) and write it to out_fn.
        """

    f = h5py.File(embedding_fn, 'r')
    num_layers = f["0"].shape[0]

    # write statistics to csv file: one row per word, one column per layer
    fieldnames = ['token'] + \
        list(map(lambda w: 'layer_' + w, map(str, range(num_layers))))
    writer = csv.DictWriter(open(out_fn, 'w'), fieldnames=fieldnames)
    writer.writeheader()

    mean_self_similarity = {}

    tokens = list(token2sent_indexer.keys())[:most_frequent]

    for token in Tqdm.tqdm(tokens, desc=f'Looping over top {most_frequent} tokens'):

        # Ingore padding token. There are too many combinations
        if token in ['[PAD]', '<pad>', '!']:  # gpt-2 padding token was set to !
            continue

        # Ignore punctuation
        if nlp.vocab[token].is_punct:
            continue

        similarity_by_layer = {'token': token}

        # list of tuples (sentence index, index of tokens in sentence) for token occurrences
        occurrences = token2sent_indexer[token]

        indices = range(len(occurrences))
        index_pairs = [(i, j)
                       for i, j in itertools.product(indices, indices) if i != j]

        # for very frequent tokens (e.g., stopwords), there are too many pairwise comparisons
        # so for faster estimation, take a random sample of no more than num_samples pairs
        if len(index_pairs) > num_samples:
            index_pairs = random.sample(index_pairs, num_samples)

            # calculate statistic for each layer using sampled data
        mean_self_similarity[token] = []

        for layer in range(num_layers):
            layer_similarities = []

            for i, j in Tqdm.tqdm(index_pairs, desc=f'Looping over all pairs'):
                sent_index_i, token_in_sent_index_i = occurrences[i]
                sent_embedding_i = f[str(sent_index_i)]
                token_embedding_i = sent_embedding_i[layer,
                                                     token_in_sent_index_i]

                sent_index_j, token_in_sent_index_j = occurrences[j]
                sent_embedding_j = f[str(sent_index_j)]
                token_embedding_j = sent_embedding_j[layer,
                                                     token_in_sent_index_j]

            layer_similarities.append(
                1 - cosine(token_embedding_i, token_embedding_j))

            mean_layer_similarity = round(np.nanmean(layer_similarities), 3)
            similarity_by_layer[f'layer_{layer}'] = mean_layer_similarity

            mean_self_similarity[token].append(mean_layer_similarity)

        writer.writerow(similarity_by_layer)

    print(f'Saved similarity scores to: {out_fn}')

    return mean_self_similarity


def variance_explained_by_pc(
        embedding_fn: str,
        word2sent_indexer: Dict[str, List[Tuple[int, int]]],
        variance_explained_fn: str,
        pc_fn: str) -> None:
    """
    Each word in word2sent_indexer appears in multiple sentences. Thus each occurrence of the word
    will have a different embedding at each layer. How much of the variance in these occurrence
    embeddings can be explained by the first principal component? In other words, to what extent
    can these different occurrence embeddings be replaced by a single, static word embedding?

    Create a table of size (#words x #layers) and write the variance explained to variance_explained_fn.
    Write the first principal component for each word to pc_fn + str(layer_index), where each row
    starts with a word followed by space-separated numbers.
    """
    f = h5py.File(embedding_fn, 'r')
    num_layers = f["0"].shape[0]

    # write statistics to csv file: one row per word, one column per layer
    # excluding first layer, since we don't expect the input embeddings to be the same at all for gpt2/bert
    # and we expect them to be identical for elmo
    fieldnames = ['word'] + \
        list(map(lambda w: 'layer_' + w, map(str, range(1, num_layers))))
    writer = csv.DictWriter(
        open(variance_explained_fn, 'w'), fieldnames=fieldnames)
    writer.writeheader()

    # files to write the principal components to
    pc_vector_files = {layer: open(pc_fn + str(layer), 'w')
                       for layer in range(1, num_layers)}

    for word in Tqdm.tqdm(word2sent_indexer):
        variance_explained = {'word': word}

        # calculate variance explained by the first principal component
        for layer in range(1, num_layers):
            embeddings = [f[str(sent_index)][layer, word_index].tolist() for sent_index, word_index
                          in word2sent_indexer[word] if f[str(sent_index)][layer, word_index].shape != ()]

            pca = PCA(n_components=1)
            pca.fit(embeddings)

            variance_explained[f'layer_{layer}'] = min(
                1.0, round(pca.explained_variance_ratio_[0], 3))
            pc_vector_files[layer].write(
                ' '.join([word] + list(map(str, pca.components_[0]))) + '\n')

        writer.writerow(variance_explained)


def explore_embedding_space(embedding_fn: str, out_fn: str, pooler='mean', num_samples=1000) -> None:
    """
    Calculate the following statistics for each layer of the model:
    1. mean cosine similarity between a sentence and its words
    2. mean dot product between a sentence and its words
    3. mean word embedding norm
    4. mean cosine similarity between randomly sampled words
    5. mean dot product between randomly sampled words
    6. mean variance explained by first principal component for a random sample of words

    num_samples sentences/words are used to estimate each of these metrics. We randomly sample words
    by first uniformly randomly sampling sentences and then uniformly randomly sampling a single word
    from each sampled sentence. This is because:
            - 	When we say we are interested in the similarity between random words, what we really
                    mean is the similarity between random _word occurrences_ (since each word has a unique
                    vector based on its context).
            - 	By explicitly sampling from different contexts, we avoid running into cases where two
                    words are similar due to sharing the same context.

    Create a dictionary mapping each layer to a dictionary of the statistics write it to out_fn.
    """

    f = h5py.File(embedding_fn, 'r')
    num_layers = f["0"].shape[0]
    num_sentences = len(f)

    # TODO(mm): Consider a specific subset of sentences

    # take are random sample of sentences
    sentence_indices = random.sample(list(range(num_sentences)), num_samples)
    print(f'Considering {len(sentence_indices)} random sentences')

    # 1. mean cosine similarity between a sentence and its words
    mean_cos_sim_between_sent_and_words = {
        f'layer_{layer}': [] for layer in range(num_layers)}

    # 4. mean cosine similarity between randomly sampled words
    mean_cos_sim_across_words = {
        f'layer_{layer}': -1 for layer in range(num_layers)}

    # 3. mean word embedding norm
    word_norm_std = {f'layer_{layer}': -1 for layer in range(num_layers)}
    word_norm_mean = {f'layer_{layer}': -1 for layer in range(num_layers)}

    # 6. mean variance explained by first principal component for a random sample of words
    variance_explained_random = {
        f'layer_{layer}': -1 for layer in range(num_layers)}

    for layer in Tqdm.tqdm(range(num_layers), desc='Looping over layers'):
        word_vectors = []  # collect 1 random token from each sentence
        word_norms = []
        mean_cos_sims = []
        mean_dot_products = []

        for sent_index in Tqdm.tqdm(sentence_indices, desc='Looping sentences'):
            # loop over the randomly selected sentences to compute
            # 1. mean cosine similarity between a sentence and its tokens
            # 3. mean word embedding norm for every token in the sentence

            # average word vectors to get sentence vector
            layer_token_embeddings = f[str(sent_index)][layer]
            # compute sentence embedding
            if pooler == 'mean':
                sentence_vector = layer_token_embeddings.mean(axis=0)
            elif pooler == 'cls':
                # take embedding of first token
                sentence_vector = layer_token_embeddings[0]
            else:
                raise KeyError(pooler)

            num_words = layer_token_embeddings.shape[0]

            # TODO(mm): Consider specific tokens in the sentence
            # TODO(mm): Exclude special tokens from here, especially padding
            # TODO(mm): We need the token index file here to access specific tokens *************************

            # randomly add a word vector (not all of them, because that would bias towards longer sentences)
            # choose a random token vector from the sentence

            # TODO(mm): Quick fix: For now randomly sample one of the first 5 firsts
            token_vector = layer_token_embeddings[random.choice(
                list(range(1, 5)))]
            word_vectors.append(token_vector)

            # compute the mean cosine similarity between the sentence and its tokens
            mean_cos_sim = np.nanmean([1 - cosine(f[str(sent_index)][layer, i], sentence_vector)
                                       for i in range(num_words) if f[str(sent_index)][layer, i].shape != ()])
            mean_cos_sims.append(round(mean_cos_sim, 3))

            # compute the mean embedding norm across all tokens in the sentence
            word_norms.extend([np.linalg.norm(f[str(sent_index)][layer, i])
                               for i in range(num_words)])

        # 1. mean cosine similarity between a sentence and its tokens
        mean_cos_sim_between_sent_and_words[f'layer_{layer}'] = round(
            float(np.mean(mean_cos_sims)), 3)

        # 4. mean cosine similarity between randomly sampled tokens (1 token per sentence)
        # TODO(mm): How to compute std here?

        mean_cos_sim_across_words[f'layer_{layer}'] = round(np.nanmean([1 - cosine(random.choice(word_vectors),
                                                                                   random.choice(word_vectors)) for _ in range(num_samples)]), 3)

        # 3. mean and std  word embedding norm
        word_norm_std[f'layer_{layer}'] = round(float(np.std(word_norms)), 3)
        word_norm_mean[f'layer_{layer}'] = round(float(np.mean(word_norms)), 3)

        # how much of the variance in randomly chosen words can be explained by their first n_components principal component?
        pca = TruncatedSVD(n_components=100)
        pca.fit(word_vectors)
        variance_explained_random[f'layer_{layer}'] = min(
            1.0, round(float(pca.explained_variance_ratio_[0]), 3))

    json.dump({
        'pooler': pooler,
        'n_random_stences': num_samples,
        'mean cosine similarity between sentence and words': mean_cos_sim_between_sent_and_words,
        'mean cosine similarity across words': mean_cos_sim_across_words,
        'word norm std': word_norm_std,
        'word norm mean': word_norm_mean,
        'variance explained for random words': variance_explained_random
    }, open(out_fn, 'w'), indent=1)


if __name__ == "__main__":
    # where the contextualized embeddings are saved (in HDF5 format)
    # EMBEDDINGS_PATH = "/datasets/WikipediaWikidataDistantSupervisionAnnotations/entities-in-context"

    # for model in ["elmo", "bert", "gpt2"]:
    # 	print(f"Analyzing {model} ...")

    # 	word2sent_indexer = json.load(open(f'{model}/word2sent.json', 'r'))
    # 	scores = json.load(open(f'{model}/scores.json', 'r'))
    # 	EMBEDDINGS_FULL_PATH = os.path.join(EMBEDDINGS_PATH, f'{model}.hdf5')

    # 	print(f"Analyzing word similarity across sentences ...")
    # 	calculate_word_similarity_across_sentences(EMBEDDINGS_FULL_PATH, word2sent_indexer,
    # 		f'{model}/self_similarity.csv')

    # 	print(f"Analyzing variance explained by first principal component ...")
    # 	variance_explained_by_pc(EMBEDDINGS_FULL_PATH, word2sent_indexer,
    # 		f'{model}/variance_explained.csv', os.path.join(EMBEDDINGS_PATH, f'pcs/{model}.pc.'))

    # 	print(f"Exploring embedding space ...")
    # 	explore_embedding_space(EMBEDDINGS_FULL_PATH, f'{model}/embedding_space_stats.json')

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str,
                        # default="/datasets/WikipediaWikidataDistantSupervisionAnnotations/entities-in-context/embeddings",
                        default="/datasets/WiC/preprocessed/embeddings",
                        help="The dir holding the embedding and index files.")

    parser.add_argument("--output_dir", type=str,
                        # default="/datasets/WikipediaWikidataDistantSupervisionAnnotations/entities-in-context/embeddings",
                        default="/datasets/WiC/preprocessed/embeddings",
                        help="The output dir where to store embedding space stats.")

    parser.add_argument("--index_file", type=str,
                        # default="Q76.tsv_bert-base-cased_index.json",
                        default="dev.tsv_bert-base-cased_index.json",
                        help="The name of the index .csv file.")

    parser.add_argument("--embeddings_file", type=str,
                        # default="Q76.tsv_bert-large-uncased_embeddings.hdf5",
                        default="dev.tsv_bert-base-cased_embeddings.hdf5",
                        help="The name of the embeddings .hdf5 file.")

    parser.add_argument("--pooler", type=str,
                        default="mean",
                        choices=['cls', 'mean'],
                        help="How to compute sentence embeddings.")

    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Random seed.")

    parser.add_argument("--n_random_sentences", type=int,
                        default=100,
                        help="How many random sentences should be considered when computing embedding space stats.")

    parser.add_argument("--do_analyze", action='store_true')

    parser.add_argument("--do_explore", action='store_true')

    args = parser.parse_args()

    # Fix random seeds (this is important since we randomly sample sentences and tokens from them)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load input files
    index_file = os.path.join(args.input_dir, args.index_file)
    embeddings_file = os.path.join(args.input_dir, args.embeddings_file)
    token2sent_indexer = json.load(open(index_file, 'r'))
    print(f'Token index contains {len(token2sent_indexer.keys())} tokens')

    if args.do_analyze:
        print(f"Analyzing word similarity across sentences ...")
        output_file = embeddings_file.split('.hdf5')[0] + '_self_similarity.csv'
        mean_self_similarity = calculate_token_similarity_across_sentences(
            embeddings_file, token2sent_indexer, out_fn=output_file, most_frequent=-1)

    if args.do_explore:
        print(f"Exploring embedding space ...")
        output_file = embeddings_file.split(
            '.hdf5')[0] + f'_embedding_stats_{args.pooler}.json'
        explore_embedding_space(embeddings_file, output_file,
                                pooler=args.pooler, num_samples=args.n_random_sentences)

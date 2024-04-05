import csv
import json
import itertools
import random
from typing import Union, Callable

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ########################## PART 1: PROVIDED CODE ##############################
def load_datasets(data_directory: str) -> "Union[dict, dict]":
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize_w2v(
        text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
        word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[: max_words - 1]

    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
        tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


def collate_cbow(batch):
    """
    Collate function for the CBOW model. This is needed only for CBOW but not skip-gram, since
    skip-gram indices can be directly formatted by DataLoader. For more information, look at the
    usage at the end of this file.
    """
    sources = []
    targets = []

    for s, t in batch:
        sources.append(s)
        targets.append(t)

    sources = torch.tensor(sources, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)

    return sources, targets


def train_w2v(model, optimizer, loader, device):
    """
    Code to train the model. See usage at the end.
    """
    model.train()

    for x, y in tqdm(loader, miniters=20, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)

        loss = F.cross_entropy(y_pred, y)
        loss.backward()

        optimizer.step()

    return loss


class Word2VecDataset(torch.utils.data.Dataset):
    """
    Dataset is needed in order to use the DataLoader. See usage at the end.
    """

    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets
        assert len(self.sources) == len(self.targets)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]


# ########################## PART 2: PROVIDED CODE ##############################
def load_glove_embeddings(file_path: str) -> "dict[str, np.ndarray]":
    """
    Loads trained GloVe embeddings downloaded from:
        https://nlp.stanford.edu/projects/glove/
    """
    word_to_embedding = {}
    with open(file_path, "r") as f:
        for line in f:
            word, raw_embeddings = line.split()[0], line.split()[1:]
            embedding = np.array(raw_embeddings, dtype=np.float64)
            word_to_embedding[word] = embedding
    return word_to_embedding


def load_professions(file_path: str) -> "list[str]":
    """
    Loads profession words from the BEC-Pro dataset. For more information on BEC-Pro,
    see:
        https://arxiv.org/abs/2010.14534
    """
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip the header.
        professions = [row[1] for row in reader]
    return professions


def load_gender_attribute_words(file_path: str) -> "list[list[str]]":
    """
    Loads the gender attribute words from: https://aclanthology.org/N18-2003/
    """
    with open(file_path, "r") as f:
        gender_attribute_words = json.load(f)
    return gender_attribute_words


def compute_partitions(XY: "list[str]") -> "list[tuple]":
    """
    Computes all of the possible partitions of X union Y into equal sized sets.

    Parameters
    ----------
    XY: list of strings
        The list of all target words.

    Returns
    -------
    list of tuples of strings
        List containing all of the possible partitions of X union Y into equal sized
        sets.
    """
    return list(itertools.combinations(XY, len(XY) // 2))


def p_value_permutation_test(
        X: "list[str]",
        Y: "list[str]",
        A: "list[str]",
        B: "list[str]",
        word_to_embedding: "dict[str, np.array]",
) -> float:
    """
    Computes the p-value for a permutation test on the WEAT test statistic.

    Parameters
    ----------
    X: list of strings
        List of target words.
    Y: list of strings
        List of target words.
    A: list of strings
        List of attribute words.
    B: list of strings
        List of attribute words.
    word_to_embedding: dict of {str: np.array}
        Dict containing the loaded GloVe embeddings. The dict maps from words
        (e.g., 'the') to corresponding embeddings.

    Returns
    -------
    float
        The computed p-value for the permutation test.
    """
    # Compute the actual test statistic.
    s = weat_differential_association(X, Y, A, B, word_to_embedding, weat_association)

    XY = X + Y
    partitions = compute_partitions(XY)

    total = 0
    total_true = 0
    for X_i in partitions:
        # Compute the complement set.
        Y_i = [w for w in XY if w not in X_i]

        s_i = weat_differential_association(
            X_i, Y_i, A, B, word_to_embedding, weat_association
        )

        if s_i > s:
            total_true += 1
        total += 1

    p = total_true / total

    return p


# ######################## PART 1: YOUR WORK STARTS HERE ########################


def build_current_surrounding_pairs(indices: "list[int]", window_size: int = 2):
    """
    A helper function that is applied to each sample in order to construct pairs of current word and
    its surrounding words. This will be used later in the assignment.
    Concretely, this function pairs up each current word w(t) with its surrounding words (...w(t+2),
    w(t+1), w(t-1), (w(t-2)...), with respect to a window size.
    Note: To ensure that current_indices has a consistent inner length, you will need to omit
    the start and end of the sample_indices list from the surrounding since it would otherwise have
    unbalanced sides.
    :param indices:
    :param window_size:
    :return:
    >>> text = "dogs and cats are playing".split()
    >>> surroundings, currents = build_current_surrounding_pairs(text, window_size=1)
    >>> print(currents)
    ['and', 'cats', 'are']
    >>> print(surroundings)
    [['dogs', 'cats'], ['and', 'are'], ['cats', 'playing']]
    >>> indices = [word_to_index[t] for t in text]
    >>> surroundings, currents = build_current_surrounding_pairs(indices, window_size=1)
    >>> print(currents)
    [3, 4887, 11]
    >>> print(surroundings)
    [[110, 4887], [3, 11], [4887, 31]]
    """
    surroundings = []
    currents = indices[window_size:-window_size]
    for ix, current in zip(range(window_size, len(indices) - window_size), currents):
        surrounding = indices[ix - window_size: ix] + indices[ix + 1: ix + window_size + 1]
        surroundings.append(surrounding)
    return surroundings, currents


def expand_surrounding_words(ix_surroundings: "list[list[int]]", ix_current: "list[int]"):
    """
    A helper function used in Skip-Gram to expand a list of surroundings into pairs of a single
    surrounding with the target (the latter will be repeated).
    Using the output of build_current_surrounding_pairs, convert the surrounding into
    pairs of context-target pair. The resulting lists should be longer.
    :param ix_surroundings:
    :param ix_current:
    :return:
    >>> # dogs and cats are playing
    >>> surroundings = [['dogs', 'cats'], ['and', 'are'], ['cats', 'playing']]
    >>> currents = ['and', 'cats', 'are']
    >>> surrounding_expanded, current_expanded =expand_surrounding_words(surroundings, currents)
    >>> print(surrounding_expanded)
    ['dogs', 'cats', 'and', 'are', cats', 'playing']
    >>> print(current_expanded)
    ['and', 'and', 'cats', 'cats', 'are', 'are']
    >>> ix_surroundings =[[110, 4887], [3, 11], [4887, 31]]
    >>> ix_currents = [3, 4887, 11]
    >>> ix_surr_expanded, ix_curr_expanded = expand_surrounding_words(ix_surroundings, ix_currents)
    >>> print(ix_surr_expanded)
    [110, 4887, 3, 11, 4887, 31]
    >>> print(ix_curr_expanded)
    [3, 3, 4887, 4887, 11, 11]
    """
    ix_surroundings_expanded = [surrounding for surroundings in ix_surroundings for surrounding in surroundings]
    ix_current_expanded = [current for current in ix_current for _ in range(len(ix_surroundings[0]))]
    return ix_surroundings_expanded, ix_current_expanded


def cbow_preprocessing(indices_list: "list[list[int]]", window_size: int = 2):
    """
    Preprocess the dataset to be ready for CBOW to use for training.
    Concretely, use the build_current_surrounding_pairs function you implemented
    above to complete this function. The difference is that the input is a list of indices (so a nested
    list), but the output format should be the same.
    :param indices_list:
    :param window_size:
    :return:
    """
    surroundings = []
    currents = []
    for indices in indices_list:
        surr, curr = build_current_surrounding_pairs(indices, window_size)
        surroundings.extend(surr)
        currents.extend(curr)
    return surroundings, currents


def skipgram_preprocessing(indices_list: "list[list[int]]", window_size: int = 2):
    """
    Preprocess the dataset to be ready for Skip-Gram to use for training.
    Concretely, use the build_current_surrounding_pairs function you implemented
    above to complete this function. The difference is that the input is a list of indices (so a nested
    list), but the output format should be the same.
    Note: Here, you need to return all possible pairs between a word w(t) and its surroundings. In
    the paper, it’s a sampling method based on distance, but we will not do that for simplicity,
    instead we’ll just use everything.
    :param indices_list:
    :param window_size:
    :return:
    """
    surroundings = []
    currents = []
    for indices in indices_list:
        surr, curr = build_current_surrounding_pairs(indices, window_size)
        surr_expanded, curr_expanded = expand_surrounding_words(surr, curr)
        surroundings.extend(surr_expanded)
        currents.extend(curr_expanded)
    return surroundings, currents

# Build CBOW and Skip-Gram as torch modules, and train them using the provided train
# function. You will need to write the following classes:
# ●SharedNNLM: A helper class (not a nn.Module) that loads the embedding and project
# layers and binds the weights (such that the weights are shared).
# ●SkipGram: The Skip-Gram model. It will use SharedNNLM. You only need to implement
# the forward pass.
# ●CBOW: The CBOW model. It will use SharedNNLM. You only need to implement the
# forward pass.


class SharedNNLM:
    def __init__(self, num_words: int, embed_dim: int):
        """
        SkipGram and CBOW actually use the same underlying architecture,
        which is a simplification of the NNLM model (no hidden layer)
        and the input and output layers share the same weights. You will
        need to implement this here.

        Notes
        -----
          - This is not a nn.Module, it's an intermediate class used
            solely in the SkipGram and CBOW modules later.
          - Projection does not have a bias in word2vec
        Initializes SharedNNLM model. This class will be used by SkipGram and CBOW. This class is a
        simplification of the NNLM model (no hidden layer) and the input and output layers share the
        same weights. The projection in word2vec does not have a bias. Your __init__ function
        should initialize the model’s embeddings and create a projection layer (which does not have a
        bias term).
        """

        # TODO vvvvvv
        self.embedding = nn.Embedding(num_words, embed_dim)
        self.projection = nn.Linear(embed_dim, num_words, bias=False)

        # TODO ^^^^^
        self.bind_weights()

    def bind_weights(self):
        """
        Bind the weights of the embedding layer with the projection layer.
        This mean they are the same object (and are updated together when
        you do the backward pass).
        """
        emb = self.get_emb()
        proj = self.get_proj()

        proj.weight = emb.weight

    def get_emb(self):
        return self.embedding

    def get_proj(self):
        return self.projection


class SkipGram(nn.Module):
    """
    Use SharedNNLM to implement skip-gram. Only the forward() method differs from CBOW.
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        """
        Executes the forward pass for the SkipGram model. Given the index of a target word w(t), this
        function returns predicted distribution of the index of a surrounding word.
        :param x:
        :return:
        """
        x = self.emb(x)
        x = self.proj(x)
        return x


class CBOW(nn.Module):
    """
    Use SharedNNLM to implement CBOW. Only the forward() method differs from SkipGram,
    as you have to sum up the embedding of all the surrounding words (see paper for details).
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        """
        Executes the forward pass for the CBOW model. Given the indices of the surrounding words,
        this function returns the predicted distribution of the index of w(t).
        :param x:
        :return:
        """
        x = self.emb(x)
        x = x.sum(dim=1)
        x = self.proj(x)
        return x


def compute_topk_similar(
        word_emb: torch.Tensor, w2v_emb_weight: torch.Tensor, k
) -> list:
    """
    Helper function used in retrieve_similar_words and
    word_analogy, it allows you to retrieve the K indices with highest cosine similarity
    given a vector and an embedding weight.
    Helper function used in retrieve_similar_words and word_analogy, it allows you to
    retrieve the K indices with highest cosine similarity given a vector and an embedding weight.
    Concretely, this function computes the cosine similarity between the embedding of a single word
    and the embedding of all words and then returns the indices of the top K most similar results
    (excluding the word itself).
    :param word_emb:
    :param w2v_emb_weight:
    :param k:
    :return:
    """
    cosine = F.cosine_similarity(word_emb, w2v_emb_weight)
    _, topk_ix = torch.topk(cosine, k + 1)
    topk_ix = topk_ix[1:]
    # Convert to list
    return topk_ix.tolist()


@torch.no_grad()
def retrieve_similar_words(
        model: nn.Module,
        word: str,
        index_map: "dict[str, int]",
        index_to_word: "dict[int, str]",
        k: int = 5,
) -> "list[str]":
    """
    A function that, given a model and index map, takes a
    word and finds K most similar words.
    A function that, given a model and index map, takes a word and finds the K most similar words.
    Specifically, given your implementation of compute_topk_similar and a word2vec model,
    this function finds the K most similar words from your vocabulary. Make sure you set your model
    to evaluation mode and disable gradient calculation.
    :param model:
    :param word:
    :param index_map:
    :param index_to_word:
    :param k:
    :return:
    """
    ix = index_map[word]
    emb = model.emb(torch.tensor(ix))
    w2v_emb_weight = model.proj.weight
    topk_ix = compute_topk_similar(emb, w2v_emb_weight, k)
    topk_words = [index_to_word[ix] for ix in topk_ix]
    return topk_words


@torch.no_grad()
def word_analogy(
        model: nn.Module,
        word_a: str,
        word_b: str,
        word_c: str,
        index_map: "dict[str, int]",
        index_to_word: "dict[int, str]",
        k: int = 5,
) -> "list[str]":
    """
    A function that computes word analogies, e.g. man is to woman what ? is to girl.
    Using your compute_topk_similar function and a word2vec model, this function will
    compute the following analogy:
    "word_a" is to "word_b" what "?" is to "word_c"
    It can also be represented as as:
    word_a - word_b + word_c = ?
    Your function will find the K most similar words to "?" from your vocabulary. Make sure you set
    your model to evaluation mode and disable gradient computation.
    :param model:
    :param word_a:
    :param word_b:
    :param word_c:
    :param index_map:
    :param index_to_word:
    :param k:
    :return:
    """
    emb_a = model.emb(torch.tensor(index_map[word_a]))
    emb_b = model.emb(torch.tensor(index_map[word_b]))
    emb_c = model.emb(torch.tensor(index_map[word_c]))
    emb = emb_a - emb_b + emb_c
    w2v_emb_weight = model.proj.weight
    topk_ix = compute_topk_similar(emb, w2v_emb_weight, k)
    topk_words = [index_to_word[ix] for ix in topk_ix]
    return topk_words


# ######################## PART 2: YOUR WORK STARTS HERE ########################
# In this part of the assignment, you will be investigating techniques for measuring gender bias in
# word embeddings. Specifically, you will be investigating gender bias in GloVe embeddings. We
# will use the 300 dimensional glove.6B vectors. The embeddings can be downloaded from
# here (download the glove.6B.zip file and use the glove.6B.300d.txt file contained
# within).

def compute_gender_subspace(
        word_to_embedding: "dict[str, np.array]",
        gender_attribute_words: "list[tuple[str, str]]",
        n_components: int = 1,
) -> np.array:
    """
    To begin, you will implement compute_gender_subspace which estimates the "gender
    direction" in an embedding space. To estimate this gender direction, you will use the gender
    attribute words we have provided. You will make use of the sklearn implementation of PCA to
    estimate the gender direction.
    Concretely, your function will take a dictionary of word embeddings and a list of pairs of gender
    attribute words (e.g., man/woman) and estimate a gender subspace. The steps for estimating
    the gender subspace are:
    1. Convert each pair of gendered words (e.g., man/woman) to their embeddings.
    2. Normalize each pair of embeddings. That is, compute the mean embedding for each pair
    (e.g., Mean(Emb(man), Emb(woman))) and subtract the mean from each embedding
    in the pair (e.g., Emb(man) - Mean):
    >>> mean = (word_to_embedding["man"]+ word_to_embedding["woman"]) / 2
    >>> man_embedding = word_to_embedding["man"] - mean
    >>> woman_embedding = word_to_embedding["woman"] - mean
    3. Run PCA using the resulting list of normalized embeddings.
    :param word_to_embedding:
    :param gender_attribute_words:
    :param n_components:
    :return:
    :return:
    """
    # Step1
    pairs = []
    for word1, word2 in gender_attribute_words:
        emb1 = word_to_embedding[word1]
        emb2 = word_to_embedding[word2]
        pairs.append((emb1, emb2))
    # Step2
    normalized_pairs = []
    for emb1, emb2 in pairs:
        mean = (emb1 + emb2) / 2
        emb1 = emb1 - mean
        emb2 = emb2 - mean
        normalized_pairs.append(emb1)
        normalized_pairs.append(emb2)
    # Step3
    pca = PCA(n_components=n_components)
    pca.fit(normalized_pairs)
    return pca.components_


def project(a: np.array, b: np.array) -> "tuple[float, np.array]":
    """
    Now that you’ve estimated a “gender direction” in the GloVe embedding space, we will want to
    project embeddings onto this estimated direction. In this section, you will implement a function
    which computes a vector projection. The projection of a vector a onto a vector b is defined as:
    Your function will also need to return the scaling coefficient s:
    which we will use later in the assignment. We denote the dot product between vectors a and b
    by a·b.
    :param a:
    :param b:
    :return:
    """
    s = np.dot(a, b) / np.dot(b, b)
    projection = s * b
    return s, projection


def compute_profession_embeddings(
        word_to_embedding: "dict[str, np.array]", professions: "list[str]"
) -> "dict[str, np.array]":
    """
    Now that we’ve estimated a gender direction in the embedding space and implemented a
    function to project vectors onto this direction, we will investigate gender bias within
    representations for professions (e.g., nurse, mechanic, doctor). In this section, you will compute
    representations for professions from the BEC-Pro dataset (Bartl et al., 2020).
    Since professions sometimes contain multiple words (e.g., aerospace engineer), your function
    will need to average the GloVe embeddings for professions which consist of multiple words. You
    can split the professions into subwords using simple whitespace tokenization (e.g.,
    profession.split()).
    :param word_to_embedding:
    :param professions:
    :return:
    """
    profession_to_embedding = {}
    for profession in professions:
        words = profession.split()
        embedding = np.mean([word_to_embedding[word] for word in words], axis=0)
        profession_to_embedding[profession] = embedding
    return profession_to_embedding


def compute_extreme_words(
        words: "list[str]",
        word_to_embedding: "dict[str, np.array]",
        gender_subspace: np.array,
        k: int = 10,
        max_: bool = True,
) -> "list[str]":
    """
    You will now implement the function compute_extreme_words which, given a list of words,
    computes the K words with either the smallest or largest scalar coefficients onto a given gender
    direction. You will make use of the project function you previously implemented to get the
    scaling coefficients.
    Concretely, for this function you will need to compute the scalar coefficients for each word
    embedding (corresponding to a word in words) onto the estimated gender direction. Then, you
    will return the words with either the largest scalar coefficients or the smallest. You can compute
    the scalar coefficients using only the first direction in your estimated gender subspace (e.g., the
    first principle component).
    :param words:
    :param word_to_embedding:
    :param gender_subspace:
    :param k:
    :param max_:
    :return:
    """
    words_to_coeff = {}
    for word in words:
        emb = word_to_embedding[word]
        s, _ = project(emb, gender_subspace[0])
        words_to_coeff[word] = s
    sorted_words = sorted(words_to_coeff, key=words_to_coeff.get, reverse=max_)
    return sorted_words[:k]


# You will now implement the DirectBias metric from Bolukbasi et al., 2016 (Section 5.2):
# Concretely, given a set of words N which we expect to be gender neutral. We compute the total
# absolute cosine similarity (raised to the power of C) between words from N and the estimated
# gender direction g.

def cosine_similarity(a: np.array, b: np.array) -> float:
    """
    A helper function for compute_direct_bias which computes the cosine similarity between
    two vectors. Note: Do not assume that a or b is a unit vector.
    :param a:
    :param b:
    :return:
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_direct_bias(
        words: "list[str]",
        word_to_embedding: "dict[str, np.array]",
        gender_subspace: np.array,
        c: float = 0.25,
):
    """
    For a given set of words and an estimated gender subspace, computes the DirectBias metric.
    You will compute the DirectBias metric using only the first direction in your estimated gender
    subspace (e.g., the first principle component).
    :param words:
    :param word_to_embedding:
    :param gender_subspace:
    :param c:
    :return:
    """
    total = 0
    for word in words:
        for w in word.split():
            emb = word_to_embedding[w]
            total += np.abs(cosine_similarity(emb, gender_subspace[0])) ** c
    return total / len(words)


# You will now implement the Word Embedding Association Test (WEAT; Caliskan et al., 2017).
# WEAT is used to test for social bias within word representations.

def weat_association(
        w: str, A: "list[str]", B: "list[str]", word_to_embedding: "dict[str, np.array]"
) -> float:
    """
    Computes a word’s mean cosine similarity with the words from each attribute word set. Then,
    returns the difference between these two means.
    :param w:
    :param A:
    :param B:
    :param word_to_embedding:
    :return:
    """
    emb_w = word_to_embedding[w]
    mean_A = np.mean([cosine_similarity(emb_w, word_to_embedding[a]) for a in A])
    mean_B = np.mean([cosine_similarity(emb_w, word_to_embedding[b]) for b in B])
    return mean_A - mean_B


def weat_differential_association(
        X: "list[str]",
        Y: "list[str]",
        A: "list[str]",
        B: "list[str]",
        word_to_embedding: "dict[str, np.array]",
        weat_association_func: Callable,
) -> float:
    """
    This function computes the WEAT test statistic for given sets of target words (X, Y), attribute
    words (A, B), and embeddings. You should make use of your previous weat_association implementation.
    :param X:
    :param Y:
    :param A:
    :param B:
    :param word_to_embedding:
    :param weat_association_func:
    :return:
    """
    total = 0
    for x in X:
        total += np.mean([weat_association_func(x, A, B, word_to_embedding)])
    for y in Y:
        total -= np.mean([weat_association_func(y, A, B, word_to_embedding)])
    return total


def debias_word_embedding(
        word: str, word_to_embedding: "dict[str, np.array]", gender_subspace: np.array
) -> np.array:
    """
    Now that you have implemented two metrics for gender bias in word embeddings, you will now
    implement HardDebias, a technique for mitigating bias in word embeddings.
    To implement HardDebias, you will make use of your estimated gender direction from Part 2 of
    this assignment. You will be required to implement two functions for HardDebias:
    debias_word_embedding and hard_debias.
    Given a word and an estimated gender subspace, this function subtracts the embedding’s
    projection onto the estimated gender subspace from itself (making the embedding orthogonal to
    the estimated gender subspace).
    :param word:
    :param word_to_embedding:
    :param gender_subspace:
    :return:
    """
    emb = word_to_embedding[word]
    _, projection = project(emb, gender_subspace[0])
    return emb - projection


def hard_debias(
        word_to_embedding: "dict[str, np.array]",
        gender_attribute_words: "list[str]",
        n_components: int = 1,
) -> "dict[str, np.array]":
    """
    Given a dictionary of word embeddings, this function uses debias_word_embedding to
    debias all of the word embeddings.
    :param word_to_embedding:
    :param gender_attribute_words:
    :param n_components:
    :return:
    """
    subspace = compute_gender_subspace(word_to_embedding, gender_attribute_words, n_components)
    debiased_word_to_embedding = {}
    for word in word_to_embedding:
        debiased_word_to_embedding[word] = debias_word_embedding(word, word_to_embedding, subspace)
    return debiased_word_to_embedding


# First, using your implementation of compute_direct_bias, you will measure the gender bias
# in the profession embeddings before and after applying HardDebias. When computing
# DirectBias, you can use c = 0.25. Report these DirectBias values on Gradescope and
# comment on what the ideal score for DirectBias (one to two sentences).


if __name__ == "__main__":
    random.seed(2022)
    torch.manual_seed(2022)

    # # Parameters (you can change them)
    # sample_size = 2500  # Change this if you want to take a subset of data for testing
    # batch_size = 64
    # n_epochs = 2
    # num_words = 50000
    #
    # # Load the data
    # data_path = "../input/a1-data"  # Use this for kaggle
    # # data_path = "data"  # Use this if running locally
    #
    # # If you use GPUs, use the code below:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ###################### PART 1: TEST CODE ######################
    # print("=" * 80)
    # print("Running test code for part 1")
    # print("-" * 80)
    #
    # # Prefilled code showing you how to use the helper functions
    # train_raw, valid_raw = load_datasets(data_path)
    # if sample_size is not None:
    #     for key in ["premise", "hypothesis", "label"]:
    #         train_raw[key] = train_raw[key][:sample_size]
    #         valid_raw[key] = valid_raw[key][:sample_size]
    #
    # full_text = (
    #         train_raw["premise"]
    #         + train_raw["hypothesis"]
    #         + valid_raw["premise"]
    #         + valid_raw["hypothesis"]
    # )
    #
    # # Process into indices
    # tokens = tokenize_w2v(full_text)
    #
    # word_counts = build_word_counts(tokens)
    # word_to_index = build_index_map(word_counts, max_words=num_words)
    # index_to_word = {v: k for k, v in word_to_index.items()}
    #
    # text_indices = tokens_to_ix(tokens, word_to_index)
    #
    # # Train CBOW
    # sources_cb, targets_cb = cbow_preprocessing(text_indices, window_size=2)
    # loader_cb = DataLoader(
    #     Word2VecDataset(sources_cb, targets_cb),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=collate_cbow,
    # )
    #
    # model_cb = CBOW(num_words=len(word_to_index), embed_dim=200).to(device)
    # optimizer = torch.optim.Adam(model_cb.parameters())
    #
    # for epoch in range(n_epochs):
    #     loss = train_w2v(model_cb, optimizer, loader_cb, device=device).item()
    #     print(f"Loss at epoch #{epoch}: {loss:.4f}")
    #
    # # Training Skip-Gram
    #
    # # TODO: your work here
    # model_sg = SkipGram(num_words=len(word_to_index), embed_dim=200).to(device)
    #
    # # RETRIEVE SIMILAR WORDS
    # word = "man"
    #
    # similar_words_cb = retrieve_similar_words(
    #     model=model_cb,
    #     word=word,
    #     index_map=word_to_index,
    #     index_to_word=index_to_word,
    #     k=5,
    # )
    #
    # similar_words_sg = retrieve_similar_words(
    #     model=model_sg,
    #     word=word,
    #     index_map=word_to_index,
    #     index_to_word=index_to_word,
    #     k=5,
    # )
    #
    # print(f"(CBOW) Words similar to '{word}' are: {similar_words_cb}")
    # print(f"(Skip-gram) Words similar to '{word}' are: {similar_words_sg}")
    #
    # # COMPUTE WORDS ANALOGIES
    # a = "man"
    # b = "woman"
    # c = "girl"
    #
    # analogies_cb = word_analogy(
    #     model=model_cb,
    #     word_a=a,
    #     word_b=b,
    #     word_c=c,
    #     index_map=word_to_index,
    #     index_to_word=index_to_word,
    # )
    # analogies_sg = word_analogy(
    #     model=model_sg,
    #     word_a=a,
    #     word_b=b,
    #     word_c=c,
    #     index_map=word_to_index,
    #     index_to_word=index_to_word,
    # )
    #
    # print(f"CBOW's analogies for {a} - {b} + {c} are: {analogies_cb}")
    # print(f"Skip-gram's analogies for {a} - {b} + {c} are: {analogies_sg}")
    #
    # # ###################### PART 1: TEST CODE ######################
    #
    # # Prefilled code showing you how to use the helper functions
    # word_to_embedding = load_glove_embeddings("data/glove/glove.6B.300d.txt")
    #
    # professions = load_professions("data/professions.tsv")
    #
    # gender_attribute_words = load_gender_attribute_words(
    #     "data/gender_attribute_words.json"
    # )
    #
    # # === Section 2.1 ===
    # gender_subspace = compute_gender_subspace(word_to_embedding, gender_attribute_words)
    #
    # # === Section 2.2 ===
    # a = word_to_embedding
    # b =
    # scalar_projection, vector_projection = "your work here"
    #
    # # === Section 2.3 ===
    # profession_to_embedding = "your work here"
    #
    # # === Section 2.4 ===
    # positive_profession_words = "your work here"
    # negative_profession_words = "your work here"
    #
    # print(f"Max profession words: {positive_profession_words}")
    # print(f"Min profession words: {negative_profession_words}")
    #
    # # === Section 2.5 ===
    # direct_bias_professions = "your work here"
    #
    # # === Section 2.6 ===
    #
    # # Prepare attribute word sets for testing
    # A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
    # B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
    #
    # # Prepare target word sets for testing
    # X = ["doctor", "mechanic", "engineer"]
    # Y = ["nurse", "artist", "teacher"]
    #
    # word = "doctor"
    # weat_association = "your work here"
    # weat_differential_association = "your work here"
    #
    # # === Section 3.1 ===
    # debiased_word_to_embedding = "your work here"
    # debiased_profession_to_embedding = "your work here"
    #
    # # === Section 3.2 ===
    # direct_bias_professions_debiased = "your work here"
    #
    # print(f"DirectBias Professions (debiased): {direct_bias_professions_debiased:.2f}")
    #
    # X = [
    #     "math",
    #     "algebra",
    #     "geometry",
    #     "calculus",
    #     "equations",
    #     "computation",
    #     "numbers",
    #     "addition",
    # ]
    #
    # Y = [
    #     "poetry",
    #     "art",
    #     "dance",
    #     "literature",
    #     "novel",
    #     "symphony",
    #     "drama",
    #     "sculpture",
    # ]
    #
    # # Also run this test for debiased profession representations.
    # p_value = "your work here"
    #
    # print(f"p-value: {p_value:.2f}")
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Assuming helper functions are already defined and necessary data is loaded

    # Load GloVe embeddings and profession names
    # glove_path = "path/to/glove.6B.300d.txt"
    word_to_embedding = load_glove_embeddings("data/glove/glove.6B.300d.txt")
    professions_path = "data/professions.tsv"
    professions = load_professions(professions_path)
    gender_words_path = "data/gender_attribute_words.json"
    gender_attribute_words = load_gender_attribute_words(gender_words_path)

    # Step 1: Compute DirectBias before and after HardDebias
    gender_direction = compute_gender_subspace(word_to_embedding, gender_attribute_words)
    direct_bias_before = compute_direct_bias(professions, word_to_embedding, gender_direction, c=0.25)
    debiased_embeddings = hard_debias(word_to_embedding, gender_attribute_words)
    debiased_gender_direction = compute_gender_subspace(debiased_embeddings, gender_attribute_words)
    direct_bias_after = compute_direct_bias(professions, debiased_embeddings, debiased_gender_direction, c=0.25)

    # Step 2: WEAT Evaluation
    # Define target and attribute sets as per assignment requirements
    X = ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"]  # Example target words for Math
    Y = ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"]  # Example target words for Arts
    A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]  # Male attribute words
    B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]  # Female attribute words

    weat_score_before = weat_differential_association(X, Y, A, B, word_to_embedding, weat_association)
    p_value_before = p_value_permutation_test(X, Y, A, B, word_to_embedding)
    weat_score_after = weat_differential_association(X, Y, A, B, debiased_embeddings, weat_association)
    p_value_after = p_value_permutation_test(X, Y, A, B, debiased_embeddings)

    # Step 3: Gender Bias T-SNE Plot before debiasing
    profession_embeddings = [word_to_embedding[profession] for profession in professions if
                             profession in word_to_embedding]
    tsne = TSNE(n_components=2, random_state=42)
    # Filter professions to match embeddings list size
    filtered_professions = [profession for profession in professions if profession in word_to_embedding]

    # Ensure we're iterating over the filtered list
    profession_embeddings_2d = tsne.fit_transform(
        [word_to_embedding[profession] for profession in filtered_professions])

    plt.figure(figsize=(10, 10))
    for i, profession in enumerate(filtered_professions):
        plt.scatter(profession_embeddings_2d[i, 0], profession_embeddings_2d[i, 1])
        plt.annotate(profession, (profession_embeddings_2d[i, 0], profession_embeddings_2d[i, 1]), fontsize=9)
    plt.title('T-SNE visualization of profession embeddings before debiasing')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.show()

    # Print results
    print(f"DirectBias before debiasing: {direct_bias_before}")
    print(f"DirectBias after debiasing: {direct_bias_after}")
    print(f"WEAT score before debiasing: {weat_score_before}, p-value: {p_value_before}")
    print(f"WEAT score after debiasing: {weat_score_after}, p-value: {p_value_after}")


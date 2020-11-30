import operator
import os
import random

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

random.seed = 7


class Vocabulary:
    def __init__(self, unknown_token, padding_token, embedding_norm=0.25, embedding_dim=300):
        self.unknown_token = unknown_token
        self.padding_token = padding_token
        self.vocab = {}
        self.embeddings = None
        self.reverse_index = None
        self.vocab_file_name = "vocab.csv"
        self.embedding_file_name = "embeddings.csv"
        self.missing_words = None
        self.embedding_norm = embedding_norm
        self.embedding_dim = embedding_dim

    def add_word(self, text):
        if text not in self.vocab:
            self.vocab[text] = 1
        else:
            self.vocab[text] += 1

    def get_word(self, index):
        return list(self.embeddings.keys())[index]

    def get_word_index(self, word):
        if self.reverse_index is None:
            self._build_reverse_index()

        if word in self.reverse_index:
            return self.reverse_index[word]
        else:
            return self.reverse_index[self.unknown_token]

    def _build_reverse_index(self):
        self.reverse_index = {}
        for (index, word) in enumerate(self.embeddings.keys()):
            self.reverse_index[word] = index

    def get_embedding(self, word):
        if word in self.embeddings:
            return self.embeddings[word]
        else:
            return self.embeddings[self.unknown_token]

    def get_embedding_dim(self):
        return self.embedding_dim

    def load_word2vec_embeddings(self, file_name, embed_type):
        word2vec_embeddings = None
        if embed_type == "graph":
            word2vec_embeddings = Word2Vec.load(file_name)
        elif embed_type == "normal":
            word2vec_embeddings = KeyedVectors.load_word2vec_format(file_name, binary=False)
        else:
            word2vec_embeddings = KeyedVectors.load_word2vec_format(file_name, binary=True, unicode_errors='ignore')
        self.embeddings = {}
        self.missing_words = []
        self._assert_meta_token_embeddings()  # NOTE: always add meta tags in the beginning
        for key in self.vocab.keys():
            if key not in word2vec_embeddings:
                self.missing_words.append(key)
            else:
                self.embeddings[key] = word2vec_embeddings[key]

        # Update the embedding dimension if the loaded embeddings have a different dimension than the initialisation
        loaded_embedding_dimension = len(next(iter(self.embeddings.values())))
        if loaded_embedding_dimension != self.embedding_dim:
            print("Warning: Updating the embedding dimension from %d to %d" %
                  (self.embedding_dim, loaded_embedding_dimension))
            self.embedding_dim = loaded_embedding_dimension

    def fill_embeddings(self, fill_randomly=True):
        # If no embeddings have been loaded, fill everything
        if self.embeddings is None:
            self.embeddings = {}
            self.missing_words = []
            for key in self.vocab.keys():
                self.missing_words.append(key)

        self._assert_meta_token_embeddings()

        for missing_word in self.missing_words:
            if fill_randomly:
                self.embeddings[missing_word] = self._generate_random_embedding()
            else:
                self.embeddings[missing_word] = [0] * self.embedding_dim
        print("ALLA: ", self.embeddings)
        self.missing_words = None

    def remove_rare_words(self, min_frequency=1):
        removed_words = []
        for word, frequency in self.vocab.items():
            if frequency < min_frequency:
                removed_words.append(word)
        for word in removed_words:
            del self.vocab[word]
            del self.embeddings[word]

    def print_stats(self):
        sorted_vocab = sorted(self.vocab.items(), key=operator.itemgetter(1), reverse=True)
        print()
        print("Vocabulary stats")
        print("Total token count: %i" % sum(self.vocab.values()))
        print("Unique word count: %i" % len(sorted_vocab))
        print("Top 10 most common words: ", list(sorted_vocab)[0:10])
        print()

        if self.missing_words is not None:
            print("Found embeddings for %i words" % len(self.embeddings.items()))
            print("Generated random embeddings for %i missing words" % len(self.missing_words))
            missing_word_counts = [[word, self.vocab[word]] for word in self.missing_words]
            sorted_missing_words = sorted(missing_word_counts, key=operator.itemgetter(1), reverse=True)
            print("Top 10 most common missing words: ", list(sorted_missing_words)[0:10])
            print()

    def write(self, vocab_dir):
        os.makedirs(vocab_dir, exist_ok=True)
        embeddings_file = open(os.path.join(vocab_dir, self.embedding_file_name), "w")
        for word, embedding in self.embeddings.items():
            embeddings_file.write("%s %s\n" % (word, " ".join([str(x) for x in embedding])))
        embeddings_file.close()

        vocab_file = open(os.path.join(vocab_dir, self.vocab_file_name), "w")
        for word, count in self.vocab.items():
            vocab_file.write("%s %s\n" % (word, count))
        vocab_file.close()

    def load(self, vocab_dir):
        self.embeddings = {}
        self.vocab = {}

        embeddings_file = open(os.path.join(vocab_dir, self.embedding_file_name))
        for line in embeddings_file:
            split_line = line.split(" ")
            word = split_line[0]
            embedding = split_line[1:]
            self.embeddings[word] = embedding
        embeddings_file.close()

        vocab_file = open(os.path.join(vocab_dir, self.vocab_file_name))
        for line in vocab_file:
            word, count = line.split(" ", maxsplit=1)
            self.vocab[word] = count
        vocab_file.close()

    def _assert_meta_token_embeddings(self):
        if self.unknown_token not in self.embeddings:
            self.embeddings[self.unknown_token] = self._generate_random_embedding()

        if self.padding_token not in self.embeddings:
            self.embeddings[self.padding_token] = self._generate_zero_embedding()

    def _generate_random_embedding(self):
        return [random.uniform(-self.embedding_norm, self.embedding_norm) for x in range(self.embedding_dim)]

    def _generate_zero_embedding(self):
        return [0.0 for x in range(self.embedding_dim)]

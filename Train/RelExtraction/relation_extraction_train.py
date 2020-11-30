import logging
import re

from subprocess import run, PIPE
from tempfile import NamedTemporaryFile
from os.path import dirname
from os import chmod
from sklearn.metrics import confusion_matrix
import math
import os
import random
import Train.RelExtraction.training_curves

from os.path import join
from typing import List
from flair.data import TaggedCorpus, Sentence, Token
from flair.embeddings import TokenEmbeddings, WordEmbeddings, CharacterEmbeddings, RelativeOffsetEmbeddings, DocumentCNNEmbeddings, ConceptEmbeddings
from flair.models import TextClassifier
from flair.trainers import TextClassifierTrainer
from flair.file_utils import cached_path
from sklearn.metrics import classification_report





def train(data_dir: str, model_dir: str, dataset_format: str='macss_tdt', num_filters: int=150,
          word_embeddings: str='de-fasttext', offset_embedding_dim: int=100, learning_rate: float=.1,
          batch_size: int=32, max_epochs: int=1, dropout: float=.5, use_char_embeddings: bool=False,
          seed: int=0, dev_size: float=.1, test_size: float=.2, concept_embedding_dim: int=100):


    all_data = open('all_data.txt', encoding='utf8').read().split("\n")
    test_dev_percent = math.floor((len(all_data) * 25) / 100)
    k_folds = math.floor(len(all_data) / test_dev_percent)
    random.shuffle(all_data)
    config_name = '1_Some_Setting_Name'

    for i in range(k_folds):
        data_path = 'resources/' + config_name + '/' + str(i + 1)
        test_dev_set = all_data[(test_dev_percent * (i + 1)) - test_dev_percent:test_dev_percent * (i + 1)]
        train = all_data[0:(test_dev_percent * (i + 1)) - test_dev_percent] + all_data[
                                                                              test_dev_percent * (i + 1):len(all_data)]
        random.shuffle(test_dev_set)
        test_perc = math.floor((len(test_dev_set) * 60) / 100)
        test = test_dev_set[0:test_perc]
        dev = test_dev_set[test_perc:len(test_dev_set)]
        os.makedirs(data_path, exist_ok=True)
        train_txt = open(data_path + '/train.txt', 'w+')
        test_txt = open(data_path + '/test.txt', 'w+')
        dev_txt = open(data_path + '/dev.txt', 'w+')
        os.system('cp -r ./Data/vocabulary/ ' + data_path)
        train_txt.write('\n'.join(train))
        test_txt.write('\n'.join(test))
        dev_txt.write('\n'.join(dev))

        train_txt.close()
        test_txt.close()
        dev_txt.close()

        #print("Train Directory: ", data_dir, dev_size, seed, "\n")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S')

        if dataset_format not in ['macss_tdt']:
            raise ValueError(f"Dataset format '{dataset_format}' not supported.")

        corpus: TaggedCorpus = dataset_loader[dataset_format](data_path, 'train.txt', 'dev.txt', 'test.txt')
        label_dictionary = corpus.make_label_dictionary()  # rel-type

        # Comment out the embeddings that you don't need
        embedding_types: List[TokenEmbeddings] = [
            # mEx Fine-Tuned Word Embeddings
            #WordEmbeddings('../../Resources/mex-ft-wiki-de-finetuned-biomedical.gensim'),

            # Default German FastText Word Embeddings
            #WordEmbeddings('../../Resources/ft-wiki-de.gensim'),

            # Relative Offset Embeddings
            RelativeOffsetEmbeddings('offset_e1', max_len=200, embedding_dim=offset_embedding_dim),
            RelativeOffsetEmbeddings('offset_e2', max_len=200, embedding_dim=offset_embedding_dim),

            # Concept Embeddings
            ConceptEmbeddings('concept_1', max_len=200, embedding_dim=concept_embedding_dim),
            ConceptEmbeddings('concept_2', max_len=200, embedding_dim=concept_embedding_dim),
        ]

        if use_char_embeddings:
            embedding_types += CharacterEmbeddings()

        document_embeddings: DocumentCNNEmbeddings = DocumentCNNEmbeddings(embedding_types,
                                                                           num_filters=num_filters,
                                                                           dropout=dropout)

        classifier: TextClassifier = TextClassifier(document_embeddings=document_embeddings,
                                                    label_dictionary=label_dictionary,
                                                    multi_label=False)

        trainer: TextClassifierTrainer = TextClassifierTrainer(classifier, corpus, label_dictionary)

        trainer.train(data_path,
                      learning_rate=learning_rate,
                      mini_batch_size=batch_size,
                      max_epochs=3,
                      use_tensorboard=False,
                      embeddings_in_memory=False)

        #plotter = training_curves.Plotter()
        #plotter.plot_training_curves(data_path + '/loss.tsv', ["LOSS", "F1", "ACC"])
    #os.system('mv runs/ resources/' + config_name + '/tf_runs')



def load_corpus(path_to_data, train_file='train.txt', dev_file='dev.txt', test_file='test.txt'):
    print("VOC>", join(path_to_data, 'vocabulary/embeddings.csv'))
    idx2item = load_idx2item(join(path_to_data, 'vocabulary/embeddings.csv'))

    sentences_train: List[Sentence] = load_sentences(join(path_to_data, train_file),
                                                           idx2item=idx2item, is_test=False)

    sentences_dev: List[Sentence] = load_sentences(join(path_to_data, dev_file),
                                                         idx2item=idx2item, is_test=False)

    sentences_test: List[Sentence] = load_sentences(join(path_to_data, test_file),
                                                          idx2item=idx2item, is_test=False)

    return TaggedCorpus(sentences_train, sentences_dev, sentences_test)


def load_idx2item(path_to_file):
    idx2item = {}
    with open(path_to_file) as f:
        for idx, line in enumerate(f.readlines()):
            item, _ = line.split(' ', 1)
            if item in idx2item:
                raise ValueError("Item '{}' at line {} appears" +
                    "multiple times in embeddings.".format(item, idx))
            idx2item[idx] = item
    return idx2item


def load_sentences(path_to_file, idx2item, is_test=True, attach_id=False):
    def add_offset_to_sentence(sentence, offsets, tag):
        for token, offset in zip(sentence.tokens, offsets):
            token.add_tag(tag, offset)
            #print(token, token.tags)
    # A code
    def add_concept_to_sentence(sentence, offsets, tag):
        for token, offset in zip(sentence.tokens, offsets):
            token.add_tag(tag, offset)

    def add_fix_concept_to_sentence(sentence, offsets, tag):
        for token, offset in zip(sentence.tokens, offsets):
            token.add_tag(tag, offset)

    def int_list_from_string(s):
        return list(map(int, s.split(' ')))

    sentences: List[Sentence] = []
    with open(path_to_file) as f:
        for line in f.readlines():
            if line != '\n':
                x_line = line.rstrip().split(':')
                id_ = x_line[0]
                label = x_line[1]
                token_indices = x_line[2]
                offsets_e1 = x_line[3]
                offsets_e2 = x_line[4]
                concept_1 = x_line[-2]
                concept_2 = x_line[-1]

                token_indices, offsets_e1, offsets_e2, concept_1, concept_2= [
                    int_list_from_string(s) for s in [token_indices, offsets_e1, offsets_e2, concept_1, concept_2]]

                sentence: Sentence = Sentence()
                if not is_test:
                    sentence.add_label(label)


                for token_idx in map(int, token_indices):
                    token = idx2item[token_idx]
                    sentence.add_token(Token(token))

                add_offset_to_sentence(sentence, map(int, offsets_e1), tag='offset_e1')
                add_offset_to_sentence(sentence, map(int, offsets_e2), tag='offset_e2')

                add_concept_to_sentence(sentence, map(int, concept_1), tag='concept_1')
                add_concept_to_sentence(sentence, map(int, concept_2), tag='concept_2')

                add_fix_concept_to_sentence(sentence, map(int, concept_1), tag='fix_concept_1')
                add_fix_concept_to_sentence(sentence, map(int, concept_2), tag='fix_concept_2')

                if attach_id:
                    setattr(sentence, 'id_', id_)

                # print ("sentence-len:", len(sentence))
                if len(sentence) > 0:
                    sentence._infer_space_after()
                    sentences.append(sentence)

    return sentences


dataset_loader = {
    'macss_tdt': load_corpus,
}


def evaluate(test_file, model_file, dataset_format='macss_test', semeval_scoring=False):
    if semeval_scoring:
        eval_script = cached_path(
            'https://raw.githubusercontent.com/vzhong/semeval/master/dataset/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl',
            cache_dir='scripts')
        chmod(eval_script, 0o777)

    classifier: TextClassifier = TextClassifier.load_from_file(model_file)

    load_dataset = dataset_loader[dataset_format]
    sentences_test: List[Sentence] = load_dataset(dirname(test_file), test_file, is_test=False)
    sentences_pred: List[Sentence] = load_dataset(dirname(test_file), test_file, is_test=True)


    sentences_pred = classifier.predict(sentences_pred)

    if semeval_scoring:
        id_labels_true = [(sentence.id_, sentence.labels[0]) for sentence in sentences_test]
        id_labels_pred = [(sentence.id_, sentence.labels[0]) for sentence in sentences_pred]

        input_files = []
        for id_labels in [id_labels_true, id_labels_pred]:
            tmp_file = NamedTemporaryFile(delete=True)
            input_files.append(tmp_file)
            with open(tmp_file.name, 'w') as f:
                for id_, label in id_labels:
                    f.write('{}\t{}\n'.format(id_, label.name))
            tmp_file.file.close()

        p = run([eval_script, input_files[0].name, input_files[1].name], stdout=PIPE, encoding='utf-8')
        main_result = p.stdout
        print(main_result)

    else:
        y_true = [sentence.labels[0].name for sentence in sentences_test]
        y_pred = [sentence.labels[0].name for sentence in sentences_pred]
        print(classification_report(y_true, y_pred))
        print ("##################")

        y_true = [re.sub("^not_.*$", "no_relation", re.sub("\(.*\)", "", sentence.labels[0].name)) for sentence in sentences_test]
        y_pred = [re.sub("^not_.*$", "no_relation", re.sub("\(.*\)", "", sentence.labels[0].name)) for sentence in sentences_pred]

        print(classification_report(y_true, y_pred))

        print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    data_dir = ""
    model_dir = ""

    train(data_dir, model_dir, 'macss_tdt', 150, '', 50, 0.05, 64, 10, 0.2, False, 2, 0.1, 0.2)




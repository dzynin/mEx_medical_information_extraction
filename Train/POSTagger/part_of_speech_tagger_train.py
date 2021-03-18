from flair.data import Corpus
from flair.datasets import UniversalDependenciesCorpus

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from typing import List
from pathlib import Path
import torch
import math
import random
import os
from shutil import copyfile, rmtree

from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split

enable_hdt = False
kf = KFold(n_splits=5, shuffle=True)
data_path = "Data"
fcg_data = np.array(open('Data/FCG.conll', encoding='utf8').read().split('\n\n'))
if enable_hdt:
    hdt_train_data = np.array(open('Data/HDT_train.conll', encoding='utf8').read().split('\n\n'))
    hdt_dev_data = np.array(open('Data/HDT_dev.conll', encoding='utf8').read().split('\n\n'))
# -------------------------------------------------------
gs_data = np.array(open('Data/GS_data.conll', encoding='utf8').read().split('\n\n'))
config_name = '1_def_flair'

curr_kf = 0
for train_index, test_index in kf.split(gs_data):
    if enable_hdt:
        data_path = 'resources_with_hdt/' + config_name + '/' + str(curr_kf + 1) + '/penn'
    else:
        data_path = 'resources_no_hdt/' + config_name + '/' + str(curr_kf + 1) + '/penn'

    dev_idx, train_idx = train_test_split(train_index, train_size=0.085, shuffle=True)
    train, dev, test = gs_data[train_idx], gs_data[dev_idx], gs_data[test_index]
    test = np.concatenate((test, fcg_data))

    if enable_hdt:
        train = np.concatenate((train, hdt_train_data))
        dev = np.concatenate((dev, hdt_dev_data))

    os.makedirs(data_path, exist_ok=True)
    train_txt = open(data_path + '/train.txt', 'w+')
    dev_txt = open(data_path + '/dev.txt', 'w+')
    test_txt = open(data_path + '/test.txt', 'w+')

    train_txt.write('\n\n'.join(train))
    dev_txt.write('\n\n'.join(dev))
    test_txt.write('\n\n'.join(test))

    train_txt.close()
    dev_txt.close()
    test_txt.close()
    curr_kf += 1

    corpus: Corpus = UniversalDependenciesCorpus(data_path)

    # 2. what tag do we want to predict?
    tag_type = 'pos'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    print(corpus)

    # initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        # WordEmbeddings('de')
        # ,
        # WordEmbeddings('/home/aayach/Work/convertEmb/model_utf8/output')
        # ,
        FlairEmbeddings('german-forward')
        ,
        FlairEmbeddings('german-backward')
        ,
        # FINE TUNED CHAR'S
        # FlairEmbeddings(
        #    '/mnt/raid0/experiments/aayach/CharEmbed/enhance_flair_char_embed/res_for/taggers/language_model/best-lm.pt')
        # ,
        # FlairEmbeddings(
        #    '/mnt/raid0/experiments/aayach/CharEmbed/enhance_flair_char_embed/res_back/taggers/language_model/best-lm.pt')
        # ,
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # initialize sequence tagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # initialize trainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(data_path, learning_rate=0.1, mini_batch_size=40, max_epochs=10, save_final_model=True)



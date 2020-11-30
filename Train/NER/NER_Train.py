from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.visual.training_curves import Plotter
from typing import List
import math
import os
import random

data_path = './all_data.txt'
k_folds = -1
test_dev_percent = 25
all_data = open(data_path, encoding='utf8').read().split("\n\n")
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

    train_txt.write('\n\n'.join(train))
    test_txt.write('\n\n'.join(test))
    dev_txt.write('\n\n'.join(dev))

    train_txt.close()
    test_txt.close()
    dev_txt.close()

    # define columns
    columns = {0: 'text', 1: 'ner'}

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_path, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt',
                                  column_delimiter='\t',
                                  document_separator_token='\n')

    for sent in corpus.train:
        for tok in sent:
            labs = tok.labels
            tok.remove_labels('ner')
            for lab in labs:
                tok.add_label('ner', lab.value.replace('\n', ''))

    for sent in corpus.test:
        for tok in sent:
            labs = tok.labels
            tok.remove_labels('ner')
            for lab in labs:
                tok.add_label('ner', lab.value.replace('\n', ''))

    for sent in corpus.dev:
        for tok in sent:
            labs = tok.labels
            tok.remove_labels('ner')
            for lab in labs:
                tok.add_label('ner', lab.value.replace('\n', ''))


    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    # initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        #WordEmbeddings('de'),

        #WordEmbeddings('../../Resources/mex-ft-wiki-de-finetuned-biomedical.gensim'),

        #FlairEmbeddings("de-forward"),
        #FlairEmbeddings("de-backward"),

        #PooledFlairEmbeddings('german-forward'),
        #PooledFlairEmbeddings('german-backward'),

        #PooledFlairEmbeddings(
        #    '../../Resources/mEx_Finetuned_Flair_Context_Embeddings_forwards.pt'),
        #PooledFlairEmbeddings(
        #    '../../Resources/mEx_Finetuned_Flair_Context_Embeddings_backwards.pt'),

        FlairEmbeddings(
            '../../Resources/mEx_Finetuned_Flair_Context_Embeddings_forwards.pt')
        ,
        FlairEmbeddings(
            '../../Resources/mEx_Finetuned_Flair_Context_Embeddings_backwards.pt')
        ,

        # TransformerWordEmbeddings('bert-base-german-cased', allow_long_sentences=True),
    ]


    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    #embeddings = TransformerWordEmbeddings('bert-base-german-cased', allow_long_sentences=True)

    # initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True,
                                            locked_dropout=0.3)

    # initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus, use_tensorboard=False)

    trainer.train(data_path,
                  train_with_dev=False,
                  max_epochs=1,
                  mini_batch_size=65)

    #plotter = Plotter()
    #plotter.plot_training_curves(data_path + '/loss.tsv')
    #plotter.plot_weights(data_path + '/weights.txt')

#os.system('mv runs/ resources/' + config_name + '/tf_runs')


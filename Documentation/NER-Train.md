# NER Train
## Requirements:
* The Training Data (CoNLL) data must be in the same train script directory.
* The data file must be **all_data.txt** named.

## Usage:
```shell
conda activate NER
python NER_Train.py
```

## Embeddings Types:
At this part of code you could select the types of embeddings you want to use in your model.
```python
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
```
# RelEx Train
## Requirements:
* The Training Data (RelVec) data must be in the same train script directory.
* The data file must be **all_data.txt** named
* The Data directory that contains (**embeddings.csv**, **vocab.csv**) files generated from the *Brat2Vec* script

## Usage:
```
conda activate RelEx
python relation_extraction_train.py
```

## Embeddings Types:
At this part of code you could select the types of embeddings you want to use in your model. <br>
currently you'll need the Default or Fine-tuned Word embeddings in the Resources folder, because flair changed the server location.
```
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
```
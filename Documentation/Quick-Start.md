# Quick Start:
## Requirements and Installation:
The project is based on [Flair 0.8](https://github.com/flairNLP/flair/releases/tag/v0.8) and 
[Flair 0.2.0](https://github.com/amrayach/flair) which have been tweaked to enable Relation Extraction, for that reason we need
two separate Virtual Environments to run the examples.
* Python 3.6
* mEx Models, optional:Default/Fine-tuned Word Embeddings
* mEx Models must be available under **./mEx_nlp/Resources/**
* NER, RelEx Virtual Envs
### Virtual Environments:
#### Download & Install Miniconda (or other environment manager):
Get the miniconda installer for your os under [conda](https://docs.conda.io/en/latest/miniconda.html)
#### Create NER VirtualEnv:
```
conda create -n NER python=3.6
pip install requierments_ner.txt

#  Activate Env
conda activate NER
```
#### Create RelEx VirtualEnv:
 ```
conda create -n RelEx python=3.6
pip install requierments_relex.txt

# Activate Env
conda activate RelEx
```
## NER Prediction Server:
For all the following examples, the NER Prediction Server must be on.
make sure the port is available and same in **app.py** and **named_entity_recognition.py**
 ```
conda activate NER
cd mEx_nlp/NamedEntityRecognition/
python app.py
 ```

---

## Named-Entity-Recognition Jupyter Notebook Manual:
This is a quick tutorial on how to use the NER model.
 ```
conda activate NER
jupyter notebook NER_Manual.ipynb
 ```
---
## Relation-Extraction Jupyter Notebook Manual:
This is a quick tutorial on how to use the RelEx model.
 ```
conda activate RelEx
jupyter notebook REL_Manual.ipynb
 ```
---
## Model Results Virtualization:
This part was built using [Streamlit](https://www.streamlit.io/), [Spacy](https://spacy.io/) and [spacy-streamlit](https://github.com/explosion/spacy-streamlit).
 ```
conda activate RelEx
streamlit run visualize_model_predictions.py
 ```
---
![](NER_viz.png)
---
![](REL_viz.png)
---
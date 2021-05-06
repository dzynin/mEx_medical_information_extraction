[![Logo](Documentation/DFKI_Logo.jpg)](https://www.dfki.de/web/)

# mEx - Medical Information Extraction:

---
## Introduction:
**I**n the information extraction and natural language processing domain,
accessible datasets are crucial to reproduce and compare results.
Publicly available implementations and tools can serve as benchmark and
facilitate the development of more complex applications. However, in the
context of clinical text processing the number of accessible datasets is
scarce - and so is the number of existing tools. One of the main reasons
is the confidentiality of the data. This problem is even more evident
for non-English languages.

**I**n order to address this situation, we introduce a workbench: a
collection of German clinical text processing models. The models are
trained on a de-identified corpus of German nephrology reports and
provide promising results on in-domain data. Moreover, our models can be
also successfully applied to other biomedical text in German. Our
workbench is made publicly available, so it can be used out of the box,
as a benchmark or transferred to related problems.

---

## Tools:

Our workbench includes a range of different tools to process German
clinical text:

* [Quick Start](Documentation/Quick-Start.md)

### Tutorials:
* [Tutorial 1: Data](Documentation/Data.md)
* [Tutorial 2: Data-Preparation](Documentation/Data-Preparation.md)
* [Tutorial 3: Pipeline](Documentation/Pipeline.md)
* [Tutorial 4: NER-Train](Documentation/NER-Train.md)
* [Tutorial 5: RelEx-Train](Documentation/RelEx-Train.md)


### Models:
* [Concept Detection]() !! Link is still missing !!
* [Relation Detection]() !! Link is still missing !!
* [Part of Speech Tagger]() !! Link is still missing !!
* [Negation Detection](http://macss.dfki.de/german_trigger_set.html)
* [Dependency Tree Parser](http://macss.dfki.de/dependency_parser.html)


## mEx Models Overview:

| Task | Language | Dataset | Score | Download Model|
| -------------------------------  | ---  | ----------- | ---------------- | ------------- |
| Named Entity Recognition |German | German Nephrology Corpus (Charite)   |  **83.25** (F1)  | [*named_entity_recognition_mex_model(custom_flair_embeddings).pt*](https://cloud.dfki.de/owncloud/index.php/s/WWbnqJ6N8gQQWMD)|
| Relation Extraction |German | German Nephrology Corpus (Charite)   |  **84.0** (F1)  | [*relation_extraction_mex_model(Custom_Word_Concept_Relative_Embeddings).pt*](https://cloud.dfki.de/owncloud/index.php/s/zDH7FHNbXQXkcLx)|
| Part-of-Speech Tagging |German| German Nephrology Corpus (Charite)  | **98.57** (Acc.) | [*part_of_speech_tagger_mex_model(default_word_flair_embeddings).pt*](https://cloud.dfki.de/owncloud/index.php/s/e7G9deea7eRksCY)|

---

### Extra Resources:

* Default FastText Embeddings (Gensim Format): [ft-wiki-de.gensim](https://cloud.dfki.de/owncloud/index.php/s/FwyZY3GcXzeCJiy) & [ft-wiki-de.gensim.vectors.npy](https://cloud.dfki.de/owncloud/index.php/s/sXRQQMa885mf2Wa)
* Fine-Tuned FastText Embeddings (Gensim Format): [mex-ft-wiki-de-finetuned-biomedical.gensim](https://cloud.dfki.de/owncloud/index.php/s/y8gn55TWpDZFdq8) & [mex-ft-wiki-de-finetuned-biomedical.gensim.npy](https://cloud.dfki.de/owncloud/index.php/s/rfGoDsCoySLWs5f)
* mEx Fine-Tuned Flair Context Embeddings (Forward & Backwards): [Backwards](https://cloud.dfki.de/owncloud/index.php/s/Rx5qcrKKpx79cm9) & [Forwards](https://cloud.dfki.de/owncloud/index.php/s/D3G8rPBp9ZXYb5T)

---

---

## Related-Resources:

---
For more information we refer to our short publication (_a more detailed
version is currently under review_):

Roland Roller, Laura Seiffe, Ammer Ayach, Sebastian Möller, Oliver
Marten, Michael Mikhailov, Christoph Alt, Danilo Schmidt, Fabian
Halleck, Marcel Naik, Wiebke Duettmann and Klemens Budde.  [**Information
Extraction Models for German Clinical Text**](https://ieeexplore.ieee.org/document/9374385). In 2020 IEEE International
Conference on Healthcare Informatics (ICHI). Oldenburg, 2020.

### Citing:

```
@INPROCEEDINGS{9374385,  
    author={Roller, Roland and Seiffe, Laura and Ayach, Ammer and Möller, Sebastian and Marten, Oliver and Mikhailov, Michael and Alt, Christoph and Schmidt, Danilo and Halleck, Fabian and Naik, Marcel and Duettmann, Wiebke and Budde, Klemens},  
    booktitle={2020 IEEE International Conference on Healthcare Informatics (ICHI)},   
    title={Information Extraction Models for German Clinical Text},   
    year={2020},  
    pages={1-2},  
    doi={10.1109/ICHI48887.2020.9374385}
    }
```

---

**License: CC BY-NC 4.0** 

[![License: CC BY-NC 4.0](https://i.creativecommons.org/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

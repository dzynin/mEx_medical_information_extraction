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

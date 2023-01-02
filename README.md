# Efficient Task-Oriented Dialogue Systems with Response Selection as an Auxiliary Task

**Abstract:** The adoption of pre-trained language models in task-oriented dialogue systems has resulted in significant enhancements of their text generation abilities. However, these architectures are slow to use because of the large number of trainable parameters and can sometimes fail to generate diverse responses. To address these limitations, we propose two models with auxiliary tasks for response selection - (1) distinguishing distractors from ground truth responses and (2) distinguishing synthetic responses from ground truth labels. They achieve state-of-the-art results on the MultiWOZ 2.1 dataset with combined scores of 107.5 and 108.3 and outperform a baseline with three times more parameters. We publish reproducible code and checkpoints and discuss the effects of applying auxiliary tasks to T5-based architectures.

**Paper:** https://arxiv.org/abs/2208.07097

**Implementation:**
- (1) https://github.com/radi-cho/RSTOD/tree/encoder
- (2) https://github.com/radi-cho/RSTOD/tree/main

**Checkpoints:** https://drive.google.com/drive/folders/1W7MoU2LfOeaVND0BCGtWVJ-lJqqK0CDj

# Cite

```bibtex
@misc{https://doi.org/10.48550/arxiv.2208.07097,
  doi = {10.48550/ARXIV.2208.07097},
  url = {https://arxiv.org/abs/2208.07097},
  author = {Cholakov, Radostin and Kolev, Todor},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Efficient Task-Oriented Dialogue Systems with Response Selection as an Auxiliary Task},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

# Credits
- Our work was presented at [ICNLSP](https://www.icnlsp.org/) 2022.
- This research initially started during [SRS](http://www.math.bas.bg/srs/newsite/srs/)'22.
- The [MTTOD](https://aclanthology.org/2021.findings-emnlp.112/) architecture was used as a baseline. We also used some of the training and evaluation scripts from https://github.com/bepoetree/MTTOD. Make sure to also cite:

```bibtex
@inproceedings{lee-2021-improving-end,
    title = "Improving End-to-End Task-Oriented Dialog System with A Simple Auxiliary Task",
    author = "Lee, Yohan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.112",
    doi = "10.18653/v1/2021.findings-emnlp.112",
    pages = "1296--1303",
    abstract = "The paradigm of leveraging large pre-trained language models has made significant progress on benchmarks on task-oriented dialogue (TOD) systems. In this paper, we combine this paradigm with multi-task learning framework for end-to-end TOD modeling by adopting span prediction as an auxiliary task. In end-to-end setting, our model achieves new state-of-the-art results with combined scores of 108.3 and 107.5 on MultiWOZ 2.0 and MultiWOZ 2.1, respectively. Furthermore, we demonstrate that multi-task learning improves not only the performance of model but its generalization capability through domain adaptation experiments in the few-shot setting. The code is available at github.com/bepoetree/MTTOD.",
}
```

- Refer to https://github.com/budzianowski/multiwoz for dataset details.

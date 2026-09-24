---
title: README
emoji: 🐋
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# Illinois Social Attitudes Aggregate Corpus (ISAAC)

ISAAC is a corpus of 527,060,919 English-language Reddit posts from 2007 to 2023 about
six social group distinctions: ability, age, race, sexuality, skin tone and body weight.
Every post carries 59 fields, including labels for moralization, sentiment, emotion,
linguistic generalization and estimated user location.

## Where everything lives

| | |
| --- | --- |
| **Corpus** (341 GB, one configuration per social group) | [BabakScrapes/isaac-reddit](https://huggingface.co/datasets/BabakScrapes/isaac-reddit) |
| **Classifiers** (9 models) | the repositories on this page |
| **Try the classifiers without code** | [BabakScrapes/isaac-classifiers](https://huggingface.co/spaces/BabakScrapes/isaac-classifiers) |
| **Project website**, browsing and direct download | [isaac.psychology.illinois.edu](https://isaac.psychology.illinois.edu/) |
| **Pipeline source**, keyword lists, pattern sets | [GitHub](https://github.com/BabakHemmatian/Illinois_Social_Attitudes) |
| **Python loader** | [`isaac-data`](https://pypi.org/project/isaac-data/) |
| **Data Use Agreement** | [Data_Use_Agreement.md](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md) |

The corpus is hosted on a personal account rather than this organization for storage
reasons only. It is the same project and the same terms.

## Terms

The nine classifiers on this page are released under a Creative Commons Attribution 4.0
International License, without access restrictions. The corpus is governed by the ISAAC
Data Use Agreement, which prohibits re-identification and redistribution. The location
model is not published here: access can be requested by email under the ISAAC Model Use
Agreement.

## Citation

One citation covers the whole project, including the corpus, the pipeline, the models and
every access route. Please do not cite the repositories separately.

```bibtex
@article{hemmatian2026isaac,
  author  = {Hemmatian, Babak and Hadjarab, Sarah and Chen, Jessica and Kurdi, Benedek},
  title   = {The {Illinois} Social Attitudes Aggregate Corpus ({ISAAC}): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale},
  year    = {2026},
  journal = {arXiv},
  eprint  = {2609.27059},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi     = {10.48550/arXiv.2609.27059},
  url     = {https://arxiv.org/abs/2609.27059}
}
```

## Questions

[isaac.corpus.support@gmail.com](mailto:isaac.corpus.support@gmail.com)

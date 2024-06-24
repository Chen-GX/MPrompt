<p align="center">
   <img src="logo/Logo.png" width="60%" align='center' />
</p>

# MPrompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension (EMNLP 2023 Findings)

**This repository contains the code for our research paper titled "[MPrompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension](https://aclanthology.org/2023.findings-emnlp.343/)", which has been accepted for the Findings of EMNLP 2023.**

> Guoxin Chen, Yiming Qian, Bowen Wang, and Liangzhi Li. 2023. MPrompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5163–5175, Singapore. Association for Computational Linguistics.

##  Overall Framework

<p align="center">
   <img src="figure/Figure 1_The overall framework of MPrompt.png" width="95%" align='center' />
<p align="center">
  <small> Figure 1: The overall framework of MPrompt</small>

# Requirements
* Python 3.8
* Ubuntu 22.04
* Python Packages

```
conda create -n MPrompt python=3.9
conda activate MPrompt
pip install -r requirements.txt
```

# Data
The folder `./qa_datasets` contains the example data. 
```
cd ./qa_datasets
unzip *.zip
```

# Training
For a single training, all scripts are located in the ./tdk_scripts folder:

```
cd ./tdk_scripts

bash boolq_tdk.sh
```


For parameter grid search, all scripts are located in the ./search_param_scripts folder:

```
cd ./search_param_scripts

bash boolq_prompt_len.sh
```

# Citation
**If our work contributes to your research, please acknowledge it by citing our paper. We greatly appreciate your support.**
```
@inproceedings{chen-etal-2023-mprompt,
    title = "{MP}rompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension",
    author = "Chen, Guoxin  and
      Qian, Yiming  and
      Wang, Bowen  and
      Li, Liangzhi",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.343",
    doi = "10.18653/v1/2023.findings-emnlp.343",
    pages = "5163--5175",
}
```
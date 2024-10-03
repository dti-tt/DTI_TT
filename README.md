# DTI-TT: Drug–Target Interaction Prediction Based on Two Tower Recommendation System

## Overview

This repository contains the code for DTI-TT, a tool designed to predict drug-target interactions (DTIs) using a Two Tower recommendation system.

![TT_bg](https://github.com/user-attachments/assets/9e30aeac-8326-44d0-8e79-27dc38c3868e)

---

## Project Structure

```bash
DTI_TT/
│
├── data/                                 # Dataset files
│   ├── coldstart_test.csv                # Cold-start test dataset
│   ├── coldstart_train.csv               # Cold-start training dataset
│   ├── warmstart_test.csv                # Warm-start test dataset
│   └── warmstart_train.csv               # Warm-start training dataset
│
├── DeepPurpose/                          # Deep learning library for DTI
│   ├── DTI_Baseline.py                   # Modified DTI.py for MLP-based Fusion
│   ├── DTI_TwoTower.py                   # Modified DTI.py for Two-Tower model
│   └── utils.py                          # Modified utils.py
│
├── embeddings/                           # Pretrained embeddings directory
│   ├── KPGT/                             # Pretrained KPGT directory
│   └── proteinBERT.pkl                   # Pretrained Protein BERT model
│
├── README.md                             # Project documentation
├── main.py                               # Main script to run experiments
└── requirements.txt                      # Python dependencies
```

---

## Installation

To get started with DTI-TT, follow these steps to set up your environment.

### Prerequisites

Ensure you have Python 3.10.

### Setup

1. Clone this repository:

```bash
git clone https://github.com/dti-tt/DTI_TT.git
cd DTI_TT
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Here is a list of key dependencies included in `requirements.txt`
- TensorFlow
- PyTorch
- DGL (Deep Graph Library)
- DeepPurpose
- RDKit
- scikit-learn
- pandas

3. Download the dataset and pretrained embeddings and models

#### Dataset, pretrained ProteinBERT embeddings:
  You can download the dataset and pretrained ProteinBERT embeddings in [here](https://drive.google.com/drive/folders/1TcfRF-1dblXWCbGo_VYw15wLAnbxLWQK?usp=sharing).

#### Pretrained KPGT models:
  - Download the pretrained KPGT models at [lihan97/KPGT](https://github.com/lihan97/KPGT) or [here](https://drive.google.com/drive/folders/1TcfRF-1dblXWCbGo_VYw15wLAnbxLWQK?usp=sharing) and save it to: `DTI_TT/src/embedddings/`
  - Download the KPGT embeddings `updated_smiles_dict.pkl` in [here](https://drive.google.com/drive/folders/1TcfRF-1dblXWCbGo_VYw15wLAnbxLWQK?usp=sharing) and save it to: `DTI_TT/src/KPGT/usage/`
  
---

## Usage

The `main.py` script allows you to run the model using different scenarios (cold-start or warm-start) and model architectures (Neural Network or Two Tower).

### Running the Model

Run the script with the following commands, specifying the desired scenario and model type:

1. Warm-Start with Neural Network:
```bash
python main.py bs warm
```
2. Warm-Start with Two Tower Model:
```bash
python main.py tt warm
```
3. Cold-Start with Neural Network:
```bash
python main.py bs cold
```
4. Cold-Start with Two Tower Model:
```bash
python main.py tt cold
```

### Dataset Information

The dataset includes both cold-start and warm-start splits to evaluate the model's performance in different scenarios:
- Cold-Start: The model encounters previously unseen drugs.
- Warm-Start: The model works with known drugs and targets.

Download the dataset, create a `DTI_TT/data` folder, and save the dataset in this folder. 

---

## Acknowledgments

- DeepPurpose [(kexinhuang12345/DeepPurpose)](https://github.com/kexinhuang12345/DeepPurpose): For drug-target interaction features. 
- DGL [(dmlc/dgl)](https://github.com/dmlc/dgl): For graph-based computations.
- RDKit [(RDKit)](https://www.rdkit.org/): For chemical informatics.

## Credits

This project incorporates code from [lihan97/KPGT](https://github.com/lihan97/KPGT), originally developed by lihan97.

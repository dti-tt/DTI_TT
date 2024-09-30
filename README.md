# DTI-TT: Drug–Target Interaction Prediction Based on Two Tower Recommendation System

## Overview

This repository contains the code for DTI-TT, a tool designed to predict drug-target interactions (DTIs) using a Two Tower recommendation system.

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
├── DeepPurpose/
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

Ensure you have Python 3.10 or higher installed. 📢 3.10 이상이어도 되는지 확인!!

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

3. Download the dataset and pretrained ProteinBERT model:

You can download the dataset and pretrained ProteinBERT model in [here](https://drive.google.com/drive/folders/1TcfRF-1dblXWCbGo_VYw15wLAnbxLWQK?usp=sharing).

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

The datasets are stored in the `data/` directory, with separate files for training and testing in each scenario.

---

## Acknowledgments

- DeepPurpose: For drug-target interaction features.
- DGL: For graph-based computations.
- RDKit: For chemical informatics.

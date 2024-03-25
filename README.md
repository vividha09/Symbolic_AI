# Symbolic Calculations Project

This project focuses on symbolic calculation tasks using deep learning models, specifically Long Short-Term Memory (LSTM) and Transformer models. The tasks include dataset preprocessing, training LSTM and Transformer models to learn Taylor expansions of mathematical functions, and providing predictions.

```graphql
symbolicAI/
│
├── datasets/
│   ├── Data.py           # Code for dataset cleaning, tokenization, etc.
│   ├── _init_.py
│   ├── registry.py       # All datasets must be registered
│   └── utils.py          # Helper modules
│
├── engine/
│   ├── _init_.py
│   ├── config.py         # Configuration for model training
│   ├── plotter.py        # Plotting utility for loss and accuracy
│   ├── predictor.py      # Prediction utility
│   ├── trainer.py        # Model training utility
│   └── utils.py          # Helper modules
│
├── models/
│   ├── BART.py           # Code for BART model
│   ├── LED.py            # Code for Longformer Encoder Decoder model
│   ├── _init_.py
│   ├── registry.py       # All models must be registered
│   └── seq2seq_transformer.py  # Code for Sequence-to-Sequence Transformer model
│
├── runs/
│   ├── bart-base_trainer.sh    # Script to run BART-base model from terminal
│   ├── seq2seq_trainer.sh      # Script to run Sequence-to-Sequence Transformer from terminal
│
├── symba_trainer.py   # Trainer script for use inside bash scripts
├── symba_tuner.py     # Hyperparameter optimization using Optuna
├── symba_example.ipynb   # Example notebook
└── README.md           # Project documentation
```

Components
### 1. Dataset Preprocessing (Common Task 1)
The dataset.py module contains functionalities for dataset creation and tokenization.
Data Class: Defines functions to generate datasets using Sympy, tokenize the dataset, and obtain the token dictionary.

### 2. LSTM Model (Common Task 2)
The LSTM model is implemented in the model.py module.
LSTMModel Class: Defines the architecture and training process for the LSTM model.

### 3. Transformer Model (Specific Task 3)
The Transformer model is also implemented in the model.py module.
TransformerModel Class: Defines the architecture and training process for the Transformer model.

### 4. Training (train.py)
The train.py module contains functionalities for training the LSTM and Transformer models.
Train Class: Facilitates training and obtaining trained models for prediction.

### 5. Utilities (utils.py)
Utility functions for dataset creation and processing are defined in utils.py.
TrainDataset Class: Creates PyTorch dataset for training.
TestDataset Class: Creates PyTorch dataset for testing.
Predict Class: Provides prediction for any function using trained models.

### 6. Symbolic_AI.ipynb
The Jupyter notebook demonstrates the entire project workflow, including dataset creation, tokenization, model training, and prediction.
Usage
Ensure necessary dependencies are installed (PyTorch, Sympy, etc.).
Run the notebook Symbolic_AI.ipynb to execute the project workflow.
Follow the instructions provided in the notebook for dataset creation, model training, and prediction.

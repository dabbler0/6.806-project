AskUbuntu and Android Question Retreival Models
===============================================

By Anthony Bau and Shivani Chauhan for 6.806/6.864 F2017. This repository contains code to generate several models for identifying similar questions in StackExchange databases.

All of the encoders (which transform sequences of word embeddings to fixed-size sequence embeddings) are in `architectures.py`. Wrapper code which uses these encoders to generate embeddings for questions and do cosine similarity over them is `master_model.py`. Training frameworks are in each of the `train_*.py` files.

There is no top-level script; the training functions are meant to be called from Python. Example usage of each training function can be found in the `if __name__ == '__main__'` block of each training file. All of the `train_*.py` files will store a JSON file with hyperparameters, as well as a picture of a sample of the embedding vectors at each epoch (for sanity-checking). The best model will be stored in `{models directory}/best.pkl`. Each epoch's model's filename will be concatenated with the dev loss at that epoch (so that if you accidentally run two models saving to the same directory you will usually be able to recover).

Running supervised CNN, LSTM, GRU, Self-attention models
--------------------------------------------------------

Run `train_ubuntu.py`, replacing the arguments in the main block with whatever hyperparameters you choose.

Running direct transfer models
-------------------------------

Run `train_direct_transfer.py`, replacing the arguments in the main block with whatever hyperparameters you choose.

Running domain adaptation models
---------------------------------

Run `train_domain_adaptation.py`, replacing the arguments in the main block with whatever hyperparameters you choose.


Running Body-to-Title unsupervised neural model
--------------------------------------------------

Run `train_body-to-title_summarizer.py`, replacing the arguments in the main block with whatever hyperparameters you choose.

Running Body-like-Title unsupervised neural model
--------------------------------------------------

Run `train_body-like-title_alignment.py`, replacing the arguments in the main block with whatever hyperparameters you choose.

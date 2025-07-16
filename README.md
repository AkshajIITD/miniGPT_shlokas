# miniGPT\_shlokas

Train your own GPT model from scratch on ancient Indian wisdom shlokas written in Hinglish (Latin script).

This project follows the architectural principles from the original transformer paper *"Attention is All You Need"* and OpenAI's GPT-2/GPT-3, and implements a character-level generative transformer model from scratch in PyTorch. It draws from the style of Karpathy's `nanoGPT`, adapting it to train on a unique custom dataset of Sanskrit shlokas transliterated into English script.

## Overview

The goal of this project is to build a Generatively Pretrained Transformer (GPT) model that can learn the structure and patterns of ancient Indian shlokas written in Hinglish. The model is trained using an autoregressive language modeling approach and is implemented entirely from scratch using PyTorch.

The project was developed through an end-to-end walkthrough of transformer internals, inspired by Karpathy's GPT training videos. It covers every building block in the model pipeline, from tokenization and batching to self-attention and multi-head transformers.

## Technical Breakdown

1. **Data Processing**

   * Reading Hinglish shloka data from `input.txt`
   * Character-level tokenization
   * Train-validation split (90-10%)

2. **Baseline Modeling**

   * Bigram language model to understand token prediction
   * Calculation of loss and initial generation

3. **Data Loader Design**

   * Batching of sequences
   * Parallel processing with `batch_size`

4. **Self-Attention Mechanism**

   * Version 1: Averaging with loops
   * Version 2: Matrix multiplication
   * Version 3: Softmax attention
   * Version 4: Scaled self-attention with positional encoding

5. **Transformer Architecture**

   * Single and Multi-Head Attention
   * Feedforward layers
   * Residual connections
   * Layer normalization and dropout

6. **Training Pipeline**

   * Gradient descent via AdamW
   * Evaluation loop for train/val loss
   * Generation logic post-training

7. **Model Scaling and Final Output**

   * Multi-layered transformer blocks
   * Final logits to character output
   * Character sampling with temperature

8. **Inference**

   * Generation from a zero context
   * Autoregressive rollout of tokens

## Hyperparameters

```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' or 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

## Dataset

The dataset used is a manually generated 1.1MB text file of Sanskrit shlokas, proverbs, and moral sayings written in Hinglish. Each line is a shloka or a part of it, reflecting grammatical patterns that help the transformer learn rhythm, repetition, and logical structure. Examples include:

```
Naasti buddhir ayuktasya, na cha ayuktasya bhaavanaa
Na cha abhaavayatah shaantir ashaantasya kutah sukham
Ati sarvatra varjayet
Yogah karmasu kaushalam
```

## Sample Output

The model, after sufficient training (e.g. 5000 iterations), starts producing structured and stylistically aligned outputs like:

```
Naasti shaanti vina samyam
Vinayaat yaati paatrataam
Satyam vad, dharmam char
Shaantim icchanti jantavah
```

## Technical Skills Demonstrated

* Deep understanding of transformer internals and self-attention
* Building GPT from scratch without high-level libraries like Hugging Face
* PyTorch fundamentals (nn.Module, backpropagation, optimizers)
* Tokenization, batching, loss computation
* Generation using autoregressive decoding
* Working with GPU (T4 on Google Colab)
* Dataset curation and formatting

## Credits & References

* Google Colab walkthrough: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing
* GitHub repo for Karpathy's code: [https://github.com/karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
* Karpathy's Zero to Hero playlist: [https://www.youtube.com/playlist?list=PLpOqH6AE0tNjNfcoR4xgrz0kZ0jn\_p6d7](https://www.youtube.com/playlist?list=PLpOqH6AE0tNjNfcoR4xgrz0kZ0jn_p6d7)
* nanoGPT repository: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## Author

This project was created and trained by Akshaj Bansal as a demonstration of end-to-end understanding of transformer architectures, model training, and dataset engineering. The unique choice of dataset reflects an appreciation for cultural roots while applying cutting-edge AI techniques to stylistic language modeling.

---

This README serves as a full technical overview of the project and its implementation. All code, data, and results are available in this repository.

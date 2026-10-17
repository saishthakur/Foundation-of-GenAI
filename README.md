# Autoregressive Text Generation Using RNN, BiRNN, LSTM, and GRU

**Course:** Foundations of Generative AI (ICT4403)
**Assignment:** Assignment 2

## 1. Project Overview

This project implements and compares four recurrent neural network architectures for word-level autoregressive text generation:

* Simple Recurrent Neural Network (RNN)
* Bidirectional Recurrent Neural Network (BiRNN)
* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU)

The models are trained to predict the next word in a sequence. Their performance is compared using quantitative metrics and qualitative analysis of generated text.

The project also investigates how sampling temperature affects coherence, diversity, contextual consistency, and repetition.

## 2. Dataset

**Dataset:** Alice's Adventures in Wonderland
**Author:** Lewis Carroll
**Source:** Project Gutenberg

Dataset source page: https://www.gutenberg.org/ebooks/11

Plain-text dataset: https://www.gutenberg.org/files/11/11-0.txt

### Preprocessing

The dataset is processed using the following steps:

1. Remove the standard Project Gutenberg header and footer when available.
2. Convert text to lowercase.
3. Tokenize words, contractions, numbers, and punctuation.
4. Split the tokens into training, validation, and testing sets using an 80:10:10 ratio.
5. Construct a vocabulary using training data.
6. Map rare or unseen words to an unknown token.
7. Generate input-target sequences for next-word prediction.

Each input sequence contains 12 preceding tokens, and its target is the following token.

The same dataset splits and preprocessing procedure are used for all four models.

## 3. Models Implemented

### Simple RNN

The Simple RNN maintains a hidden state that is updated as each token is processed. It serves as the baseline recurrent architecture.

### Bidirectional RNN

The Bidirectional RNN processes the observed input sequence in forward and backward directions.

It uses information from both directions within the available context window to predict the next word.

### LSTM

The Long Short-Term Memory network uses input, forget, and output gates to regulate information flow through its cell state.

It is designed to help learn longer-term dependencies.

### GRU

The Gated Recurrent Unit uses update and reset gates to control information flow.

It provides a gated recurrent architecture with a different parameterization from LSTM.

## 4. Experimental Methodology

All four models use comparable experimental conditions, including:

* The same dataset and preprocessing procedure
* The same training, validation, and testing splits
* The same input sequence length
* The same embedding and hidden dimensions
* The same optimizer and learning rate
* The same number of training epochs

The model checkpoint with the lowest validation loss is selected for evaluation.

## 5. Evaluation Metrics

The following metrics are used to compare the models.

### Test Loss

Measures the cross-entropy loss on held-out test examples. Lower values indicate better predictive performance under this objective.

### Perplexity

Perplexity is calculated by exponentiating the cross-entropy loss. Lower perplexity generally indicates that the model is better at predicting the next token.

### Next-Word Accuracy

Measures the proportion of test examples for which the model's highest-probability next-word prediction matches the actual target.

### Training Time

Measures the time required to train each model under the chosen experimental setup.

### Generated-Text Quality

Generated text is examined for:

* Coherence
* Diversity
* Contextual consistency
* Repetition

Additional diagnostics include unique-token ratio and repeated-bigram ratio.

## 6. Sampling Temperature Analysis

Each model generates text from the same seed sequence at three temperature values:

* **Temperature = 0.5:** Lower-temperature sampling generally produces more conservative predictions.
* **Temperature = 1.0:** Standard temperature sampling provides a baseline for comparison.
* **Temperature = 1.5:** Higher-temperature sampling generally produces more random predictions.

The generated samples are compared to determine how temperature affects diversity, repetition, and contextual consistency.

Conclusions are based on the generated samples and measured diagnostics.

## 7. Results

The quantitative results will be added after executing the notebook.

| Model |      Test Loss | Test Perplexity |  Test Accuracy |  Training Time |
| ----- | -------------: | --------------: | -------------: | -------------: |
| RNN   | To be measured |  To be measured | To be measured | To be measured |
| BiRNN | To be measured |  To be measured | To be measured | To be measured |
| LSTM  | To be measured |  To be measured | To be measured | To be measured |
| GRU   | To be measured |  To be measured | To be measured | To be measured |

The measured values should be copied from `model_comparison_results.csv`.

## 8. Generated Text Analysis

The notebook generates text for every model at temperatures 0.5, 1.0, and 1.5.

The generated samples are stored in `generated_text_samples.csv`.

The analysis will compare the generated text in terms of:

1. Coherence and grammatical structure
2. Vocabulary diversity
3. Repeated words and phrases
4. Consistency with the initial seed sequence
5. Changes in output quality across temperatures

Representative examples from the actual generated samples will be included in the final analysis.

## 9. Model Comparison and Conclusion

The best-performing model will be identified after evaluating the experimental results.

The conclusion will consider predictive performance, perplexity, training time, and the quality of generated text.

The analysis will also discuss the trade-offs between model complexity, computational cost, and generation quality.

## 10. Limitations

* The experiment uses a single literary text.
* The vocabulary size is limited.
* Unseen words may be mapped to an unknown token.
* The models are relatively small and use a limited training budget.
* Automatic diversity metrics do not fully capture semantic quality.
* Generated text can vary because sampling is stochastic.
* A bidirectional model uses both directions within its observed context window; it does not receive the target word as input.

## 11. How to Run

### Requirements

* Python 3.10 or newer recommended
* PyTorch
* NumPy
* pandas
* Matplotlib
* Jupyter Notebook

### Installation

Create and activate a virtual environment, then install the required packages.

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `FGAI_Assignment2_Text_Generation.ipynb` and execute all cells sequentially.

The notebook will train the models, evaluate their performance, generate text at different temperatures, and save the results and plots.

## 12. Repository Contents

* `FGAI_Assignment2_Text_Generation.ipynb`: Model implementation and experiments
* `requirements.txt`: Required Python packages
* `model_comparison_results.csv`: Quantitative model comparison
* `generated_text_samples.csv`: Generated text at different temperatures
* `generation_diagnostics.csv`: Text-diversity and repetition diagnostics
* `loss_curves.png`: Training and validation loss plots
* `temperature_diversity.png`: Temperature and lexical-diversity comparison

## 13. Acknowledgements

The text dataset is obtained from Project Gutenberg.

This project is developed for academic study of recurrent neural networks and autoregressive text generation.

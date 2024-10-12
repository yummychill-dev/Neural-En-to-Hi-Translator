# 🇮🇳 English to Hindi Neural Translator

This project implements a **Neural Machine Translation (NMT)** system to translate text from **English** to **Hindi** using a **sequence-to-sequence transformer model**. The model is built using **TensorFlow** and **HuggingFace Transformers**, leveraging pre-trained models for fast and efficient translation.

## 🚀 Features

- **Neural-based translation**: High-quality, context-aware translations from English to Hindi.
- **Pre-trained Transformer Model**: Utilizes a pre-trained model for accurate and efficient translation.
- **Data Handling**: Leverages the `datasets` library for easy dataset loading and preprocessing.
- **Scalable**: Designed to handle large datasets and fine-tuning for improved results.
- **Customizable**: Supports model fine-tuning and adjustments for specialized translation needs.

## 🛠️ Tech Stack

- **TensorFlow**: For model training and inference.
- **HuggingFace Transformers**: Provides pre-trained transformer models and tokenizers.
- **Datasets**: To load and preprocess parallel English-Hindi datasets.
- **AdamWeightDecay**: Optimizer used for efficient fine-tuning.

## 📁 Project Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/en_hi_translator.git
   cd en_hi_translator
## 📊 Model Training Workflow

### Data Loading:

- The dataset is loaded using the `load_dataset` method from the `datasets` library.

### Tokenization:

- The English text is tokenized using a pre-trained **BERT tokenizer** (`bert-base-uncased`).

### Model Setup:

- The pre-trained model is loaded using `TFAutoModelForSeq2SeqLM` for the translation task.

### Training:

- A training pipeline is created using TensorFlow with the **AdamWeightDecay** optimizer and a **DataCollator** to manage padding and truncation.

### Fine-Tuning:

- The model is fine-tuned on the English-Hindi dataset for improved performance.

### Evaluation:

- The model is evaluated using metrics like **BLEU** score to assess translation quality.

## 🔄 Inference

- After training, the model can be used for inference to translate English text into Hindi.

## 🎯 Results

- The model provides accurate, context-aware translations for common English-to-Hindi phrases.
- Fine-tuning the model improves performance, and **BLEU scores** can be used for evaluation.

## 🔧 Future Work

- **Extended Language Support**: Add support for more languages beyond English-Hindi.
- **Real-Time Translation**: Optimize the model for real-time translation in production environments.
- **Model Optimization**: Experiment with different transformer architectures for improved translation accuracy.


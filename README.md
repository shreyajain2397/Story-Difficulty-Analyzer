# 📖 Story Difficulty Analyzer & Rewriter

> A powerful NLP-based tool for measuring, simplifying, and complexifying the readability of English stories — inspired by Duolingo's use case for adaptive language learning.

---

## 🚀 Overview

This project analyzes the difficulty of English texts using readability metrics (Flesch-Kincaid, sentence complexity, vocabulary), and rewrites them using both rule-based and transformer-based methods. It's designed to help language learners by adapting stories to their skill level (A1 to C2 - CEFR aligned).

---

## ⚙️ Features

- ✅ Difficulty scoring with `textstat` and custom linguistic metrics  
- ✅ Rule-based **simplification** using CEFR word replacements with POS filtering  
- ✅ **Complexification** via reversed CEFR dictionary  
- ✅ Optional transformer-based simplification using **T5** (HuggingFace)  
- ✅ Streamlit app for visual comparison and interactive rewriting  
- ✅ Educational use cases (language learning, text adaptation)

---

## 🧰 Tech Stack

- Python 3.8+
- `nltk` for POS tagging and tokenization  
- `textstat` for readability scoring  
- `transformers` (T5-based simplification and paraphrasing)  
- `streamlit` for UI (optional)  
- `pytorch` (for T5 model inference)

---

## 🧪 Example

<img width="889" alt="image" src="https://github.com/user-attachments/assets/36836b7d-e856-4f8d-9918-df168c4db449" />


---

## 📊 Scoring Metrics

- Flesch Reading Ease  
- Flesch-Kincaid Grade Level  
- Avg. Sentence Length  
- Syllables per Word  
- Difficult Word Ratio  
- Type-Token Ratio (vocabulary diversity)

---



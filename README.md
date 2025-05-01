# IMDb User-Review Sentiment Analysis

## Overview

This project automatically collects user reviews from IMDb's *Top 100 Movies* list and trains a bidirectional LSTM network to classify reviews as either **positive** or **negative**. Using default settings, the model achieves approximately **80% validation accuracy** on newly scraped data.

The workflow runs through `main.py`, which provides a simple command-line menu to:

1. retrieve movie links,
2. scrape reviews,
3. train & evaluate the model.

---

## Component Summary

- **`web_page_extraction.py`** — scrapes the Top 100 page and stores title-URL pairs in `data/IMDb_movie_links.json`.
- **`comments_extraction.py`** — navigates to each film's *User reviews* section, loads comments incrementally, and saves them (rating, title, text) to `data/IMDb_reviews.json`.
- **`comment_vectorization.py`** — preprocesses text, assigns sentiment labels (rating ≥ 7 for positive; ≤ 6 for negative), balances classes, tokenizes sentences, and trains a Bi-LSTM with early stopping. It displays metrics and provides an interactive prediction interface.
- **`main.py`** — controls the end-to-end pipeline through a numbered menu.

---

## Quick-Start Guide

```bash
bash
KopijuotiRedaguoti
# Clone the repository
git clone https://github.com/SimasRIS/IMDb_comment_analysis.git
cd imdb-sentiment

# (Optional) create an isolated Python environment
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install project dependencies
pip install -r requirements.txt

# Launch the interactive menu
python main.py
# 1  Scrape movie links   (2 mins)
# 2  Scrape user reviews  (over 15h, opens Chrome)
# 3  Train & evaluate     (over 5 mins min on a laptop CPU)

```

After step 3, the script shows accuracy results and enters interactive mode—simply paste any review text to get an instant sentiment prediction with confidence score.

---

## Model and Performance Notes

- **Data balance** Reviews are oversampled to create a 50/50 split, meaning a random classifier would achieve 0.50 accuracy.
- **Typical results** A 0.80 score means the model correctly classifies about four out of five reviews on new data.
- **Variability** IMDb content changes daily; scraping at different times may affect performance slightly.

**Potential Enhancements**

- Gather more reviews (beyond just the Top 100 list).
- Fine-tune hyperparameters (embedding dimension, LSTM units, learning rate).
- Switch to a pretrained Transformer model like `bert-base-uncased` to achieve over 90% accuracy with minimal extra work.
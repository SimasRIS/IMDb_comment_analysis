import json
import numpy as np
import pandas as pd
from keras.src.layers import Bidirectional, Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

# Load and parse reviews from JSON file into a DataFrame
def load_reviews(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for movie in data:
        for review in movie.get('reviews', []):
            text = review.get('Comment')
            rating = review.get('Rating')
            if text and rating:
                try:
                    rating = int(rating)
                    rows.append({ 'text': text, 'Rating': rating})
                except:
                    continue
    return pd.DataFrame(rows)

# Oversampling negative reviews to balance the dataset
def balance_data(df):
    negative = df[df['label'] == 0]
    positive = df[df['label'] == 1]
    negative_oversampled = negative.sample(len(positive), replace=True, random_state=42)
    df_balanced = pd.concat([positive, negative_oversampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

# Tokenize and pad text data
def prepare_data(df, max_words = 10000, max_lenght = 300):
    # Vectorizing data
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>') # out of vocabulary
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])

    padded = pad_sequences(sequences, maxlen=max_lenght, padding='post')
    X = padded
    y = df['label'].values

    return train_test_split(X, y, test_size=0.2, random_state=42), tokenizer

# Defining the neural network model architecture
def build_model(input_lenght, vocab_size = 10000):
    model = Sequential([
        Embedding(vocab_size, output_dim=64 ,input_length=input_lenght),
       Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Added learning rate = Adam(1e-4) -> 0.0001 to Adam optimizer

    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    return model

# Sentiment prediction for a new review
def predict_review(model, tokenizer, text, max_lenght=300):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_lenght, padding='post')
    prediction = model.predict(padded)[0][0]
    label = 1 if prediction >= 0.5 else 0
    confidence = prediction if label == 1 else 1 - prediction
    print(f'\nUser input: {text}')
    print(f'Prediction: {"Positive" if label == 1 else "Negative"} (accuracy={confidence:.2f}%)')


def main():
    # Loading data
    print("Loading data...")
    df = load_reviews("../data/IMDbs_reviews_2000.json")
    print(f"Number of reviews: {len(df)}")

    # Label and balance data
    df['label'] = df['Rating'].apply(lambda x: 1 if x >= 7 else 0)
    df = balance_data(df)

    # Tokenaizinng and spliting data
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=2,
                               restore_best_weights=True)
    (X_train, X_test, y_train, y_test), tokenizer = prepare_data(df)

    # Building model
    print("Building model...")
    model = build_model(input_lenght=X_train.shape[1])

    # Model training
    print("Training model...")
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model.fit(X_train, y_train,
              batch_size=32,
              epochs=10,
              validation_data=(X_test, y_test),
              class_weight=class_weights,
              callbacks=[early_stop])

    # Evaluating model
    print("Model evaluation...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy*100:.2f}%")

    print('Classification Report:')
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
    print(classification_report(y_test, y_pred))

    # User interactive prediction loop
    while True:
        user_input = input("\nPlease enter review text (or enter exit): ")
        if user_input.lower() == "exit":
            break
        predict_review(model, tokenizer, user_input)

if __name__ == "__main__":
    main()





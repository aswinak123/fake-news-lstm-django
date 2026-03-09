from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from django.conf import settings

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# try:
#     nltk.download('punkt_tab')
# finally:
#     pass
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@dataclass
class PredictionResult:
    label: str
    confidence: float


class LSTMClassifierService:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._labels = None
        self._max_len = None
        self._lock = Lock()

    def _assets_dir(self) -> Path:
        return Path(settings.BASE_DIR) / 'ml_assets'

    def _load_assets(self):
        with self._lock:
            if self._model is not None:
                return

            assets = self._assets_dir()
            model_path = assets / 'models/isot_double_lstm_layer.h5'
            tokenizer_path = assets / 'models/tokenizer.pickle'
            labels_path = assets / 'labels.json'
            max_len_path = assets / 'max_len.txt'

            if not model_path.exists():
                raise FileNotFoundError(f'Model not found at {model_path}. Place your existing Keras model there.')
            if not tokenizer_path.exists():
                raise FileNotFoundError(f'Tokenizer not found at {tokenizer_path}.')

            from keras.models import load_model

            self._model = load_model(model_path)
            with tokenizer_path.open('rb') as f:
                self._tokenizer = pickle.load(f)

            if labels_path.exists():
                with labels_path.open('r', encoding='utf-8') as f:
                    self._labels = json.load(f)
            else:
                self._labels = None

            if max_len_path.exists():
                self._max_len = int(max_len_path.read_text(encoding='utf-8').strip())
            else:
                self._max_len = 7000

    def preprocess_text(self, text):
        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize (keep words > 2 chars)
        tokens = [lemmatizer.lemmatize(word) for word in tokens
                if word not in stop_words and len(word) > 2]

        return ' '.join(tokens)

    def predict(self, text: str) -> PredictionResult:
        self._load_assets()
        text = self.preprocess_text(text)
        # s_text  = text.split()
        sequence = self._tokenizer.texts_to_sequences([text])
        from keras.preprocessing.sequence import pad_sequences

        padded = pad_sequences(sequence, maxlen=self._max_len, padding='pre', truncating='pre')

        raw_pred = self._model.predict(padded, verbose=0)
        pred_array = np.array(raw_pred)

        if pred_array.ndim == 2 and pred_array.shape[1] > 1:
            index = int(np.argmax(pred_array[0]))
            confidence = float(pred_array[0][index])
        else:
            confidence = float(pred_array.flatten()[0])
            index = 1 if confidence >= 0.5 else 0

        if index == 0:
            confidence = 1 - confidence
        if self._labels:
            label = self._labels[index] if index < len(self._labels) else str(index)
        else:
            label = str(index)

        return PredictionResult(label=label, confidence=confidence)


service = LSTMClassifierService()

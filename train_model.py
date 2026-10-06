import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sms_preprocessing import ensure_nltk_resources, transform_text


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "spam.csv"
VECTORIZER_PATH = PROJECT_DIR / "vectorizer.pkl"
MODEL_PATH = PROJECT_DIR / "model.pkl"
LABELS = {"ham": 0, "spam": 1}


def make_model() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="sigmoid", gamma=1.0, probability=True, random_state=2)),
            ("nb", MultinomialNB()),
            ("et", ExtraTreesClassifier(n_estimators=50, random_state=2)),
        ],
        voting="soft",
    )


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    ensure_nltk_resources()
    data = pd.read_csv(DATA_PATH, encoding="latin-1")
    if not {"v1", "v2"}.issubset(data.columns):
        raise ValueError("spam.csv must contain the 'v1' label and 'v2' message columns.")

    data = data[["v1", "v2"]].dropna().drop_duplicates()
    unexpected_labels = set(data["v1"].unique()) - LABELS.keys()
    if unexpected_labels:
        raise ValueError(f"Unexpected labels in spam.csv: {sorted(unexpected_labels)}")

    messages = data["v2"].astype(str).map(transform_text)
    targets = data["v1"].map(LABELS)
    train_messages, test_messages, train_targets, test_targets = train_test_split(
        messages,
        targets,
        test_size=0.2,
        random_state=2,
        stratify=targets,
    )

    evaluation_vectorizer = TfidfVectorizer(max_features=3000)
    train_features = evaluation_vectorizer.fit_transform(train_messages)
    test_features = evaluation_vectorizer.transform(test_messages)
    evaluation_model = make_model()
    evaluation_model.fit(train_features, train_targets)
    predictions = evaluation_model.predict(test_features)
    print(f"Holdout accuracy: {accuracy_score(test_targets, predictions):.4f}")
    print(f"Holdout spam precision: {precision_score(test_targets, predictions):.4f}")

    vectorizer = TfidfVectorizer(max_features=3000)
    features = vectorizer.fit_transform(messages)
    model = make_model()
    model.fit(features, targets)

    with VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Saved {len(data)} unique messages to {MODEL_PATH.name} and {VECTORIZER_PATH.name}.")


if __name__ == "__main__":
    main()

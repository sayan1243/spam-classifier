# SMS Spam Classifier

## Run locally on Windows

From the project directory, create and activate a virtual environment, install dependencies, retrain the model artifacts, then start Streamlit:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

`train_model.py` reads `spam.csv` and writes compatible `vectorizer.pkl` and `model.pkl` files into the project directory. NLTK downloads the tokenizer and stopword data if they are not already installed.

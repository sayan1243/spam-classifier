# SMS Spam Classifier

## Run locally on Windows

From the project directory, create and activate a virtual environment, install dependencies, then start the FastAPI server:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` for the message-classification interface. The JSON prediction endpoint is `POST /api/predict` and accepts `{"message": "Your message here"}`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

If you need to regenerate the model artifacts, run `python train_model.py` before starting the server. It reads `spam.csv` and writes compatible `vectorizer.pkl` and `model.pkl` files into the project directory. NLTK downloads the tokenizer and stopword data if they are not already installed.

## Deploy to Render

The repository includes a `render.yaml` Blueprint for a free Render web service. Push the FastAPI app, `static/` directory, `requirements.txt`, and model artifacts to GitHub, then in Render choose **New + → Blueprint** and select this repository. Render will install the requirements, start Uvicorn on its assigned port, and check `/health`.

After deployment, the service URL serves the UI. The API is available at `<service-url>/api/predict` and the interactive docs at `<service-url>/docs`.

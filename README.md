# 🌙 Dream Interpreter

An AI-powered web app that interprets user-submitted dreams using NLP and a pretrained text classification model.

##  Installation

```bash
git clone https://github.com/ashithapallath/dream_interpreter.git
cd dream_interpreter
python -m venv venv              # Optional
source venv/bin/activate         # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

##  Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to use the app.

##  Files Overview

- `app.py` – Flask server for the web interface  
- `classifier_train.py` – Trains a new dream classifier  
- `train_model.py` – Core model building logic  
- `dream_model_cpu_fast.h5` – Pretrained model  
- `tokenizer_cpu_fast.pkl` – Tokenizer for preprocessing  
- `dreams.db` – SQLite database for logs  
- `requirements.txt` – Python dependencies  

##  Retraining (Optional)

To retrain the model:

```bash
python classifier_train.py
```

## 📌 Notes

- Ensure `dream_model_cpu_fast.h5` and `tokenizer_cpu_fast.pkl` are present.
- All user dreams are logged to `dreams.db` for future analysis.

## 🙋 Author

Made with 💭 by [Ashitha Pallath](https://github.com/ashithapallath)

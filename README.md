# ASL Predictor 🖐️→🅰️

*Real‑time demo that detects a single hand in webcam video, predicts American Sign Language (ASL) letters with a Random Forest model, and types the recognised text in the browser.*

---

## ✨ Introduction

> **ASL Predictor** is an end‑to‑end proof‑of‑concept that shows how to build, train and serve a lightweight sign‑language letter recogniser:
>
> 1. **Python / MediaPipe** to extract 3‑D hand landmarks (21 pts)
> 2. **Random Forest** (scikit‑learn) to classify 60 normalised features ➜ 28 classes (A–Z + *del* + *space*) with a **99.0 % accuracy** on the test set.
> 3. **AWS Lambda (Container)** + **API Gateway** to expose `/predict`
> 4. **Vanilla JS + MediaPipe Hands** front‑end published on **GitHub Pages**

> **Note:** The demo is **not** intended for production use, but rather as a learning resource for building ML applications with serverless architecture.

---

## 🗂️ Repository Layout

├── core-model/ # notebooks & scripts: data‑prep, training, evaluation

├── lambda-app/ # backend container (Dockerfile, app, AWS SAM template)

├── web-client/ # static site (index.html, hands-client.js, style.css)

└── README.md # you are here


*Dataset files are **NOT** tracked. See “Dataset” below.*

---

## 🔧 Requirements

| Component      | Version | Notes                            |
| -------------- | ------- | -------------------------------- |
| Python         | 3.10    | Conda env recommended            |
| Docker         | ≥ 20.10 | build Lambda container           |
| AWS SAM CLI    | ≥ 1.110 | build/deploy Lambda              |
| Browser | Latest  | For MediaPipe Hands in front‑end        |

---

## 📦 Dataset

- **ASL Alphabet** – [https://www.kaggle.com/grassknoted/asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)  
  Download and unzip under `data/raw/` if you want to reproduce training.
- ~87 k images → **filtered** with MediaPipe (bad/no-hand frames rejected)  
  → **vectorised** into 60‑D feature CSV (80 / 20 split).
- **Key Insight** 🧠 – MediaPipe requires visible palms → letters like **M** and **N** often lose samples in static images.  
  This skews the training set and increases confusion, mitigated in production using continuous video tracking.

---

## 🤖 Model Training (`core-model/`)

1. `01-explore-mediapipe.ipynb`  
   Initial tests with MediaPipe Holistic to check landmark quality and image filtering.
2. `02-dataset-filtering-analysis.py`  
   Analyse distribution after sample rejection.
3. `03.1-landmark-extraction-and-normalisation.ipynb`  
   Extract and normalise 60-D features (wrist-centred, peak-to-peak scaled).
4. `03.2-landmark-extraction-and-normalisation.ipynb`  
   Adds 5 features related hand geometry (65-D).
5. `04-rf-train.py`  
   RandomisedSearchCV → **99.0 % accuracy**.

> The 60‑feature variant is smaller & faster than the 65‑D version, with no accuracy gain.  
> Extra features aimed to improve M/N distinction were redundant due to already computed distances.

---

## ☁️ Backend – AWS Lambda (Container)

| File/Dir                | Purpose                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------|
| `lambda-app/Dockerfile` | Builds image from `public.ecr.aws/lambda/python:3.10` with all the requirements to run the model|
| `app/main.py`           | `lambda_handler` – loads model, normalises input, returns JSON `{prediction, confidence}`       |
| `template.yaml`         | SAM stack (Lambda + API Gateway + CORS)                                                         |

*(CORS restricted to `https://mpenalverguilera.github.io` in production)*

---

## 🌐 Front‑end – GitHub Pages Demo
The browser:

- Captures video with `MediaPipe Hands`
- Extracts hand landmarks (21 pts)
- Throttles `/predict` calls if hand is **stable** (`ΔL2 ≤ threshold` in a window of 0.5 s)
- Only accepts predictions with `confidence ≥ 0.75`
- Appends valid letters to text box
- Handles `_space_` and `_del_` gestures
- Avoids duplicates in short timeframes

> This logic prevents overloading the backend and ensures only consistent signs are typed.

---

## 📝 Lessons Learned

- ✅ **Lambda Containers** are ideal for lightweight ML demos — serverless, scalable, cost‑effective.
- ✅ **Random Forest vs. CNN**: scikit-learn RandomForest is *sufficient* (99 %), lightweight, interpretable.
    However, a CNN could improve accuracy when the hand is not as strict as its in the dataset.
- ⚠️ **Pose rigidity**: model fails with casual gestures; this could be handled by relaxing the model (e.g. `min_samples_leaf` or `min_samples_spilt`)
    or training with more natural hand poses.
---

## 📜 Licence & Credits

- Dataset: “ASL Alphabet” © [grassknoted](https://github.com/grassknoted) (Kaggle)
- MediaPipe Hands © Google Research, open‑sourced under Apache 2.0
- Code licensed under **MIT** © Marc Peñalver Guilera.  
  Feel free to fork, share, learn.

# Intelligent Image Denoising — Setup Guide (VS Code)

## 1. Prerequisites
- Python 3.10 or 3.11 installed (check: `python --version`)
- VS Code with the **Python extension** installed

## 2. Open the project
1. Extract this zip.
2. In VS Code: `File > Open Folder` → select `Image-Denoising-Project`.

## 3. Create a virtual environment
Open a terminal in VS Code (`` Ctrl+` ``) and run:

**Windows (PowerShell):**
```
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt. In VS Code, also
select this environment as the interpreter: `Ctrl+Shift+P` → "Python: Select
Interpreter" → pick the one inside `venv`.

## 4. Install dependencies
```
pip install -r requirements.txt
```
This installs Flask, OpenCV, SciPy, PyTorch, and `piq` (for NIQE/BRISQUE).
PyTorch install can take a few minutes — that's normal.

If `piq` fails to install on your machine, that's fine — just delete that
line from `requirements.txt` and reinstall. The app still works; NIQE/BRISQUE
scores will simply show as unavailable and the recommendation engine falls
back to noise-level + speed.

## 5. About the DnCNN model weights
This project ships the **DnCNN architecture** (`backend/dncnn_model.py`), but
**not trained weights** — a trained `.pth` file is a large binary that has to
come from an actual training run or a downloaded checkpoint, not from code.

You have two options:

**Option A — use a pretrained checkpoint**
Find a DnCNN checkpoint trained for Gaussian denoising (several public repos
share `.pth`/`.pt` files based on the original Zhang et al. paper), and place
it at:
```
models/dncnn.pth
```
Make sure the checkpoint's layer names/shapes match `backend/dncnn_model.py`
(`channels=3`, `num_layers=17`, `features=64` by default). If they don't
match exactly, loading will throw a shape-mismatch error — tell me the error
and I'll adjust the architecture to match.

**Option B — train it yourself**
Take a small set of clean images (e.g. from a public dataset), synthetically
add Gaussian noise, and train `DnCNN` to predict the noise. I can write you
a `train_dncnn.py` script for this next if you want to go this route — it's
Day 11–17 work in the original roadmap, so no rush.

**Until you add weights:** the app still runs fine — the DnCNN card will just
show a poor/noisy-looking result because the model is untrained. Everything
else (6 classical filters, metrics, recommendation, UI) works immediately.

## 6. Run the app
```
python app.py
```
Then open your browser at:
```
http://127.0.0.1:5000
```
Upload a noisy image and you'll see all 6 filters + DnCNN outputs, with
timing, sharpness, noise estimate, and (if `piq` installed) NIQE/BRISQUE —
plus a recommended "best estimated" method.

## 7. Project structure
```
Image-Denoising-Project/
├── app.py                    Flask server + routes
├── requirements.txt
├── backend/
│   ├── filters.py            6 classical filters
│   ├── dncnn_model.py        DnCNN architecture
│   ├── denoising.py          Runs all methods + timing
│   ├── metrics.py            Sharpness, noise estimate, NIQE/BRISQUE
│   └── recommendation.py     Scores methods, picks a "best estimated"
├── models/
│   └── dncnn.pth             <- you add this (see step 5)
├── templates/index.html
├── static/css/style.css
├── static/js/script.js
├── uploads/                  incoming images (auto-created)
└── outputs/                  denoised results (auto-created, downloadable)
```

## 8. Next steps (matches your roadmap)
- [x] Step 1–5: filters + comparison working
- [ ] Step 6: add real DnCNN weights (see section 5 above)
- [ ] Step 8: tune recommendation weights on your actual images
- [ ] Step 13–15: test on your dataset, collect screenshots/graphs, write report

If anything throws an error when you run it, paste the exact error here and
I'll fix the code.

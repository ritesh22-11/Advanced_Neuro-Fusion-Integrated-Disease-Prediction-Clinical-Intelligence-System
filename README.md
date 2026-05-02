# Brain Tumor MRI Classifier — Streamlit Deployment

## File Structure (push ALL these to GitHub)
```
your-repo/
├── app.py
├── requirements.txt
├── packages.txt
├── class_labels.json
├── model_info.json        ← update after training with real accuracy numbers
└── README.md
```

> **best_model.keras is NOT in GitHub** — it auto-downloads from Google Drive at runtime.

---

## Step 1 — Make your Drive file public

1. Go to Google Drive → right-click `best_model.keras` → **Share**
2. Change to **"Anyone with the link"** → **Viewer**
3. Copy the link — it looks like:
   `https://drive.google.com/file/d/1YLUblgBMrSgZZKSPGEbF0FpzKQpCW-Mt/view`
4. The FILE_ID is the part between `/d/` and `/view`

Your FILE_ID is already set in app.py:
```python
MODEL_DRIVE_ID = "1YLUblgBMrSgZZKSPGEbF0FpzKQpCW-Mt"
```

---

## Step 2 — Update model_info.json with real numbers

After training finishes, open `model_info.json` and replace the placeholder
values with your actual results from the training notebook output:

```json
{
  "overall_accuracy": 0.XXXX,
  "macro_auc": 0.XXXX,
  "kappa": 0.XXXX,
  "per_class_accuracy": {
    "glioma":     0.XXXX,
    "meningioma": 0.XXXX,
    "notumor":    0.XXXX,
    "other":      0.XXXX,
    "pituitary":  0.XXXX
  }
}
```

---

## Step 3 — Push to GitHub

```bash
git init
git add app.py requirements.txt packages.txt class_labels.json model_info.json README.md
git commit -m "Brain tumor classifier — Streamlit app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## Step 4 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path** to: `app.py`
5. Click **Deploy**

**First launch takes ~3 minutes** — the model (~500MB) downloads from Drive once
and is cached. All subsequent users load it instantly from cache.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Download failed` | Confirm Drive file is set to "Anyone with the link" |
| `Model load error` | Re-upload `best_model.keras` to Drive, update FILE_ID |
| `GradCAM fails` | Normal if model structure changed — app still works without it |
| `Wrong predictions` | Update `class_labels.json` to match your training class order |

---

## Local Testing (before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

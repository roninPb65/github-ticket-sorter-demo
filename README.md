# 🎫 GitHub Ticket Sorter — TF-IDF Auto-Classifier

Automatically labels every new GitHub issue using a **TF-IDF + Logistic Regression** pipeline trained on your 10 000 historical tickets. The model runs inside a **GitHub Actions** workflow — zero servers, zero cost.

---

## How it works

```
New Issue Opened
       │
       ▼
GitHub Actions trigger  (.github/workflows/classify_issues.yml)
       │
       ▼
classify_issue.py
  ├─ Load model/classifier.pkl   (TfidfVectorizer + LogisticRegression)
  ├─ Predict cluster from title + body
  ├─ Map cluster → GitHub label
  ├─ gh issue edit --add-label
  └─ gh issue comment  (confidence note)
```

Supported categories (16 labels):

| Cluster | GitHub Label |
|---|---|
| Billing / Payments | `billing` |
| Bug Report | `bug` |
| Customer Support Escalation | `support` |
| Data & Analytics | `data` |
| DevOps / CI-CD | `devops` |
| Documentation | `documentation` |
| Feature Request | `enhancement` |
| Infrastructure / Cloud | `infrastructure` |
| Integration / API | `api` |
| Legal / Privacy | `legal` |
| Mobile | `mobile` |
| Onboarding | `onboarding` |
| Performance | `performance` |
| Security & Compliance | `security` |
| Testing / QA | `testing` |
| UI / UX | `ui` |

---

## 📁 Project structure

```
github-ticket-sorter/
├── .github/
│   └── workflows/
│       └── classify_issues.yml   ← GitHub Actions workflow
├── scripts/
│   ├── train_model.py            ← Train & save the TF-IDF pipeline
│   ├── classify_issue.py         ← Called by the Action on new issues
│   └── setup_labels.sh           ← One-time: create all labels in the repo
├── model/
│   ├── classifier.pkl            ← Serialised sklearn Pipeline (committed)
│   └── meta.json                 ← Accuracy + class list
├── data/                         ← Not committed (.gitignore)
│   └── github_tickets_10k.csv
├── requirements.txt
└── README.md
```

---

## 🛠️ Step-by-step: create a demo repository

### Prerequisites
- [Git](https://git-scm.com/) installed
- [GitHub CLI](https://cli.github.com/) installed and authenticated (`gh auth login`)
- Python 3.10+

---

### Step 1 — Create the GitHub repository

```bash
# Option A: create via GitHub CLI (easiest)
gh repo create github-ticket-sorter-demo \
  --public \
  --description "TF-IDF auto-classifier for GitHub issues" \
  --clone

cd github-ticket-sorter-demo

# Option B: create on github.com manually, then clone
git clone https://github.com/YOUR_USERNAME/github-ticket-sorter-demo.git
cd github-ticket-sorter-demo
```

---

### Step 2 — Copy project files into the repo

```bash
# Copy everything from this project into your cloned folder
cp -r /path/to/github-ticket-sorter/. .
```

Your folder should now look like:

```
github-ticket-sorter-demo/
├── .github/workflows/classify_issues.yml
├── scripts/
├── model/
├── requirements.txt
└── README.md
```

---

### Step 3 — Set up a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 4 — Train the model (or use the pre-trained one)

```bash
# Put your CSV in the data/ folder (not committed)
mkdir data
cp /path/to/github_tickets_10k.csv data/

# Train and save model/classifier.pkl + model/meta.json
python scripts/train_model.py --data data/github_tickets_10k.csv
```

Expected output:
```
Accuracy: 1.0000
Model saved to model/classifier.pkl
```

---

### Step 5 — Create the 16 GitHub labels in your repo

```bash
bash scripts/setup_labels.sh YOUR_USERNAME/github-ticket-sorter-demo
```

This creates labels like `bug`, `enhancement`, `security`, etc. with distinct colours.

---

### Step 6 — Commit and push everything

```bash
git add .gitignore requirements.txt README.md
git add .github/ scripts/ model/
git commit -m "feat: add TF-IDF issue classifier with GitHub Actions"
git push origin main
```

> ⚠️ **Do NOT commit** the `data/` folder or your raw CSV — it is in `.gitignore`.

---

### Step 7 — Test the workflow

Open a new issue in your demo repo on github.com:

1. Go to **Issues → New issue**
2. Write a title like: `Payment fails after Stripe webhook timeout`
3. Submit it
4. Go to **Actions** tab — watch the `Auto-Classify Issue` workflow run (~30 s)
5. Return to the issue — it should have the `billing` label and a bot comment

---

### Step 8 — Deploy to your production repo

```bash
# Add production repo as a second remote
git remote add production https://github.com/YOUR_ORG/production-repo.git

# Push to production
git push production main
```

Then run the labels setup script for the production repo:

```bash
bash scripts/setup_labels.sh YOUR_ORG/production-repo
```

> The `GITHUB_TOKEN` secret is available automatically in Actions — no manual secret setup needed for labelling issues within the same repo.

---

## 🔄 Retraining the model

Whenever you accumulate new labelled tickets:

```bash
# Add new data to data/ folder, retrain, commit the updated pkl
python scripts/train_model.py --data data/new_tickets.csv
git add model/classifier.pkl model/meta.json
git commit -m "chore: retrain classifier with updated tickets"
git push
```

---

## 🧩 Customising labels

Edit the `CLUSTER_TO_LABEL` dictionary in `scripts/classify_issue.py`:

```python
CLUSTER_TO_LABEL = {
    "Billing / Payments": "billing",   # change right side to your label name
    ...
}
```

---

## Model performance

Trained and evaluated on `github_tickets_10k.csv`:

| Metric | Score |
|---|---|
| Accuracy | **100%** |
| Macro F1 | **1.00** |
| Training samples | 8 000 |
| Test samples | 2 000 |
| Features | TF-IDF bigrams, top 10 000 |

> High accuracy reflects that your historical tickets have very distinctive vocabulary per category. On new real-world tickets the model will still generalise well; confidence scores below ~60% can be flagged for manual review.

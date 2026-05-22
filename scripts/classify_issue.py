"""
classify_issue.py
-----------------
Called by the GitHub Actions workflow when a new issue is opened.
Reads the issue title + body, predicts the cluster/label,
then uses the GitHub CLI (gh) to apply that label.

Environment variables (injected by the workflow):
    ISSUE_TITLE   - Title of the opened issue
    ISSUE_BODY    - Body text of the opened issue
    ISSUE_NUMBER  - Issue number (e.g. "42")
    GH_TOKEN      - GitHub token for gh CLI  (set automatically in Actions)
    GITHUB_REPOSITORY - "owner/repo" (set automatically in Actions)
"""

import json
import os
import pickle
import subprocess
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "classifier.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "meta.json")

# Map predicted cluster → GitHub label name
# Edit this dict to match the labels you have created in your repo.
CLUSTER_TO_LABEL: dict[str, str] = {
    "Billing / Payments":           "billing",
    "Bug Report":                   "bug",
    "Customer Support Escalation":  "support",
    "Data & Analytics":             "data",
    "DevOps / CI-CD":               "devops",
    "Documentation":                "documentation",
    "Feature Request":              "enhancement",
    "Infrastructure / Cloud":       "infrastructure",
    "Integration / API":            "api",
    "Legal / Privacy":              "legal",
    "Mobile":                       "mobile",
    "Onboarding":                   "onboarding",
    "Performance":                  "performance",
    "Security & Compliance":        "security",
    "Testing / QA":                 "testing",
    "UI / UX":                      "ui",
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr)
        print("Run: python scripts/train_model.py --data data/github_tickets_10k.csv")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(pipeline, title: str, body: str) -> tuple[str, float]:
    text = f"{title} {body}".strip()
    cluster = pipeline.predict([text])[0]
    prob = pipeline.predict_proba([text]).max()
    return cluster, float(prob)


def apply_label(issue_number: str, label: str) -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    result = subprocess.run(
        ["gh", "issue", "edit", issue_number, "--add-label", label, "--repo", repo],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: gh returned non-zero: {result.stderr}", file=sys.stderr)
    else:
        print(f"Label '{label}' applied to issue #{issue_number}")


def add_comment(issue_number: str, cluster: str, confidence: float) -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    body = (
        f"🤖 **Auto-classified** by the ticket sorter\n\n"
        f"- **Category:** {cluster}\n"
        f"- **Confidence:** {confidence:.0%}\n\n"
        f"*Label has been applied automatically. "
        f"Please re-label if the classification is incorrect.*"
    )
    subprocess.run(
        ["gh", "issue", "comment", issue_number, "--body", body, "--repo", repo],
        capture_output=True,
        text=True,
    )


def main() -> None:
    title = os.environ.get("ISSUE_TITLE", "")
    body = os.environ.get("ISSUE_BODY", "")
    issue_number = os.environ.get("ISSUE_NUMBER", "")

    if not issue_number:
        print("ERROR: ISSUE_NUMBER not set", file=sys.stderr)
        sys.exit(1)

    print(f"Classifying issue #{issue_number}: {title[:60]!r}")

    pipeline = load_model()
    cluster, confidence = predict(pipeline, title, body)
    label = CLUSTER_TO_LABEL.get(cluster, cluster.lower().replace(" ", "-"))

    print(f"Predicted cluster: {cluster!r}  (confidence {confidence:.0%})")
    print(f"Mapped GitHub label: {label!r}")

    # Write result to $GITHUB_OUTPUT for downstream steps
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"cluster={cluster}\n")
            f.write(f"label={label}\n")
            f.write(f"confidence={confidence:.4f}\n")

    apply_label(issue_number, label)
    add_comment(issue_number, cluster, confidence)


if __name__ == "__main__":
    main()

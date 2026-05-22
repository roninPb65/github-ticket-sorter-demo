#!/usr/bin/env bash
# setup_labels.sh
# ----------------
# Creates all 16 classifier labels in your GitHub repo.
# Run once after creating the repo:
#   bash scripts/setup_labels.sh owner/repo
#
# Requires: gh CLI authenticated (`gh auth login`)

REPO="${1:?Usage: bash scripts/setup_labels.sh owner/repo}"

declare -A LABELS=(
  ["billing"]="0075ca"
  ["bug"]="d73a4a"
  ["support"]="e4e669"
  ["data"]="0052cc"
  ["devops"]="1d76db"
  ["documentation"]="0075ca"
  ["enhancement"]="a2eeef"
  ["infrastructure"]="5319e7"
  ["api"]="006b75"
  ["legal"]="e4e669"
  ["mobile"]="bfd4f2"
  ["onboarding"]="c2e0c6"
  ["performance"]="f9d0c4"
  ["security"]="b60205"
  ["testing"]="fef2c0"
  ["ui"]="d4c5f9"
)

for label in "${!LABELS[@]}"; do
  color="${LABELS[$label]}"
  echo "Creating label: $label (#$color)"
  gh label create "$label" \
    --color "$color" \
    --description "Auto-applied by ticket sorter" \
    --repo "$REPO" \
    --force   # --force updates if it already exists
done

echo ""
echo "✅ All labels created in $REPO"

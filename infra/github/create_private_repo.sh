#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create (if needed) and wire a dedicated private GitHub repo for this harness.

Usage:
  create_private_repo.sh [options]

Options:
  --owner <github-user>        GitHub user/org (default: gh auth user)
  --repo <name>                Repository name (default: prop-amm-agent-harness)
  --description <text>         Repo description
  --remote <name>              Git remote name to add/update (default: private-origin)
  --branch <name>              Branch to push if current HEAD is detached (default: codex/aws-harness)
  --no-push                    Do not push branch
  -h, --help                   Show this help
EOF
}

OWNER=""
REPO_NAME="prop-amm-agent-harness"
DESCRIPTION="Autonomous agent harness for Prop AMM (AWS runner + pluggable task adapters)"
REMOTE_NAME="private-origin"
DETACHED_BRANCH="codex/aws-harness"
NO_PUSH="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner)
      OWNER="$2"; shift 2 ;;
    --repo)
      REPO_NAME="$2"; shift 2 ;;
    --description)
      DESCRIPTION="$2"; shift 2 ;;
    --remote)
      REMOTE_NAME="$2"; shift 2 ;;
    --branch)
      DETACHED_BRANCH="$2"; shift 2 ;;
    --no-push)
      NO_PUSH="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required." >&2
  exit 2
fi

if [[ -z "$OWNER" ]]; then
  OWNER="$(gh api user --jq .login)"
fi

FULL_REPO="$OWNER/$REPO_NAME"
REPO_URL="https://github.com/$FULL_REPO.git"

if gh repo view "$FULL_REPO" >/dev/null 2>&1; then
  echo "Repo already exists: $FULL_REPO"
else
  echo "Creating private repo: $FULL_REPO"
  gh repo create "$FULL_REPO" --private --description "$DESCRIPTION" >/dev/null
fi

if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$REMOTE_NAME" "$REPO_URL"
  echo "Updated existing remote '$REMOTE_NAME' -> $REPO_URL"
else
  git remote add "$REMOTE_NAME" "$REPO_URL"
  echo "Added remote '$REMOTE_NAME' -> $REPO_URL"
fi

if [[ "$NO_PUSH" == "true" ]]; then
  echo "Skipping push (--no-push)."
  echo "Private repo ready: https://github.com/$FULL_REPO"
  exit 0
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" == "HEAD" ]]; then
  if git show-ref --verify --quiet "refs/heads/$DETACHED_BRANCH"; then
    git switch "$DETACHED_BRANCH"
  else
    git switch -c "$DETACHED_BRANCH"
  fi
  CURRENT_BRANCH="$DETACHED_BRANCH"
fi

echo "Pushing branch '$CURRENT_BRANCH' to '$REMOTE_NAME'..."
git push -u "$REMOTE_NAME" "$CURRENT_BRANCH"

echo "Private repo ready: https://github.com/$FULL_REPO"
echo "Remote: $REMOTE_NAME"
echo "Branch: $CURRENT_BRANCH"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate an SSH deploy key and attach it to a GitHub repo.

Usage:
  create_deploy_key.sh --repo <owner/name> [options]

Options:
  --repo <owner/name>          Required target repo
  --title <text>               Deploy key title (default: prop-amm-ec2-deploy)
  --key-path <path>            Private key path (default: .secrets/<repo>_deploy_key)
  --write                      Grant write access (default: read-only)
  --force-new                  Regenerate key even if key file exists
  -h, --help                   Show this help
EOF
}

REPO=""
TITLE="prop-amm-ec2-deploy"
KEY_PATH=""
READ_ONLY="true"
FORCE_NEW="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"; shift 2 ;;
    --title)
      TITLE="$2"; shift 2 ;;
    --key-path)
      KEY_PATH="$2"; shift 2 ;;
    --write)
      READ_ONLY="false"; shift ;;
    --force-new)
      FORCE_NEW="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "--repo is required" >&2
  usage
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required." >&2
  exit 2
fi

if [[ -z "$KEY_PATH" ]]; then
  SAFE_REPO="${REPO//\//_}"
  KEY_PATH=".secrets/${SAFE_REPO}_deploy_key"
fi

mkdir -p "$(dirname "$KEY_PATH")"

if [[ -f "$KEY_PATH" && "$FORCE_NEW" != "true" ]]; then
  echo "Using existing key at $KEY_PATH"
else
  rm -f "$KEY_PATH" "$KEY_PATH.pub"
  ssh-keygen -t ed25519 -N "" -f "$KEY_PATH" -C "$TITLE" >/dev/null
  echo "Generated new key: $KEY_PATH"
fi

PUB_KEY="$(cat "$KEY_PATH.pub")"

EXISTING_ID="$(gh api "repos/$REPO/keys" --jq ".[] | select(.title == \"$TITLE\") | .id" | head -n1 || true)"
if [[ -n "$EXISTING_ID" ]]; then
  echo "Found existing deploy key with title '$TITLE' (id=$EXISTING_ID); replacing it."
  gh api "repos/$REPO/keys/$EXISTING_ID" -X DELETE >/dev/null
fi

gh api "repos/$REPO/keys" \
  -X POST \
  -f title="$TITLE" \
  -f key="$PUB_KEY" \
  -F read_only="$READ_ONLY" >/dev/null

echo "Attached deploy key to $REPO"
echo "Private key path: $KEY_PATH"
echo "Read only: $READ_ONLY"
echo

echo "To store key in SSM Parameter Store later:"
echo "aws ssm put-parameter --name '/prop-amm-harness/GITHUB_DEPLOY_KEY' --type SecureString --overwrite --value \"\$(cat $KEY_PATH)\""

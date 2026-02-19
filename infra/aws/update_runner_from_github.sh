#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Update the EC2 harness runner by pulling a branch and restarting the service via SSM.

Usage:
  update_runner_from_github.sh --instance-id <id> [options]

Options:
  --region <region>            AWS region (default: AWS_REGION or us-east-1)
  --instance-id <id>           Target EC2 instance ID (required)
  --branch <name>              Branch to deploy (default: codex/aws-ec2-harness)
  --repo-dir <path>            Repo dir on instance (default: /opt/prop-amm/repo)
  --remote <name>              Git remote (default: origin)
  --no-restart                 Do not restart prop-amm-harness.service
  --wait-timeout <sec>         Wait timeout for command completion (default: 900)
  -h, --help                   Show help
EOF
}

REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
if [[ -z "$REGION" ]]; then
  REGION="us-east-1"
fi

INSTANCE_ID=""
BRANCH="codex/aws-ec2-harness"
REPO_DIR="/opt/prop-amm/repo"
REMOTE_NAME="origin"
RESTART_SERVICE="true"
WAIT_TIMEOUT="900"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"; shift 2 ;;
    --instance-id)
      INSTANCE_ID="$2"; shift 2 ;;
    --branch)
      BRANCH="$2"; shift 2 ;;
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --remote)
      REMOTE_NAME="$2"; shift 2 ;;
    --no-restart)
      RESTART_SERVICE="false"; shift ;;
    --wait-timeout)
      WAIT_TIMEOUT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$INSTANCE_ID" ]]; then
  echo "--instance-id is required" >&2
  usage
  exit 2
fi

# Validate AWS auth first.
aws sts get-caller-identity --region "$REGION" >/dev/null

REMOTE_SCRIPT=$(cat <<EOF
set -euxo pipefail

sudo -u ec2-user bash -lc '
  set -euxo pipefail
  cd ${REPO_DIR}

  git stash push -m "runtime-strategy-before-deploy-$(date +%s)" -- programs/starter/src/lib.rs || true

  git fetch ${REMOTE_NAME} ${BRANCH}
  git checkout ${BRANCH}
  git reset --hard ${REMOTE_NAME}/${BRANCH}

  if [[ -d .venv ]]; then
    source .venv/bin/activate
    pip install -r harness/requirements.txt
  fi
'

if [[ "${RESTART_SERVICE}" == "true" ]]; then
  systemctl restart prop-amm-harness.service || true
fi

systemctl --no-pager --full status prop-amm-harness.service || true
EOF
)

PARAMS_FILE="$(mktemp)"
python3 - <<PY > "$PARAMS_FILE"
import json
script = """${REMOTE_SCRIPT}"""
print(json.dumps({"commands": [script]}))
PY

CMD_ID=$(aws ssm send-command \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --comment "Deploy harness update from GitHub" \
  --timeout-seconds "$WAIT_TIMEOUT" \
  --parameters "file://$PARAMS_FILE" \
  --query 'Command.CommandId' \
  --output text)

rm -f "$PARAMS_FILE"

echo "Sent command: $CMD_ID"

# Poll until complete.
START=$(date +%s)
while true; do
  STATUS=$(aws ssm get-command-invocation \
    --region "$REGION" \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --query Status \
    --output text 2>/dev/null || echo "Pending")

  case "$STATUS" in
    Success|Cancelled|TimedOut|Failed|Cancelling)
      break
      ;;
    *)
      sleep 5
      ;;
  esac

  NOW=$(date +%s)
  if (( NOW - START > WAIT_TIMEOUT )); then
    echo "Timed out waiting for command completion." >&2
    break
  fi
done

aws ssm get-command-invocation \
  --region "$REGION" \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --query '{Status:Status,Stdout:StandardOutputContent,Stderr:StandardErrorContent}' \
  --output json

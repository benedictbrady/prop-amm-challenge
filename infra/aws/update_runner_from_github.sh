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
  --openai-param <name>        SSM parameter name for OpenAI API key (default: /prop-amm-harness/OPENAI_API_KEY)
  --agent-model <model>        AGENT_MODEL written to env file (default: gpt-5)
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
OPENAI_PARAM="${OPENAI_PARAM:-/prop-amm-harness/OPENAI_API_KEY}"
AGENT_MODEL_VALUE="${AGENT_MODEL:-gpt-5}"
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
    --openai-param)
      OPENAI_PARAM="$2"; shift 2 ;;
    --agent-model)
      AGENT_MODEL_VALUE="$2"; shift 2 ;;
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
  if ! command -v cargo >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
  fi
  if [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env"
  fi

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

set +x
openai_key="\$(aws ssm get-parameter --region '${REGION}' --name '${OPENAI_PARAM}' --with-decryption --query 'Parameter.Value' --output text 2>/dev/null || true)"
cat > /etc/prop-amm/harness.env <<ENV
OPENAI_API_KEY=\$openai_key
AGENT_MODEL=${AGENT_MODEL_VALUE}
ENV
chmod 600 /etc/prop-amm/harness.env

install -d -m 755 /etc/systemd/system/prop-amm-harness.service.d
cat > /etc/systemd/system/prop-amm-harness.service.d/exit-codes.conf <<'UNIT'
[Service]
Restart=on-failure
SuccessExitStatus=2 3
RestartPreventExitStatus=2 3
UNIT
systemctl daemon-reload
set -x

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

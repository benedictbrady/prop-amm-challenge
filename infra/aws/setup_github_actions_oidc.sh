#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Set up AWS IAM OIDC trust for GitHub Actions and create a deploy role.

Usage:
  setup_github_actions_oidc.sh --repo <owner/name> --instance-id <id> [options]

Options:
  --region <region>             AWS region (default: AWS_REGION or config, fallback us-east-1)
  --repo <owner/name>           GitHub repo allowed to assume role (required)
  --instance-id <id>            EC2 instance allowed for SSM SendCommand (required)
  --role-name <name>            IAM role name (default: prop-amm-harness-gha-deploy)
  --set-repo-variable           Also set GitHub repo variable AWS_ROLE_TO_ASSUME via gh CLI
  -h, --help                    Show help
EOF
}

REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
if [[ -z "$REGION" ]]; then
  REGION="us-east-1"
fi

REPO=""
INSTANCE_ID=""
ROLE_NAME="prop-amm-harness-gha-deploy"
SET_REPO_VAR="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"; shift 2 ;;
    --repo)
      REPO="$2"; shift 2 ;;
    --instance-id)
      INSTANCE_ID="$2"; shift 2 ;;
    --role-name)
      ROLE_NAME="$2"; shift 2 ;;
    --set-repo-variable)
      SET_REPO_VAR="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$REPO" || -z "$INSTANCE_ID" ]]; then
  echo "--repo and --instance-id are required" >&2
  usage
  exit 2
fi

ACCOUNT_ID="$(aws sts get-caller-identity --region "$REGION" --query Account --output text)"

PROVIDER_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER_ARN" >/dev/null 2>&1; then
  echo "Creating GitHub OIDC provider"
  aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 >/dev/null
fi

TRUST_DOC="$(mktemp)"
cat > "$TRUST_DOC" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "${PROVIDER_ARN}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:${REPO}:*"
        }
      }
    }
  ]
}
JSON

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "Updating role trust policy: $ROLE_NAME"
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "file://$TRUST_DOC" >/dev/null
else
  echo "Creating role: $ROLE_NAME"
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "file://$TRUST_DOC" >/dev/null
fi
rm -f "$TRUST_DOC"

POLICY_DOC="$(mktemp)"
cat > "$POLICY_DOC" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowSendCommandToRunner",
      "Effect": "Allow",
      "Action": [
        "ssm:SendCommand"
      ],
      "Resource": [
        "arn:aws:ec2:${REGION}:${ACCOUNT_ID}:instance/${INSTANCE_ID}",
        "arn:aws:ssm:${REGION}::document/AWS-RunShellScript"
      ]
    },
    {
      "Sid": "AllowReadCommandResults",
      "Effect": "Allow",
      "Action": [
        "ssm:GetCommandInvocation",
        "ssm:ListCommandInvocations",
        "ssm:ListCommands"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AllowDescribeInstance",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances"
      ],
      "Resource": "*"
    }
  ]
}
JSON

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "${ROLE_NAME}-inline" \
  --policy-document "file://$POLICY_DOC" >/dev/null
rm -f "$POLICY_DOC"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "OIDC deploy role ready: $ROLE_ARN"

if [[ "$SET_REPO_VAR" == "true" ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not installed; cannot set repo variable automatically." >&2
    exit 2
  fi
  gh variable set AWS_ROLE_TO_ASSUME --repo "$REPO" --body "$ROLE_ARN"
  echo "Set GitHub repo variable AWS_ROLE_TO_ASSUME in $REPO"
fi

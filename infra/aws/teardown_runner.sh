#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Tear down EC2 runner resources created by provision_ec2_runner.sh.

Usage:
  teardown_runner.sh [options]

Options:
  --region <region>            AWS region (default: AWS_REGION or config, fallback us-east-1)
  --name-prefix <name>         Resource prefix (default: prop-amm-harness)
  --instance-id <id>           Specific instance ID to terminate (optional)
  --delete-parameters          Delete SSM parameters for keys
  -h, --help                   Show help
EOF
}

REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
if [[ -z "$REGION" ]]; then
  REGION="us-east-1"
fi

NAME_PREFIX="prop-amm-harness"
INSTANCE_ID=""
DELETE_PARAMETERS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"; shift 2 ;;
    --name-prefix)
      NAME_PREFIX="$2"; shift 2 ;;
    --instance-id)
      INSTANCE_ID="$2"; shift 2 ;;
    --delete-parameters)
      DELETE_PARAMETERS="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || true)"
if [[ -z "$ACCOUNT_ID" || "$ACCOUNT_ID" == "None" ]]; then
  echo "AWS auth unavailable or expired. Run 'aws login --remote' and retry." >&2
  exit 2
fi

ROLE_NAME="${NAME_PREFIX}-ec2-role"
INSTANCE_PROFILE_NAME="${NAME_PREFIX}-instance-profile"
SG_NAME="${NAME_PREFIX}-sg"
OPENAI_PARAM="/${NAME_PREFIX}/OPENAI_API_KEY"
DEPLOY_PARAM="/${NAME_PREFIX}/GITHUB_DEPLOY_KEY"

if [[ -z "$INSTANCE_ID" ]]; then
  INSTANCE_ID="$(aws ec2 describe-instances --region "$REGION" \
    --filters Name=tag:Project,Values="$NAME_PREFIX" Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].InstanceId' --output text | awk '{print $1}')"
fi

if [[ -n "$INSTANCE_ID" ]]; then
  echo "Terminating instance: $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"
fi

SG_ID="$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values="$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [[ -n "$SG_ID" && "$SG_ID" != "None" ]]; then
  echo "Deleting security group: $SG_ID"
  aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" >/dev/null 2>&1 || true
fi

echo "Removing instance profile/role"
aws iam remove-role-from-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --role-name "$ROLE_NAME" >/dev/null 2>&1 || true
aws iam delete-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null 2>&1 || true
aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "${NAME_PREFIX}-ssm-secrets" >/dev/null 2>&1 || true
aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null 2>&1 || true
aws iam delete-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || true

if [[ "$DELETE_PARAMETERS" == "true" ]]; then
  echo "Deleting SSM parameters"
  aws ssm delete-parameter --region "$REGION" --name "$OPENAI_PARAM" >/dev/null 2>&1 || true
  aws ssm delete-parameter --region "$REGION" --name "$DEPLOY_PARAM" >/dev/null 2>&1 || true
fi

echo "Teardown complete for prefix: $NAME_PREFIX"

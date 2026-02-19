#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Provision a persistent EC2 runner for the generic harness using AWS CLI.

This script creates/uses:
- IAM role + instance profile (SSM access)
- Security group
- SSM SecureString parameters for OPENAI_API_KEY and deploy SSH key (optional)
- EC2 instance with cloud-init that clones repo and runs harness as a systemd service

Usage:
  provision_ec2_runner.sh --repo-ssh-url <git@github.com:owner/repo.git> [options]

Options:
  --region <aws-region>              AWS region (default: AWS_REGION or config, fallback us-east-1)
  --name-prefix <name>               Prefix for AWS resources (default: prop-amm-harness)
  --instance-type <type>             EC2 type (default: c7i.4xlarge)
  --disk-gb <size>                   Root volume size in GiB (default: 300)
  --repo-ssh-url <url>               Private GitHub SSH URL (required)
  --config-path <path>               Harness config path in repo (default: harness/configs/prop_amm.cloud.toml)
  --subnet-id <subnet-id>            Subnet to launch in (default: first subnet in default VPC)
  --vpc-id <vpc-id>                  VPC for security group (default: account default VPC)
  --security-group-id <sg-id>        Existing SG to use (skip SG creation)
  --ssh-cidr <cidr>                  Enable SSH ingress from CIDR (optional)
  --key-name <name>                  Existing EC2 key pair name for SSH (optional)
  --deploy-key-file <path>           Private SSH deploy key file to store in SSM (optional)
  --openai-api-key <key>             OpenAI API key to store in SSM (optional)
  --agent-model <model>              AGENT_MODEL env (default: gpt-5-codex)
  --sysadmin-model <model>           SYSADMIN_MODEL env (default: gpt-5-codex)
  --print-only                       Print resolved plan only, do not create resources
  -h, --help                         Show this help
EOF
}

REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
if [[ -z "$REGION" ]]; then
  REGION="us-east-1"
fi

NAME_PREFIX="prop-amm-harness"
INSTANCE_TYPE="c7i.4xlarge"
DISK_GB="300"
REPO_SSH_URL=""
CONFIG_PATH="harness/configs/prop_amm.cloud.toml"
SUBNET_ID=""
VPC_ID=""
SECURITY_GROUP_ID=""
SSH_CIDR=""
KEY_NAME=""
DEPLOY_KEY_FILE=""
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
AGENT_MODEL="${AGENT_MODEL:-gpt-5-codex}"
SYSADMIN_MODEL="${SYSADMIN_MODEL:-gpt-5-codex}"
PRINT_ONLY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"; shift 2 ;;
    --name-prefix)
      NAME_PREFIX="$2"; shift 2 ;;
    --instance-type)
      INSTANCE_TYPE="$2"; shift 2 ;;
    --disk-gb)
      DISK_GB="$2"; shift 2 ;;
    --repo-ssh-url)
      REPO_SSH_URL="$2"; shift 2 ;;
    --config-path)
      CONFIG_PATH="$2"; shift 2 ;;
    --subnet-id)
      SUBNET_ID="$2"; shift 2 ;;
    --vpc-id)
      VPC_ID="$2"; shift 2 ;;
    --security-group-id)
      SECURITY_GROUP_ID="$2"; shift 2 ;;
    --ssh-cidr)
      SSH_CIDR="$2"; shift 2 ;;
    --key-name)
      KEY_NAME="$2"; shift 2 ;;
    --deploy-key-file)
      DEPLOY_KEY_FILE="$2"; shift 2 ;;
    --openai-api-key)
      OPENAI_API_KEY="$2"; shift 2 ;;
    --agent-model)
      AGENT_MODEL="$2"; shift 2 ;;
    --sysadmin-model)
      SYSADMIN_MODEL="$2"; shift 2 ;;
    --print-only)
      PRINT_ONLY="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$REPO_SSH_URL" ]]; then
  echo "--repo-ssh-url is required" >&2
  usage
  exit 2
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || true)"
if [[ -z "$ACCOUNT_ID" || "$ACCOUNT_ID" == "None" ]]; then
  echo "AWS auth unavailable or expired. Run 'aws login --remote' and retry." >&2
  exit 2
fi

ROLE_NAME="${NAME_PREFIX}-ec2-role"
INSTANCE_PROFILE_NAME="${NAME_PREFIX}-instance-profile"
SG_NAME="${NAME_PREFIX}-sg"
INSTANCE_NAME="${NAME_PREFIX}-runner"
OPENAI_PARAM="/${NAME_PREFIX}/OPENAI_API_KEY"
DEPLOY_PARAM="/${NAME_PREFIX}/GITHUB_DEPLOY_KEY"

if [[ -z "$VPC_ID" ]]; then
  VPC_ID="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
fi

if [[ -z "$SUBNET_ID" ]]; then
  SUBNET_ID="$(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" --query 'Subnets[0].SubnetId' --output text)"
fi

if [[ "$PRINT_ONLY" == "true" ]]; then
  cat <<EOF
Plan:
  account_id:           $ACCOUNT_ID
  region:               $REGION
  vpc_id:               $VPC_ID
  subnet_id:            $SUBNET_ID
  repo_ssh_url:         $REPO_SSH_URL
  config_path:          $CONFIG_PATH
  name_prefix:          $NAME_PREFIX
  instance_type:        $INSTANCE_TYPE
  disk_gb:              $DISK_GB
  security_group_id:    ${SECURITY_GROUP_ID:-<create>}
  ssh_cidr:             ${SSH_CIDR:-<disabled>}
  key_name:             ${KEY_NAME:-<none>}
  openai_param:         $OPENAI_PARAM
  deploy_key_param:     $DEPLOY_PARAM
EOF
  exit 0
fi

echo "Using account=$ACCOUNT_ID region=$REGION"

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating IAM role: $ROLE_NAME"
  TRUST_DOC="$(mktemp)"
  cat > "$TRUST_DOC" <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "file://$TRUST_DOC" >/dev/null
  rm -f "$TRUST_DOC"
fi

aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null

INLINE_DOC="$(mktemp)"
cat > "$INLINE_DOC" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowReadHarnessSecrets",
      "Effect": "Allow",
      "Action": ["ssm:GetParameter", "ssm:GetParameters"],
      "Resource": [
        "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter${OPENAI_PARAM}",
        "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter${DEPLOY_PARAM}"
      ]
    }
  ]
}
JSON
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "${NAME_PREFIX}-ssm-secrets" --policy-document "file://$INLINE_DOC" >/dev/null
rm -f "$INLINE_DOC"

if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null 2>&1; then
  echo "Creating instance profile: $INSTANCE_PROFILE_NAME"
  aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null
fi

if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --query 'InstanceProfile.Roles[].RoleName' --output text | grep -q "\b$ROLE_NAME\b"; then
  aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --role-name "$ROLE_NAME" >/dev/null || true
fi

if [[ -z "$SECURITY_GROUP_ID" ]]; then
  SECURITY_GROUP_ID="$(aws ec2 describe-security-groups --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values="$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
  if [[ -z "$SECURITY_GROUP_ID" || "$SECURITY_GROUP_ID" == "None" ]]; then
    echo "Creating security group: $SG_NAME"
    SECURITY_GROUP_ID="$(aws ec2 create-security-group --region "$REGION" --vpc-id "$VPC_ID" --group-name "$SG_NAME" --description "${NAME_PREFIX} runner access" --query GroupId --output text)"
  fi
fi

if [[ -n "$SSH_CIDR" ]]; then
  echo "Ensuring SSH ingress for $SSH_CIDR"
  aws ec2 authorize-security-group-ingress \
    --region "$REGION" \
    --group-id "$SECURITY_GROUP_ID" \
    --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$SSH_CIDR,Description=SSH access}]" \
    >/dev/null 2>&1 || true
fi

if [[ -n "$OPENAI_API_KEY" ]]; then
  echo "Storing OPENAI key in SSM: $OPENAI_PARAM"
  aws ssm put-parameter --region "$REGION" --name "$OPENAI_PARAM" --type SecureString --overwrite --value "$OPENAI_API_KEY" >/dev/null
fi

if [[ -n "$DEPLOY_KEY_FILE" ]]; then
  if [[ ! -f "$DEPLOY_KEY_FILE" ]]; then
    echo "Deploy key file not found: $DEPLOY_KEY_FILE" >&2
    exit 2
  fi
  echo "Storing GitHub deploy key in SSM: $DEPLOY_PARAM"
  aws ssm put-parameter --region "$REGION" --name "$DEPLOY_PARAM" --type SecureString --overwrite --value "$(cat "$DEPLOY_KEY_FILE")" >/dev/null
fi

AMI_ID="$(aws ssm get-parameter --region "$REGION" --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query 'Parameter.Value' --output text)"

USER_DATA_FILE="$(mktemp)"
cat > "$USER_DATA_FILE" <<EOF
#!/bin/bash
set -euxo pipefail

dnf update -y
dnf install -y git jq curl python3 python3-pip awscli gcc gcc-c++ make nodejs npm
if ! command -v codex >/dev/null 2>&1; then
  npm install -g @openai/codex@latest
fi

install -d -m 700 /home/ec2-user/.ssh
chown -R ec2-user:ec2-user /home/ec2-user/.ssh

auth_key="\$(aws ssm get-parameter --region '${REGION}' --name '${DEPLOY_PARAM}' --with-decryption --query 'Parameter.Value' --output text 2>/dev/null || true)"
if [[ -n "\$auth_key" && "\$auth_key" != "None" ]]; then
  echo "\$auth_key" > /home/ec2-user/.ssh/id_ed25519
  chmod 600 /home/ec2-user/.ssh/id_ed25519
  chown ec2-user:ec2-user /home/ec2-user/.ssh/id_ed25519
  ssh-keyscan github.com >> /home/ec2-user/.ssh/known_hosts
  chown ec2-user:ec2-user /home/ec2-user/.ssh/known_hosts
fi

install -d -m 755 /opt/prop-amm
chown ec2-user:ec2-user /opt/prop-amm

if [[ ! -d /opt/prop-amm/repo/.git ]]; then
  su - ec2-user -c "git clone '${REPO_SSH_URL}' /opt/prop-amm/repo"
else
  su - ec2-user -c "cd /opt/prop-amm/repo && git fetch --all && git pull --ff-only"
fi

su - ec2-user -c "if ! command -v cargo >/dev/null 2>&1; then curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal; fi"
su - ec2-user -c "if [[ -f \$HOME/.cargo/env ]]; then source \$HOME/.cargo/env; fi; if ! command -v cargo-build-sbf >/dev/null 2>&1; then cargo install cargo-build-sbf --locked; fi; cargo build-sbf --install-only --force-tools-install"
su - ec2-user -c "cd /opt/prop-amm/repo && python3 -m venv .venv"
su - ec2-user -c "cd /opt/prop-amm/repo && source .venv/bin/activate && pip install --upgrade pip"
su - ec2-user -c "cd /opt/prop-amm/repo && source .venv/bin/activate && pip install -r harness/requirements.txt"

install -d -m 755 /etc/prop-amm
set +x
openai_key="\$(aws ssm get-parameter --region '${REGION}' --name '${OPENAI_PARAM}' --with-decryption --query 'Parameter.Value' --output text 2>/dev/null || true)"
cat > /etc/prop-amm/harness.env <<ENV
OPENAI_API_KEY=\$openai_key
AGENT_MODEL=${AGENT_MODEL}
SYSADMIN_MODEL=${SYSADMIN_MODEL}
ENV
chmod 600 /etc/prop-amm/harness.env

if [[ -n "\$openai_key" && "\$openai_key" != "None" ]]; then
  login_status="\$(su - ec2-user -c 'bash -lc \"codex login status 2>&1 || true\"')"
  if ! echo "\$login_status" | grep -qi "Logged in using"; then
    su - ec2-user -c "env OPENAI_API_KEY='\$openai_key' bash -lc 'printf \"%s\" \"\$OPENAI_API_KEY\" | codex login --with-api-key >/dev/null 2>/dev/null || true'"
  fi
fi
set -x

cat > /etc/systemd/system/prop-amm-harness.service <<'SERVICE'
[Unit]
Description=Prop AMM Agent Harness
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/prop-amm/repo
EnvironmentFile=-/etc/prop-amm/harness.env
ExecStart=/bin/bash -lc 'source /opt/prop-amm/repo/.venv/bin/activate && python3 harness/core/loop.py --config ${CONFIG_PATH}'
Restart=on-failure
RestartSec=20
SuccessExitStatus=2 3
RestartPreventExitStatus=2 3
StandardOutput=append:/var/log/prop-amm-harness.log
StandardError=append:/var/log/prop-amm-harness.log

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
if [[ -n "\$openai_key" && "\$openai_key" != "None" ]]; then
  systemctl enable --now prop-amm-harness.service
else
  systemctl enable prop-amm-harness.service
  echo "OPENAI_API_KEY missing; service enabled but not started. Set ${OPENAI_PARAM} in SSM and run: systemctl start prop-amm-harness.service" >> /var/log/prop-amm-harness.log
fi
EOF

KEY_ARG=()
if [[ -n "$KEY_NAME" ]]; then
  KEY_ARG=(--key-name "$KEY_NAME")
fi

INSTANCE_ID="$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --subnet-id "$SUBNET_ID" \
  --iam-instance-profile Name="$INSTANCE_PROFILE_NAME" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  "${KEY_ARG[@]}" \
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${DISK_GB},VolumeType=gp3,DeleteOnTermination=true}" \
  --user-data "file://${USER_DATA_FILE}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Project,Value=${NAME_PREFIX}}]" \
  --query 'Instances[0].InstanceId' \
  --output text)"

rm -f "$USER_DATA_FILE"

echo "Launched instance: $INSTANCE_ID"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
PRIVATE_IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)"

cat <<EOF

Runner provisioning complete.
  Instance ID:   $INSTANCE_ID
  Region:        $REGION
  Private IP:    $PRIVATE_IP
  Public IP:     $PUBLIC_IP
  SecurityGroup: $SECURITY_GROUP_ID

Connect via SSM:
  aws ssm start-session --region $REGION --target $INSTANCE_ID

Service logs (from SSM shell):
  sudo tail -f /var/log/prop-amm-harness.log
  sudo systemctl status prop-amm-harness.service
EOF

# AWS + Private GitHub Setup (Agent Harness)

This setup uses:

- Private GitHub repo (your account)
- EC2 runner (persistent, SSH/SSM accessible)
- Harness loop service running continuously
- AWS Budget notification guardrail

## 1) Re-auth AWS CLI

If commands fail with session expired:

```bash
aws login --remote
```

Then verify:

```bash
aws sts get-caller-identity
```

## 2) Create private repo and push harness

```bash
./infra/github/create_private_repo.sh \
  --owner danrobinson \
  --repo prop-amm-agent-harness
```

Remote added: `private-origin`.

## 3) Create deploy key for EC2 clone access

```bash
./infra/github/create_deploy_key.sh \
  --repo danrobinson/prop-amm-agent-harness \
  --title prop-amm-ec2-deploy
```

Private key will be written to `.secrets/danrobinson_prop-amm-agent-harness_deploy_key`.

## 4) Provision EC2 runner

```bash
./infra/aws/provision_ec2_runner.sh \
  --region us-east-1 \
  --name-prefix prop-amm-harness \
  --instance-type c7i.4xlarge \
  --disk-gb 300 \
  --repo-ssh-url git@github.com:danrobinson/prop-amm-agent-harness.git \
  --deploy-key-file .secrets/danrobinson_prop-amm-agent-harness_deploy_key \
  --openai-api-key "$OPENAI_API_KEY" \
  --sysadmin-model gpt-5.3 \
  --config-path harness/configs/prop_amm.cloud.toml
```

Optional SSH enable (otherwise use SSM only):

```bash
--ssh-cidr YOUR.IP.ADDR.ESS/32 --key-name your-existing-ec2-keypair
```

## 5) Connect and check harness

SSM shell:

```bash
aws ssm start-session --region us-east-1 --target i-xxxxxxxxxxxxxxxxx
```

On instance:

```bash
sudo systemctl status prop-amm-harness.service
sudo tail -f /var/log/prop-amm-harness.log
```

## 6) Add budget guardrail

```bash
./infra/aws/create_budget_guardrail.sh \
  --region us-east-1 \
  --amount 1000 \
  --email you@example.com
```

## 7) Configure GitHub Action-triggered updates (recommended: OIDC)

Create AWS OIDC deploy role and wire it to the private repo:

```bash
./infra/aws/setup_github_actions_oidc.sh \
  --region us-east-1 \
  --repo danrobinson/prop-amm-agent-harness \
  --instance-id i-0931eb5e7adf5600e \
  --set-repo-variable
```

Set repo variables used by workflow:

```bash
gh variable set AWS_REGION --repo danrobinson/prop-amm-agent-harness --body us-east-1
gh variable set EC2_INSTANCE_ID --repo danrobinson/prop-amm-agent-harness --body i-0931eb5e7adf5600e
```

Then use workflow:

- `.github/workflows/deploy-ec2-update.yml`
- triggers on push to `main` / `codex/aws-ec2-harness`
- also supports manual dispatch with custom branch/instance

Fallback (if you do not use OIDC): set repo secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`.

## Notes

- Harness state persists in repo workspace under `.harness/`.
- Update code by pushing to GitHub and pulling on the instance.
- Resource names are prefixed by `--name-prefix` for easy cleanup.

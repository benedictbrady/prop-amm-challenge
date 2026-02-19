# Infrastructure Automation

Automation scripts for provisioning a dedicated AWS EC2 harness runner and wiring a private GitHub repo.

## GitHub scripts

- `infra/github/create_private_repo.sh`
- `infra/github/create_deploy_key.sh`

## AWS scripts

- `infra/aws/provision_ec2_runner.sh`
- `infra/aws/create_budget_guardrail.sh`
- `infra/aws/setup_github_actions_oidc.sh`
- `infra/aws/update_runner_from_github.sh`
- `infra/aws/teardown_runner.sh`

## End-to-end guide

See `infra/docs/AWS_GITHUB_SETUP.md`.

# Cloud Execution

This harness can run unattended in cloud CI or containers.

## Option 1: GitHub Actions (included)

Workflow file: `.github/workflows/harness-cloud.yml`

Setup:

1. Add repo secret `OPENAI_API_KEY`.
2. Trigger workflow `harness-cloud` from GitHub Actions UI.
3. Optionally override:
   - `config_path`
   - `agent_model`
   - `dry_run`

Outputs:

- Uploads `.harness/` as an artifact at the end of each run.
- Uploads current strategy file snapshot (`programs/starter/src/lib.rs`).
- Restores/saves `.harness/` with GitHub cache for cross-run resume.

## Option 2: Container job (Docker)

Build image:

```bash
docker build -f harness/cloud/Dockerfile -t prop-amm-harness .
```

Run image:

```bash
docker run --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e AGENT_MODEL="gpt-5" \
  -e HARNESS_CONFIG="harness/configs/prop_amm.cloud.toml" \
  -e HARNESS_WORKDIR="/workspace" \
  -v "$PWD":/workspace \
  prop-amm-harness
```

## Using a different agent runtime

If you prefer Codex CLI, Claude Code, OpenHands, etc., only change `agent.command_template` in config. The loop and task adapter stay unchanged.

## Option 3: Persistent AWS EC2 Runner

For heavy-duty long-running jobs with shell access, use the scripts under `infra/`:

- `infra/github/create_private_repo.sh`
- `infra/github/create_deploy_key.sh`
- `infra/aws/provision_ec2_runner.sh`
- `infra/aws/create_budget_guardrail.sh`

Full guide: `infra/docs/AWS_GITHUB_SETUP.md`.

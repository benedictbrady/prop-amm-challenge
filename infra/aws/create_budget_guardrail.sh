#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create or update an AWS monthly budget guardrail for harness spend.

Usage:
  create_budget_guardrail.sh --email <address> [options]

Options:
  --region <region>          AWS region (default: AWS_REGION or config, fallback us-east-1)
  --name <budget-name>       Budget name (default: prop-amm-harness-monthly)
  --amount <usd>             Monthly limit in USD (default: 1000)
  --email <address>          Notification email (required)
  -h, --help                 Show help
EOF
}

REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
if [[ -z "$REGION" ]]; then
  REGION="us-east-1"
fi

BUDGET_NAME="prop-amm-harness-monthly"
AMOUNT="1000"
EMAIL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"; shift 2 ;;
    --name)
      BUDGET_NAME="$2"; shift 2 ;;
    --amount)
      AMOUNT="$2"; shift 2 ;;
    --email)
      EMAIL="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$EMAIL" ]]; then
  echo "--email is required" >&2
  usage
  exit 2
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || true)"
if [[ -z "$ACCOUNT_ID" || "$ACCOUNT_ID" == "None" ]]; then
  echo "AWS auth unavailable or expired. Run 'aws login --remote' and retry." >&2
  exit 2
fi

BUDGET_DOC="$(mktemp)"
cat > "$BUDGET_DOC" <<JSON
{
  "BudgetName": "${BUDGET_NAME}",
  "BudgetType": "COST",
  "TimeUnit": "MONTHLY",
  "BudgetLimit": {
    "Amount": "${AMOUNT}",
    "Unit": "USD"
  },
  "CostFilters": {},
  "CostTypes": {
    "IncludeTax": true,
    "IncludeSubscription": true,
    "UseAmortized": false,
    "UseBlended": false,
    "IncludeRefund": false,
    "IncludeCredit": false,
    "IncludeUpfront": true,
    "IncludeRecurring": true,
    "IncludeOtherSubscription": true,
    "IncludeSupport": true,
    "IncludeDiscount": true
  },
  "TimePeriod": {
    "Start": "2026-01-01T00:00:00Z"
  }
}
JSON

NOTIFS_DOC="$(mktemp)"
cat > "$NOTIFS_DOC" <<JSON
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "${EMAIL}"
      }
    ]
  },
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 100,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "${EMAIL}"
      }
    ]
  }
]
JSON

if aws budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name "$BUDGET_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "Updating existing budget: $BUDGET_NAME"
  aws budgets update-budget --account-id "$ACCOUNT_ID" --new-budget "file://$BUDGET_DOC" --region "$REGION" >/dev/null
  echo "Budget updated. Existing notifications retained."
else
  echo "Creating budget: $BUDGET_NAME"
  aws budgets create-budget \
    --account-id "$ACCOUNT_ID" \
    --budget "file://$BUDGET_DOC" \
    --notifications-with-subscribers "file://$NOTIFS_DOC" \
    --region "$REGION" >/dev/null
  echo "Budget created with 80% and 100% email alerts to $EMAIL"
fi

rm -f "$BUDGET_DOC" "$NOTIFS_DOC"

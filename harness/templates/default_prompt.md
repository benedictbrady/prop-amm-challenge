Iteration: {iteration}
Mode: {mode}
Mode directive: {mode_instructions}

Target:
- {target_label}

File to edit:
- {strategy_file}

Task context:
{task_context}

Budget:
- Remaining USD: {budget_remaining:.2f}
- Spent USD: {budget_spent:.2f}

Recent history:
{recent_history}

Recent failures:
{recent_failures}

Elite candidates:
{elite_summary}

Rules:
- Keep changes robust across training folds.
- Do not run expensive final holdout manually.
- Keep edits minimal, clear, and testable.
- Preserve strict monotonicity and concavity for both sides; avoid non-concave fee schedules.
- If recent iterations failed validation, prefer conservative fixes over major rewrites.

End your response with:
- changes:
- rationale:
- risk:
- optional COST_USD=<number>

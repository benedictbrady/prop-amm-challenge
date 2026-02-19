<!-- Legacy template for harness/agent_harness.py. -->
You are improving `/Users/dan/.codex/worktrees/7efe/prop-amm-challenge`.

Iteration: {iteration}
Mode: {mode}
Mode directive: {mode_instructions}

Primary objective:
- Achieve holdout avg edge `> {target_holdout:.2f}` on a 1000-sim out-of-sample run.

File to edit:
- `{strategy_file}`

Do not optimize to one seed block. The harness already runs multiple train folds and a separate holdout fold.

Hard constraints:
- Keep outputs monotonic and concave.
- Keep code safe Rust only.
- Preserve required metadata functions.
- Do not run the expensive 1000-sim holdout yourself; use faster local checks while iterating.

Budget status:
- Remaining: `${budget_remaining:.2f}`
- Spent: `${budget_spent:.2f}`

Train folds:
{train_fold_summary}

Holdout fold:
{holdout_fold_summary}

Recent history:
{recent_history}

Elite candidates:
{elite_summary}

Deliverables this iteration:
1. Update strategy code.
2. Run lightweight sanity checks that you think are enough for this step.
3. End with a short note containing:
   - `changes:`
   - `rationale:`
   - `risk:`
4. If available, append `COST_USD=<number>` for budget accounting.

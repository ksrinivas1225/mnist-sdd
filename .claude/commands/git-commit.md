---
description: Stage changes and commit with a message that describes what changed in the model, training loop, config, or tests. Never pushes.
allowed-tools: Bash(git status), Bash(git diff), Bash(git add), Bash(git commit)
---

You are committing changes to the MNIST digit classifier repo.
Use only git Bash commands — no file reads, edits, or other tools.
Never run `git push` or any destructive git command.

## Dynamic context

### git status
```
$(git status)
```

### git diff HEAD
```
$(git diff HEAD)
```

## Instructions

1. Read the diff. Identify which files changed and what the change does.
   Focus on these areas when writing the message:
   - `model.py` — architecture, layers, activation, forward pass
   - `train.py` — training loop, optimiser, scheduler, data loading
   - `evaluate.py` — evaluation logic, metrics
   - `config.yaml` — hyperparameter or pipeline changes
   - `tests/` — new or updated tests

2. Write a commit message:
   - Subject line: imperative mood, ≤ 72 characters, no trailing period
   - If the reason for the change is clear from the diff, add a blank line
     and a short body (1–3 sentences) explaining why.

3. Stage all modified tracked files:
   ```
   git add -u
   ```

4. Commit with the message and this co-author trailer:
   ```
   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```

5. Report the commit hash and subject line. Stop there — do NOT push.

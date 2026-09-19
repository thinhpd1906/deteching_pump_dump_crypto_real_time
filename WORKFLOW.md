# AGENT WORKFLOW GUIDE

## Which agent to invoke when

Use `@<agent-name>` in Claude Code to delegate. Each agent runs in its own context window, keeping your main conversation clean.

## By pilot week

| Week | What you're doing | Primary agent(s) | Slash command |
|------|-------------------|------------------|---------------|
| 1 | Setup + labels + Gate 1 | `@data-agent` | `/gate 1` |
| 2 | Datasets across 3 modes | `@cleaning-agent`, `@feature-agent` | `/gate C` |
| 3 | Baselines + Gate 2 | `@baseline-agent` | `/gate 2` |
| 4 | Microstructure baselines | `@baseline-agent` | — |
| 5 | AT + CNN-BiLSTM implementation | `@model-agent` | — |
| 6 | Pretrain + unsupervised eval | `@training-agent` | — |
| 7 | Fine-tuning + label efficiency + Gate 3 | `@training-agent` | `/gate 3` |
| 8 | Tables + figures + draft | `@evaluation-agent`, `@paper-agent`, `@reviewer-agent` | — |

## By type of question

### "How do I download / verify data?"
→ `@data-agent`

### "Is my cleaning actually removing anything?"
→ `@cleaning-agent`

### "Add a new feature / change window size / rebuild dataset"
→ `@feature-agent`

### "Does the signal exist in my data?"
→ `@baseline-agent` (runs Random Forest → Gate 2)

### "Implement a new model architecture"
→ `@model-agent`

### "Train X model with Y seeds and Z config"
→ `@training-agent`

### "Give me the paper tables and figures"
→ `@evaluation-agent`

### "Set up Kafka / measure latency / build the pipeline"
→ `@pipeline-agent`

### "Draft the abstract / intro / methods section"
→ `@paper-agent`

### "Read my draft and tell me what a reviewer would attack"
→ `@reviewer-agent`

## Chained workflows

### Full pilot (recommended for Weeks 1-8)
```
/pilot
```
Runs everything through appropriate agents in dependency order.

### Just verify the whole state
```
/status
```
Shows what's done, what's next.

### Add a new experimental variant
```
@model-agent implement variant X
@training-agent train it with 3 seeds  
@evaluation-agent add row to Table 1
```

## Rules of engagement

- **Do NOT try to do @data-agent's job in @model-agent's context.**
  Each agent has a narrow scope. Cross-agent work causes context confusion.

- **When in doubt, invoke by name explicitly:** `@baseline-agent run B4 on adaptive mode`

- **Slash commands are for quick checks, not deep work.** `/gate 2` is fine; complex training is not.

- **Agents can call other agents.** If `@paper-agent` needs a table it doesn't see, it will invoke `@evaluation-agent`.

## Debugging when an agent gets confused

- Check `CLAUDE.md` is still correct (locked design decisions)
- The agent may have exceeded its context window on a huge file — narrow the task
- If it wants to change locked design decisions, say NO and refer to CLAUDE.md

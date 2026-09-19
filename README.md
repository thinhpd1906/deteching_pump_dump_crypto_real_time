# Claude Code Multi-Agent Setup for Thesis

## What this is

A complete `.claude/` folder with **10 specialized subagents**, **4 skills**, and **3 slash commands** designed to implement the pump-and-dump detection thesis from Week 1 to submission — with Claude Code as the orchestrator.

## What's inside

```
.claude/
├── settings.json           # Permissions (safe defaults for Windows)
├── agents/                 # 10 subagents, one per pipeline stage
│   ├── data-agent.md
│   ├── cleaning-agent.md
│   ├── feature-agent.md
│   ├── baseline-agent.md
│   ├── model-agent.md
│   ├── training-agent.md
│   ├── evaluation-agent.md
│   ├── pipeline-agent.md
│   ├── paper-agent.md
│   └── reviewer-agent.md
├── skills/                 # Reusable procedures any agent can invoke
│   ├── run-gate1/SKILL.md
│   ├── verify-scaler-parity/SKILL.md
│   ├── bootstrap-ci/SKILL.md
│   └── experiment-runner/SKILL.md
└── commands/               # Slash commands for quick actions
    ├── status.md           # /status  → project state board
    ├── pilot.md            # /pilot   → run full 8-week battery
    └── gate.md             # /gate N  → check gate N verdict

CLAUDE.md                   # Master briefing loaded at every session start
WORKFLOW.md                 # Which agent to invoke when
```

## Installation on Windows

### Step 1 — Copy to your thesis repo

```powershell
# In PowerShell, from your thesis project root:
cd C:\path\to\pnd-thesis

# Unzip pnd-thesis-agents.zip into the project root, or manually:
# Copy .claude/ folder here
# Copy CLAUDE.md here  
# Copy WORKFLOW.md here
```

### Step 2 — Verify Claude Code sees it

```powershell
claude
# Inside Claude Code, type:
/status
```

If Claude Code loaded correctly, `/status` will run the project status check.

### Step 3 — Verify subagents are visible

Inside Claude Code:

```
/agents
```

You should see all 10 agents listed. If not, check the `.claude/agents/` folder is at the project root (not nested elsewhere).

## How to use

### First-time invocation

```
Use @data-agent to run Gate 1 with 10 events
```

Claude Code will spawn a subagent conversation with only the `data-agent`'s system prompt and tools. That agent runs the task in its own context window, then reports back.

### Chained workflows

```
Use @data-agent to check Gate 1.
If Gate 1 passes, use @cleaning-agent to run Gate C.
If Gate C passes, use @baseline-agent to run all baselines.
```

Claude Code handles the delegation.

### Slash commands for common actions

```
/status          # Where am I in the pilot?
/gate 1          # Check Gate 1 verdict
/gate 2          # Check Gate 2 (RF F1)
/pilot           # Run the entire pilot battery
```

## Important notes for Windows users

- Use forward slashes `/` in file paths inside code (Python handles it)
- Docker Desktop needed for Kafka (`docker compose up -d`)
- Virtual env: `python -m venv .venv && .venv\Scripts\activate`
- If PowerShell blocks scripts: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## The Claude Pro consideration

You mentioned you have Claude Pro. Two things to know:

1. **Claude Code with Pro** works but has usage limits (~50 messages per 5-hour window on Sonnet). For a full thesis project, upgrading to **Max ($100/mo)** removes practical limits and unlocks Opus for complex tasks. Consider this before Week 5 when training loops start.

2. **Subagents don't multiply your quota.** Each subagent invocation consumes tokens from your same account. Delegating to `@baseline-agent` costs the same tokens as doing it in the main conversation — the benefit is context isolation, not token savings.

## Model selection

For each agent, Claude Code picks the model. General guidance:

- **Simple agents** (`@reviewer-agent`, `@evaluation-agent`) → Sonnet is enough
- **Complex code agents** (`@model-agent`, `@training-agent`) → Sonnet 4.5 or Opus
- **Long-context agents** (`@paper-agent` reading the full codebase) → whichever has largest context

You don't need to configure this manually — Claude Code chooses.

## Common workflows

### Week 1 (first day):
```
@data-agent
Run Gate 1 with 10 events.
```
Watch the plots; if it passes, proceed. If not, ask the agent to try timezone offsets.

### Week 3 (after data is built):
```
@baseline-agent
Run all baselines on the adaptive dataset and report Gate 2.
```

### Week 7 (headline experiment):
```
@training-agent
Run label-efficiency study with 3 seeds and report Gate 3.
```

### Week 8 (paper draft):
```
@evaluation-agent build Table 1 and Table 2.
@paper-agent write the Introduction and Evaluation Protocol sections.
@reviewer-agent read the current draft and give me the hostile review.
```

## Debugging

**"Agent doesn't know something I told it"** → Different agents have different contexts. Add the fact to `CLAUDE.md` (loaded by all sessions) if it's important globally.

**"Agent wants to change the design"** → Say no and refer it to `CLAUDE.md` locked decisions. Agents should never reopen locked choices without your explicit approval.

**"Slash command not working"** → Check `.claude/commands/*.md` files exist. Try `/help` inside Claude Code.

**"Bash commands being denied"** → Edit `.claude/settings.json` `allow` list. Add specific commands like `Bash(pytest:*)`.

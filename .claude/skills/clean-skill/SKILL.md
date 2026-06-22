---
name: clean-skill
description: Audit a Claude Code skill's per-invocation context use, then propose a context-bundling helper + SKILL.md edits to cut waste. Reads a representative session JSONL, ranks the largest tool results, traces them back to SKILL.md instructions, and proposes a fix. Stops after the proposal — the operator approves before code is written. Use when a skill feels slow, expensive, or "burns context" — typical signal is a SKILL.md over ~300 lines that reads multiple files >5 KB on every run.
---

# /clean-skill — Skill context-bundling audit & cleanup

Audits one skill's context efficiency end-to-end:

1. Pick a representative recent session that invoked the skill.
2. Quantify what burned context (counts, sizes, top tool results, anti-patterns).
3. Trace each big result back to its SKILL.md instruction.
4. Propose a helper + SKILL.md rewrite scoped to the load-bearing reads.
5. **Stop and wait for operator approval** before writing code.

The recipe behind this skill: if a SKILL.md asks the agent to read 7+ markdown indexes on every invocation, ~70% of that read budget can usually be replaced by a single Python helper that emits a projected JSON blob. The skill below packages the audit procedure so other skills can get the same treatment without re-deriving the playbook.

## Inputs

```
/clean-skill <skill-name> [<session-id>]
```

- `<skill-name>` — name of the skill in `.claude/skills/` (project-local) or `~/.claude/skills/` (user-global). Required.
- `<session-id>` — optional 36-char UUID of a session that invoked the skill. If omitted, the skill picks the most recent session under the Claude Code projects dir for the current working tree that mentions `/<skill-name>` in its first user message. If no such session is found, ask the operator which one to use.

The Claude Code projects directory is `~/.claude/projects/<encoded-cwd>/` on POSIX; on Windows it is `%USERPROFILE%\.claude\projects\<encoded-cwd>\`. The encoded form replaces `/` and `:` with `-` (e.g. a working tree at `/work/myproj` → `-work-myproj`).

## When to run

- A skill feels slow or expensive on a recent invocation.
- A SKILL.md has crossed ~300 lines and accreted historical "(added YYYY-MM-DD)" paragraphs.
- The same skill reads 3+ files larger than 5 KB on every invocation.
- Periodic hygiene — every quarter, audit one heavily-used skill.

**Skip if:** SKILL.md is already under ~250 lines AND reads ≤2 files per invocation. Not every skill is a target; abort with "no surgery warranted" and tell the operator why.

## Authorization

This skill is **read-only by default**. It writes nothing until the operator approves the proposal. After approval, it may:

- Write `tools/<skill_name>_context.py` (new file) in the project the skill belongs to.
- Edit `.claude/skills/<skill-name>/SKILL.md`.

It does **not** authorize git commits, branch changes, pushes, or any destructive operations. Operator handles git.

## Instructions

### 1. Find the target session

Determine the projects dir for the current working tree (the encoded-cwd form above), then list the newest `.jsonl` files. Pick the newest whose first user message mentions `/<skill-name>` or matches the skill's known invocation pattern. If multiple candidates look plausible, list them and ask the operator. **Do not guess silently** — auditing the wrong session produces worthless numbers.

### 2. Quantify what burned context

Run **one** Python pass over the JSONL and emit the headline numbers. Use this script verbatim — it handles the Windows cp1252 trap that breaks naive prints:

```bash
python - <<'EOF'
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
SESSION = r'<absolute-path-to-session.jsonl>'
counts = {'user':0,'assistant':0,'tool_use':0,'tool_result':0,'thinking':0}
sizes  = dict(counts); sizes['total']=0
tu_count, tu_size, results = {}, {}, []
with open(SESSION, encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line); sizes['total'] += len(line)
        msg = rec.get('message', {}); content = msg.get('content', '')
        blocks = [{'type':'text','text':content}] if isinstance(content,str) else (content or [])
        for blk in blocks:
            if not isinstance(blk, dict): continue
            t = blk.get('type','text'); sz = len(json.dumps(blk))
            if t == 'tool_use':
                n = blk.get('name','?')
                tu_count[n] = tu_count.get(n,0)+1
                tu_size[n]  = tu_size.get(n,0)+sz
                counts['tool_use'] += 1; sizes['tool_use'] += sz
            elif t == 'tool_result':
                txt = blk.get('content','')
                if isinstance(txt, list):
                    txt = ''.join(x.get('text','') if isinstance(x,dict) else str(x) for x in txt)
                results.append((blk.get('tool_use_id','?'), len(txt or ''), (txt or '')[:200]))
                counts['tool_result'] += 1; sizes['tool_result'] += sz
            elif t == 'thinking':
                counts['thinking'] += 1; sizes['thinking'] += sz
print('TOTAL FILE BYTES:', sizes['total'])
print('Counts:', counts)
print('Sizes :', sizes)
print('\nTool uses (by total bytes):')
for n, sz in sorted(tu_size.items(), key=lambda x: -x[1]):
    print(f'  {n}: count={tu_count[n]}, bytes={sz}')
print('\nLargest tool results:')
results.sort(key=lambda x: -x[1])
for tu, sz, snip in results[:15]:
    print(f'  [{sz:>6}] {tu[-8:]}  {snip[:140]}')
EOF
```

Report numbers in your audit response. Specifically: **total bytes, top 5 tool_results' bytes + source files, % of each result the skill actually used.**

### 3. Trace each big result back to its SKILL.md instruction

For each tool result over ~10 KB, walk back through the JSONL to find:

- The Read offset/limit or Bash command that produced it.
- The corresponding paragraph or step in `.claude/skills/<skill-name>/SKILL.md` that told the agent to make that call.
- Whether the same file is read more than once.
- Whether the file has a hand-curated "current section" above an accreting historical tail.

**Anti-patterns to flag:**

1. "Front-load all X in one parallel batch" — parallel ≠ free.
2. `Read offset=0 limit=all` on indexes with >50 KB of preserved historical paragraphs.
3. Re-reading the same file with two different `limit` slices (didn't budget upfront).
4. Reading whole 30-50 KB plan/spec docs to extract a 500-byte header block.
5. Multi-numbered sub-steps (X.1, X.2, X.3, X.4) that don't gate distinct phases — collapse into one block.
6. Long "added YYYY-MM-DD" / "as of YYYY-MM-DD" historical justification paragraphs.

### 4. Write the AUDIT REPORT to chat (~250 words)

Format:

```
## /<skill-name> context audit — <YYYY-MM-DD>

Session: <id> (<total-KB> total)

### Top wasted reads
| File | Bytes | Skill actually uses | Waste |
|---|---|---|---|
| <path> | <N KB> | <header / current section / N rows> | <N KB> |
...

### Anti-patterns observed
- <one bullet per pattern that fires, with file/step cite>

### Estimated savings
~N% / ~M KB if the helper-bundling pattern is applied.

### Proposed surgery
- Helper: `tools/<skill-name>_context.py` returning JSON with keys: <list>
- SKILL.md: collapse Steps X-Y into "run helper, parse JSON". Rewrite section Z. Trim ~N lines of historical paragraphs.

**Awaiting operator approval before writing code.** Reply with `proceed`, `modify <what>`, or `skip`.
```

### 5. STOP

After the audit report, stop and wait for operator approval. **Do not write the helper or edit SKILL.md preemptively.**

If the operator says `proceed`, follow Section 6. If they say `modify <what>`, regenerate the proposal with their changes and stop again. If `skip`, save the audit report under `docs/_audits/` (or `output/audits/` if the project has no docs tree) and exit.

### 6. Implementation (only after approval)

#### 6.1. Write `tools/<skill-name>_context.py`

Hard requirements:

- Read-only. Never mutates files.
- Stdout = JSON. UTF-8 forced via `sys.stdout.reconfigure(encoding='utf-8')`.
- Tolerates missing files (return `[]` / `""`).
- Each top-level JSON key replaces ONE bulk Read or subprocess call from the SKILL.md.
- Project / filter aggressively: drop fields the skill doesn't consume; cap historical tails; filter LOW-priority rows.
- Module docstring documents the schema and cites this audit's session ID.
- If the skill already has a state-management script, shell out via subprocess — don't duplicate parsing.

Smoke-test target: **≤80 KB total output**. If you exceed that, your projections are too generous — trim more.

#### 6.2. Edit `.claude/skills/<skill-name>/SKILL.md`

- The bulk-read step becomes ONE Bash call to the helper + a Read of the JSON.
- Add a JSON-key → consumer-step table so future readers don't chase keys around.
- Keep the escape hatch: "if you genuinely need a section the helper omits, Read that *one* file with a small `limit`."
- Trim historical "(added YYYY-MM-DD)" change-log paragraphs.
- Collapse multi-numbered sub-steps where they don't gate distinct phases.
- Update cross-references when you renumber steps.

Target: SKILL.md line count drops ≥15%. If it doesn't, you're not trimming enough.

#### 6.3. Save the audit report

Write the audit + the implementation diff summary under the project's audit dir (`docs/_audits/<skill>-context-audit-<YYYY-MM-DD>.md` if `docs/_audits/` exists, else `output/audits/`). This forms the trail the next `/clean-skill` invocation reads as precedent.

#### 6.4. Report back

```
## /<skill> cleanup complete

- Helper: tools/<skill>_context.py (<N> bytes smoke-test output)
- SKILL.md: <before> → <after> lines (-<N>)
- Audit doc: <path>
- Estimated next-run context savings: ~<N>%

Operator handles git commit (no auto-commit from this skill).
```

## Anti-patterns

- **Don't audit a skill whose representative session you can't identify.** A wrong session = worthless numbers. Ask the operator if unsure.
- **Don't write code before operator approves.** Step 5 is a hard stop.
- **Don't audit `/clean-skill` itself with `/clean-skill`.** Self-referential audits compound rather than reduce noise.
- **Don't skip the smoke test.** A helper that emits 200 KB JSON is worse than no helper.
- **Don't fold a one-shot research call into the helper.** Helpers are for repeat invocations; one-shot reads stay in the skill.
- **Don't auto-commit.** Bookkeeping commits need operator authorization unless the SKILL.md being edited explicitly grants it.

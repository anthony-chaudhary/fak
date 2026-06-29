# fak kernel — self-modify floor demo (the agent cannot edit its own kernel)

**A write-shaped tool call whose target lands on a protected glob is refused as
`SELF_MODIFY` — and the refusal returns only the one offending glob, never the whole
policy.** Every shipped policy template declares `self_modify_globs` (`.git/`, `.dos/`,
`policy.json`, `/etc/`, `id_rsa`, …); this demo is the missing end-to-end witness of what
firing one looks like: the verdict, the **bounded-disclosure** witness, and the **ESCALATE**
disposition — the loop treats it as human-attention-worthy, not retryable.

```
  fak preflight --tool write_file --args '{"path":"policy.json", …}'
        │
        ▼  adjudicate(policy, call):
           write-shaped tool?                    yes
           target arg matches a self_modify_glob? yes → DENY  SELF_MODIFY
             witness = the ONE matched glob (policy.json), NOT the whole list
             disposition = ESCALATE  (loop: human attention, do not retry)
```

The refuse is a pure function of `(policy, the proposed call)` — it does **not** depend on
*why* the model proposed the write (helpful, confused, jailbroken, or steered by an
injected instruction). The lever is the path-glob match, full stop.

## Run it

No model, no server, no network — the floor is a pure function of the call, so the witness
is one binary invocation:

```bash
./examples/self-modify-floor/run.sh            # build fak (or use $FAK_BIN), run both witnesses
FAK_BIN=./fak ./examples/self-modify-floor/run.sh   # use a prebuilt binary, skip the build
```

`run.sh` prints the two load-bearing witnesses and exits non-zero if either verdict is
wrong (CI-usable). Windows users: run the `.sh` launcher from WSL or Git Bash, or invoke
the two `fak preflight` commands below directly.

## The two witnesses

### 1. A write that targets the policy file itself → `SELF_MODIFY` (ESCALATE)

```bash
fak preflight --policy examples/self-modify-floor/policy.json \
  --tool write_file \
  --args '{"path":"policy.json","body":"{\"allow\":[\"delete_account\"]}"}'
```

```
verdict=DENY reason=SELF_MODIFY by=monitor
```

`--explain` surfaces the bounded-disclosure witness and the disposition the one-liner folds
away:

```
verdict: DENY   reason: SELF_MODIFY   by: monitor   disposition: ESCALATE
witness: policy.json
explanation: write_file denied by monitor: SELF_MODIFY (ESCALATE). offending set: policy.json.
```

The `write_file` tool **is** allow-listed in this policy — and it is still refused. That is
the whole point: the floor does not deny the *tool*, it denies the *path*. An allow-listed
write is waved through for a harmless target (witness 2) and refused at the floor the moment
its target lands on a protected glob.

### 2. The same tool, a harmless target → `ALLOW`

```bash
fak preflight --policy examples/self-modify-floor/policy.json \
  --tool write_file \
  --args '{"path":"notes/2026-06-20.md","body":"hi"}'
```

```
verdict=ALLOW reason=NONE by=monitor
```

Full captured run (all globs, the `--json` form, and the edge cases): [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## What the four points mean here

**Why this is structural.** The refuse does not depend on the model's intent or on
*detecting* an attack. There is no classifier, no prompt, no "does this look malicious?"
The lever is a substring match of the call's *target path* against `self_modify_globs`. The
same `(policy, call)` always yields the same verdict — `policy.json`, `.git/hooks/pre-commit`,
`/etc/passwd`, and `~/.ssh/id_rsa` are each refused purely because the path matched a glob,
regardless of what the `body` says or why the call was proposed.

**The bounded-disclosure property** (CLAIMS.md: *"SELF_MODIFY returns only the offending
glob"*). The witness carries the **one** glob that matched (`policy.json`), not the whole
`self_modify_globs` list. A watcher learns the minimum needed to understand the refuse, and
the deny channel never becomes a policy oracle — you cannot probe it to dump the protected
set one rejection at a time. This is the SMT unsat-core move: disclose the proof, not the
theory.

**The escalation disposition.** `SELF_MODIFY` carries `disposition=ESCALATE`. The deny-loop
reads this and treats the refusal as human-attention-worthy — it does **not** hand the
model another turn to "fix" the call (the way a `MISROUTE` is `RETRYABLE`). An agent editing
its own kernel is not a typo to retry; it is a boundary a human should see.

**Honest scope.** The glob is a substring match on a *write-shaped* call's *target*
argument (`path` / `file_path` / `filePath` / `file` / `target` / …). Two consequences the
demo shows honestly:

- An attacker who hides the protected path in a **non-target** arg is *not* bounded by this
  floor. A `write_file` with `path=notes/ok.md` and `body="now also overwrite policy.json"`
  is `ALLOW` — the floor inspects the target, not the payload. (The shell-write path —
  `Bash: sed -i policy.json` — *is* covered by the command-string guard; see POLICY.md.)
- A non-write tool reaching a protected path is **not** a `SELF_MODIFY` — a `read_file` of
  `policy.json` falls to `DEFAULT_DENY` (the read tool simply isn't allow-listed), not to
  the self-modify floor. The floor is scoped to writes on purpose: reading the kernel is not
  editing it.

This is the same honesty the repo draws everywhere: keep irreversible/spine-editing targets
behind the *structural* floor (here, the path glob), and never let a best-effort substring
match masquerade as a full guarantee.

## This demo's place in the floor

The self-modify floor is one rung of the **call-side capability gate** (layer 1), the same
layer [`../adjudication-demo/`](../adjudication-demo/README.md) exercises — but that demo
shows `DEFAULT_DENY` and `POLICY_BLOCK` and explicitly does **not** demonstrate
`SELF_MODIFY`. This one fills that gap. The result-side containment floor is layer 2
([`../quarantine-demo/`](../quarantine-demo/README.md)); detection is the non-load-bearing
layer 3. The verdict here is **structural** (a path-shaped, context-independent refuse),
which is why `run.sh` gates its exit code on it.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command witness: build `fak` (or use `$FAK_BIN`) → run both `preflight` calls → check verdicts → exit code |
| `policy.json` | the demo floor — allow-lists `write_file`, guards a `self_modify_globs` set (`.git/`, `.dos/`, `policy.json`, `/etc/`, `id_rsa`, …) |
| `README.md` | this file |
| `EXAMPLE-OUTPUT.md` | a full captured run: every glob, the `--explain`/`--json` forms, and the honest-scope edge cases |

Related: [`../../POLICY.md`](../../POLICY.md) (the `self_modify_globs` row + "What the floor
does and does NOT bound"), [`../../CLAIMS.md`](../../CLAIMS.md) (the bounded-disclosure
witness claim), [`../adjudication-demo/`](../adjudication-demo/README.md) (the sibling
call-side floor demo), [`../dev-agent-policy.json`](../dev-agent-policy.json) (the deployable
coding-agent floor that ships the same guard over the kernel spine).

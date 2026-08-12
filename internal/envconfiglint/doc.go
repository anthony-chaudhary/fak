// Package envconfiglint is the CONFIG_NOT_ENV ratchet: the durable gate that keeps
// behavioral configuration out of the environment (issue #2863, Track G / epic #2834).
//
// # The rule it enforces structurally
//
// Hermes' contribution rubric rejects "new HERMES_* env vars for non-secret config" —
// .env is for SECRETS only (keys/tokens/passwords); behavioral settings belong in the
// config surface. Hermes enforces that with a human reviewer reading the diff, so
// violations still land and get walked back later. fak makes the same rule deterministic
// and machine-checked: an environment READ (os.Getenv / os.LookupEnv) must name a declared
// secret, or it is a config-surface violation carrying a codemod-style fix. Same
// deny-by-structure instinct as the schema mask and the pythongate ratchet, applied to the
// CONFIGURATION surface.
//
// # The ratchet
//
// The tree already carries hundreds of behavioral env reads. Relocating them all at once is
// neither safe nor necessary; what matters is that the count only ever goes DOWN. So this
// gate does not ban env reads — it bans NEW non-secret ones. It compares the env-var names
// read in COMMITTED Go source against a frozen allowlist (baseline.go, captured the day the
// ratchet shipped) and refuses any name that is neither grandfathered nor secret-shaped.
//
// The gate governs SHIPPED source only — _test.go is excluded. A test-harness switch
// (FAK_NEGBENCH_MODEL, CODEX_AUDIT_SAMPLE) is neither a credential nor a setting that could
// relocate to a config file; it selects a fixture for a test that only runs under `go test`.
// Scanning tests demanded relocation for reads with nowhere to go and baked this lint's own
// synthetic fixtures (FOO_MODE, BAR_TOKEN) into the baseline. Partitioning by file class is
// the internal/boundarylint precedent. (baseline.go still carries the test-derived names
// from its original freeze; they are inert and drop out on the next regeneration.)
//
// Two judgments, one classifier:
//
//   - A name whose underscore-delimited tokens include a credential word (KEY, TOKEN,
//     SECRET, PASSWORD, CREDENTIAL, AUTH, …) is a DECLARED SECRET — allowed in the
//     environment. See IsSecretName.
//   - Any other new name is behavioral configuration — refused with ReasonConfigNotEnv and
//     an Offense.Fix() suggestion pointing at the config surface.
//
// # Two modes
//
//   - ScanTree is the CI gate: it reads committed content at HEAD via `git grep` (never the
//     working tree) so only source that would actually ship counts and a peer's uncommitted
//     WIP can never red your run. TestNoNewNonSecretEnvReads is the trunk guard.
//   - ClassifyDiff is the issue's literal spine — judge only the env reads on ADDED lines of
//     a unified diff. It is the shape a pre-commit / CI shell wraps around `git diff`, and
//     needs no baseline: added-line filtering already restricts the verdict to new reads.
//
// The fix is advisory by contract. This package never rewrites source: an automatic rewrite
// of a behavioral read could silently change runtime behavior, which is why #2863 asks for a
// codemod SUGGESTION, not a codemod.
//
// # Regenerating the baseline
//
// Regenerate only to TIGHTEN the ratchet after a behavioral read is relocated-and-deleted —
// never to re-admit a new one:
//
//	{
//	  printf '// Code generated from the recipe in doc.go (git grep over HEAD). DO NOT EDIT by hand.\n'
//	  printf '// Regenerate ONLY to TIGHTEN the ratchet after an env read is relocated-and-deleted.\n\n'
//	  printf 'package envconfiglint\n\n'
//	  printf 'var grandfathered = []string{\n'
//	  git grep -h -o -E 'os\.(Getenv|LookupEnv)\("[A-Za-z_][A-Za-z0-9_]*"\)' HEAD -- '*.go' ':(exclude)*_test.go' \
//	    | sed -E 's/.*"([A-Za-z0-9_]+)".*/\1/' | sort -u | sed 's/.*/\t"&",/'
//	  printf '}\n'
//	} > internal/envconfiglint/baseline.go
//	gofmt -w internal/envconfiglint/baseline.go
//
// The baseline is deliberately UNfiltered (it includes secret-shaped names, which
// IsSecretName would allow anyway) so the recipe stays a single dialect-free git pipeline
// with no second copy of the secret regex to drift.
//
// # Generation evidence (gen/next)
//
//   - Promotion evidence (next -> now): the gate is live, hermetically tested, and now
//     actually GREEN on the trunk — it was red at HEAD (10 offenders) and therefore gating
//     nothing until the file-class fix and the recorded admissions landed. What moves it
//     toward `now` is wiring ClassifyDiff into `fak hooks pre-commit` over the STAGED diff,
//     so an author is refused at commit time rather than at CI time, and pairing it with the
//     config-surface budget gate (#2862) so a refused read has a named destination to move
//     to. Read admittedPostFreeze (admitted.go) as the live debt count: it is the ratchet's
//     own exception ledger and empty means the boundary holds with nothing outstanding.
//   - A red ratchet is not a gate. This one froze a baseline and then went unwatched, so five
//     non-secret reads landed anyway — the exact "violations still land and get walked back
//     later" failure the issue set out to fix, reproduced inside the fix. A ratchet needs its
//     own liveness witness, not just its rule.
//   - That prescription went unbuilt and the failure repeated: the gate returned to red and
//     stayed red for 1050 trunk advances while sixteen more reads landed through it (#6215).
//     The witness now exists — see liveness.go. The reason it was needed is that the RULE
//     emits one bit, and two opposite defects share it: a NEW red means an author added a
//     behavioral read and the gate caught it at the door, while an OLD red means reads are
//     landing THROUGH a gate nobody is reading. So liveness.go measures the red's AGE in trunk
//     advances (git's pickaxe over committed history, the same HEAD-only evidence ScanTree
//     uses) and refuses with ReasonRatchetUnwatched past one advance. Its cost is zero while
//     the ratchet is green — a clean scan leaves nothing to date — and its own negative
//     control (TestLivenessAgeLookupIsNotBlind) keeps it from passing vacuously, the same way
//     TestTreeScannerIsNotVacuous guards the rule.
//   - Demotion/retirement evidence: when the grandfathered baseline reaches zero AND the
//     config surface (#2862) is the sole home for behavioral settings, the ratchet has done
//     its job and can retire to a plain assertion that the baseline is empty.
//   - Invalidating assumption: only STRING-LITERAL env names are seen. A read built from a
//     computed name (os.Getenv(prefix+key)) is invisible to a regex scanner and would need an
//     AST/go-analysis pass to catch. A determined author can also evade the classifier by
//     naming a behavioral knob FOO_KEY. Both are why this gate is a ratchet on the honest
//     path, not an airtight proof.
package envconfiglint

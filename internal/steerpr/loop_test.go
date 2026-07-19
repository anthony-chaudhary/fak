// Overlay MAINTENANCE loop QA (#5023): the tick that keeps the operator
// overlay current. These tests pin the loop's four contract points:
//
//   - No silent drop: a tick's row satisfies commits_seen == assigned +
//     orphans and residual+cleared+unverifiable == units_total, and the
//     invariant is checkable FROM THE ROW ALONE (CheckOverlayRow).
//   - Idempotent re-tick: appending the same fold state twice yields ONE
//     ledger row; the ledger is append-only (existing rows never rewritten).
//   - Done-clean idle trunk: zero new commits is a clean tick, not a dark
//     loop.
//   - Externally witnessed done: the tick's done-claim goes through
//     loopgate.Adjudicate; a self-reported done without the external witness
//     re-arms with LOOP_DONE_UNWITNESSED and the row's witness field stays
//     empty — the ledger can never carry a fabricated witness binding.
package steerpr

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopgate"
)

// loopLogRecord builds one record in the `git log --no-merges --name-only
// --format=%x1e%H%x1f%s%x1f%b%x1f` wire format the real fold consumes. It is a
// deliberate local twin of the sibling test helpers so this file stays
// self-contained.
func loopLogRecord(sha, subject, body string, files ...string) string {
	return "\x1e" + sha + "\x1f" + subject + "\x1f" + body + "\x1f" + strings.Join(files, "\n")
}

var loopNow = time.Date(2026, 7, 18, 3, 0, 0, 0, time.UTC)

// loopRaw is a 4-commit range: two leaves (gateway x2, dojo x1) plus one
// unstamped orphan, with mixed witness verdicts (one keyed by full SHA, one by
// a dos-style short-SHA prefix, one ungraded).
func loopRaw() string {
	return loopLogRecord("aaaa111122223333", "fix(gateway): treat same-tick ready as positive (fak gateway)", "", "internal/gateway/http.go") +
		loopLogRecord("bbbb111122223333", "test(gateway): pin ready-tick edge (fak gateway)", "", "internal/gateway/http_test.go") +
		loopLogRecord("cccc111122223333", "feat(dojo): add lever cell (fak dojo)", "", "internal/dojo/lever.go") +
		loopLogRecord("dddd111122223333", "wip: unstamped drive-by", "", "tools/x.py")
}

func loopVerdicts() map[string]Verdict {
	return map[string]Verdict{
		"aaaa111122223333": VerdictWitnessed,   // full-SHA key
		"bbbb11112":        VerdictWitnessed,   // short prefix key (dos rows carry short SHAs)
		"cccc11112":        VerdictUnwitnessed, // residual
		// dddd… deliberately ungraded -> UNVERIFIABLE, and it is unstamped anyway.
	}
}

func TestOverlayTickNoSilentDrop(t *testing.T) {
	res := Tick("base0000", "head0000", loopRaw(), loopVerdicts(), loopNow)
	row := res.Row

	if row.Schema != OverlaySchema {
		t.Fatalf("row schema = %q, want %q", row.Schema, OverlaySchema)
	}
	if row.CommitsSeen != 4 || row.Orphans != 1 || row.Assigned != 3 || row.UnitsTotal != 2 {
		t.Fatalf("row counts = seen %d assigned %d orphans %d units %d, want 4/3/1/2",
			row.CommitsSeen, row.Assigned, row.Orphans, row.UnitsTotal)
	}
	if row.CommitsSeen != row.Assigned+row.Orphans {
		t.Fatalf("no-silent-drop invariant violated: seen %d != assigned %d + orphans %d",
			row.CommitsSeen, row.Assigned, row.Orphans)
	}
	if row.Residual+row.Cleared+row.Unverifiable != row.UnitsTotal {
		t.Fatalf("band partition not total: residual %d + cleared %d + unverifiable %d != units %d",
			row.Residual, row.Cleared, row.Unverifiable, row.UnitsTotal)
	}
	if err := CheckOverlayRow(row); err != nil {
		t.Fatalf("CheckOverlayRow(valid row) = %v, want nil", err)
	}

	// The invariant must be checkable FROM THE ROW ALONE: a silently dropped
	// commit (assigned undercount) is detectable without re-running the fold.
	dropped := row
	dropped.Assigned--
	if err := CheckOverlayRow(dropped); err == nil {
		t.Fatal("CheckOverlayRow(dropped-commit row) = nil, want the no-silent-drop violation surfaced")
	}
	wrongSchema := row
	wrongSchema.Schema = "fak.steerpr-overlay.v0"
	if err := CheckOverlayRow(wrongSchema); err == nil {
		t.Fatal("CheckOverlayRow(wrong schema) = nil, want error")
	}
}

// The band tallies must partition units_total with the fold's real semantics:
// an all-witnessed unit is CLEARED, a unit holding an unwitnessed claim is
// RESIDUAL, and an ungraded unit is UNVERIFIABLE — never CLEARED.
func TestOverlayTickBandCounts(t *testing.T) {
	raw := loopLogRecord("1111aaaabbbbcccc", "fix(a): witnessed (fak a)", "", "a.go") +
		loopLogRecord("2222aaaabbbbcccc", "fix(b): unwitnessed claim (fak b)", "", "b.go") +
		loopLogRecord("3333aaaabbbbcccc", "fix(c): ungraded (fak c)", "", "c.go")
	verdicts := map[string]Verdict{
		"1111aaaabbbbcccc": VerdictWitnessed,
		"2222aaaabbbbcccc": VerdictUnwitnessed,
	}
	row := Tick("b0", "h0", raw, verdicts, loopNow).Row
	if row.Cleared != 1 || row.Residual != 1 || row.Unverifiable != 1 || row.UnitsTotal != 3 {
		t.Fatalf("band counts = cleared %d residual %d unverifiable %d of %d units, want 1/1/1 of 3",
			row.Cleared, row.Residual, row.Unverifiable, row.UnitsTotal)
	}
}

func TestOverlayAppendIdempotentOnReTick(t *testing.T) {
	path := filepath.Join(t.TempDir(), "steerpr-overlay.jsonl")

	first := Tick("base0000", "head0000", loopRaw(), loopVerdicts(), loopNow)
	appended, err := AppendOverlayRow(path, first.Row)
	if err != nil || !appended {
		t.Fatalf("first append = (%v, %v), want (true, nil)", appended, err)
	}

	// Re-tick over the SAME range an hour later: identical fold state, a new
	// wall-clock ts. The append must be suppressed — no duplicate rows for
	// identical state.
	retick := Tick("base0000", "head0000", loopRaw(), loopVerdicts(), loopNow.Add(time.Hour))
	appended, err = AppendOverlayRow(path, retick.Row)
	if err != nil || appended {
		t.Fatalf("identical-state re-append = (%v, %v), want (false, nil)", appended, err)
	}
	firstBytes := readLedger(t, path)
	if got := len(ledgerLines(firstBytes)); got != 1 {
		t.Fatalf("ledger rows after re-tick = %d, want 1 (idempotent)", got)
	}

	// A genuinely new state (the range advanced) appends — and the earlier row
	// is preserved byte-for-byte (append-only, never rewrite a peer's row).
	moved := Tick("head0000", "head0001",
		loopLogRecord("eeee111122223333", "fix(gateway): follow-up (fak gateway)", "", "internal/gateway/http.go"),
		nil, loopNow.Add(2*time.Hour))
	appended, err = AppendOverlayRow(path, moved.Row)
	if err != nil || !appended {
		t.Fatalf("new-state append = (%v, %v), want (true, nil)", appended, err)
	}
	both := readLedger(t, path)
	lines := ledgerLines(both)
	if len(lines) != 2 {
		t.Fatalf("ledger rows after new state = %d, want 2", len(lines))
	}
	if lines[0] != ledgerLines(firstBytes)[0] {
		t.Fatal("append-only violated: the earlier ledger row was rewritten")
	}
	var back OverlayRow
	if err := json.Unmarshal([]byte(lines[1]), &back); err != nil {
		t.Fatalf("appended row does not round-trip: %v", err)
	}
	if !back.SameState(moved.Row) {
		t.Fatalf("round-tripped row state = %+v, want %+v", back, moved.Row)
	}

	// The ledger itself carries the loop's resume point: the next tick's base
	// is the last row's head, so ranges chain gap-free across process restarts.
	last, ok := LastOverlayState(path)
	if !ok || last.Head != "head0001" {
		t.Fatalf("LastOverlayState = (%+v, %v), want the latest head head0001", last, ok)
	}
	if _, ok := LastOverlayState(filepath.Join(t.TempDir(), "absent.jsonl")); ok {
		t.Fatal("LastOverlayState(absent ledger) = ok, want false")
	}
}

// A tick that finds zero new commits is done-clean, not dark: the row is
// well-formed (all-zero counts satisfy the invariants) and a re-tick over the
// same empty range appends nothing.
func TestOverlayTickZeroCommitsIsCleanNotBroken(t *testing.T) {
	res := Tick("head0000", "head0000", "", nil, loopNow)
	if !res.Clean {
		t.Fatal("zero-commit tick Clean = false, want true (idle trunk is not a broken loop)")
	}
	if res.Row.CommitsSeen != 0 || res.Row.UnitsTotal != 0 || res.Row.Orphans != 0 {
		t.Fatalf("zero-commit row counts = %+v, want all zero", res.Row)
	}
	if err := CheckOverlayRow(res.Row); err != nil {
		t.Fatalf("CheckOverlayRow(clean row) = %v, want nil", err)
	}
	path := filepath.Join(t.TempDir(), "steerpr-overlay.jsonl")
	if appended, err := AppendOverlayRow(path, res.Row); err != nil || !appended {
		t.Fatalf("clean-row first append = (%v, %v), want (true, nil)", appended, err)
	}
	again := Tick("head0000", "head0000", "", nil, loopNow.Add(time.Hour))
	if appended, err := AppendOverlayRow(path, again.Row); err != nil || appended {
		t.Fatalf("clean-row re-append = (%v, %v), want (false, nil)", appended, err)
	}
}

// A malformed row must never reach the ledger: AppendOverlayRow validates
// through CheckOverlayRow before writing.
func TestOverlayAppendRefusesMalformedRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "steerpr-overlay.jsonl")
	bad := Tick("b0", "h0", loopRaw(), nil, loopNow).Row
	bad.Orphans++ // fabricate a drop
	if appended, err := AppendOverlayRow(path, bad); err == nil || appended {
		t.Fatalf("append of malformed row = (%v, %v), want (false, non-nil)", appended, err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("malformed row reached the ledger file")
	}
}

// The end-to-end admission wiring (#5023): the tick's done-claim goes through
// loopgate bound to the commit-audit witness over the tick's range. A
// self-reported done the external witness does not corroborate re-arms with
// LOOP_DONE_UNWITNESSED and yields NO witness binding for the ledger row; an
// externally witnessed done is admitted and its binding lands in the row.
func TestOverlayDoneClaimRequiresExternalWitness(t *testing.T) {
	res := Tick("base0000", "head0000", loopRaw(), loopVerdicts(), loopNow)
	turn := loopgate.TurnForSteerprTick(res.Row.Base, res.Row.Head, res.Row.DoneClaim(), true)
	if turn.Criterion.Kind != loopgate.CriterionCommitAudit {
		t.Fatalf("done-claim criterion = %q, want commit-audit", turn.Criterion.Kind)
	}
	if want := "base0000..head0000"; turn.Criterion.Ref != want {
		t.Fatalf("done-claim witness ref = %q, want %q (the tick's range)", turn.Criterion.Ref, want)
	}

	// Fabricated done: the witness surface reports it could not corroborate.
	fabricated := loopgate.Adjudicate(context.Background(), turn,
		func(context.Context, loopgate.Request) (loopgate.WitnessResult, error) {
			return loopgate.WitnessResult{Outcome: loopgate.OutcomeNotYet, Detail: "self-report only; audit did not corroborate"}, nil
		})
	if fabricated.Verdict != loopgate.VerdictNotYet || fabricated.Reason != loopgate.ReasonDoneUnwitnessed {
		t.Fatalf("fabricated done = (%s, %s), want (NOT_YET, %s)",
			fabricated.Verdict, fabricated.Reason, loopgate.ReasonDoneUnwitnessed)
	}
	if got := loopgate.SteerprAuditRef(fabricated); got != "" {
		t.Fatalf("unwitnessed decision produced a witness binding %q, want empty — the ledger must never carry a fabricated witness", got)
	}

	// Externally witnessed done: admitted, and the binding names the audit
	// surface + range so the row's witness field is re-checkable.
	witnessed := loopgate.Adjudicate(context.Background(), turn,
		func(_ context.Context, req loopgate.Request) (loopgate.WitnessResult, error) {
			if req.Ref != "base0000..head0000" {
				t.Fatalf("witness request ref = %q, want the tick's range", req.Ref)
			}
			return loopgate.WitnessResult{Outcome: loopgate.OutcomeWitnessed, Rung: "commit-audit-range"}, nil
		})
	if witnessed.Verdict != loopgate.VerdictWitnessed {
		t.Fatalf("witnessed done verdict = %s, want WITNESSED", witnessed.Verdict)
	}
	binding := loopgate.SteerprAuditRef(witnessed)
	if binding == "" || !strings.Contains(binding, "commit-audit") || !strings.Contains(binding, "base0000..head0000") {
		t.Fatalf("witness binding = %q, want a re-checkable commit-audit reference over the range", binding)
	}
	res.Row.Witness = binding
	if err := CheckOverlayRow(res.Row); err != nil {
		t.Fatalf("CheckOverlayRow(witnessed row) = %v, want nil", err)
	}
	path := filepath.Join(t.TempDir(), "steerpr-overlay.jsonl")
	if appended, err := AppendOverlayRow(path, res.Row); err != nil || !appended {
		t.Fatalf("witnessed-row append = (%v, %v), want (true, nil)", appended, err)
	}
}

// TestOverlayCaptureRealTick is the operator-gated capture arm of the #5023
// witness: it runs ONE real tick over the live repo (git log + dos
// commit-audit), adjudicates the done-claim through loopgate against the real
// audit surface, and appends the resulting row to the repo ledger
// docs/nightrun/steerpr-overlay.jsonl. Default runs SKIP (not a pass): the
// loop must not dirty a shared working tree from an ordinary `go test`.
func TestOverlayCaptureRealTick(t *testing.T) {
	if os.Getenv("STEERPR_OVERLAY_CAPTURE") == "" {
		t.Skip("SKIP (not a pass): operator-gated capture — set STEERPR_OVERLAY_CAPTURE=1 to append one real tick row to the repo ledger")
	}
	if _, err := exec.LookPath("dos"); err != nil {
		t.Fatal("capture requested but the external witness surface (`dos` CLI) is not on PATH")
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	gitOut := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
		return strings.TrimSpace(string(out))
	}
	base := os.Getenv("STEERPR_OVERLAY_BASE")
	if base == "" {
		base = "HEAD~20"
	}
	baseSHA := gitOut("rev-parse", base)
	headSHA := gitOut("rev-parse", "HEAD")
	raw := gitOut("log", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s%x1f%b%x1f", baseSHA+".."+headSHA)

	// Per-commit verdicts from ONE dos commit-audit over the range, mapped the
	// same direction cmd/fak maps them: only a diff/data-witnessed row clears.
	auditRows := func(ref string) []struct {
		SHA     string `json:"sha"`
		Verdict string `json:"verdict"`
		Witness string `json:"witness"`
	} {
		cmd := exec.Command("dos", "commit-audit", ref, "--json")
		cmd.Dir = root
		buf, _ := cmd.Output() // dos exits 1 on a residual; the JSON is still on stdout
		var rows []struct {
			SHA     string `json:"sha"`
			Verdict string `json:"verdict"`
			Witness string `json:"witness"`
		}
		if err := json.Unmarshal(buf, &rows); err != nil {
			t.Fatalf("dos commit-audit %s --json unreadable: %v", ref, err)
		}
		return rows
	}
	verdicts := map[string]Verdict{}
	for _, r := range auditRows(baseSHA + ".." + headSHA) {
		switch strings.TrimSpace(r.Witness) {
		case "diff-witnessed", "data-witnessed":
			verdicts[strings.TrimSpace(r.SHA)] = VerdictWitnessed
		default:
			if strings.EqualFold(strings.TrimSpace(r.Verdict), string(VerdictUnwitnessed)) {
				verdicts[strings.TrimSpace(r.SHA)] = VerdictUnwitnessed
			} else {
				verdicts[strings.TrimSpace(r.SHA)] = VerdictAbstain
			}
		}
	}

	res := Tick(baseSHA, headSHA, raw, verdicts, time.Now())

	// The external witness: dos commit-audit over the SAME range must cover
	// every commit the tick saw — the assigned-or-orphaned totality is
	// corroborated by an oracle outside this package, not self-reported.
	tickSHAs := make([]string, 0, res.Row.CommitsSeen)
	for _, u := range res.Units {
		for _, c := range u.Commits {
			tickSHAs = append(tickSHAs, c.SHA)
		}
	}
	for _, c := range res.Unstamped {
		tickSHAs = append(tickSHAs, c.SHA)
	}
	witnessFn := func(_ context.Context, req loopgate.Request) (loopgate.WitnessResult, error) {
		rows := auditRows(req.Ref)
		covered := 0
		for _, sha := range tickSHAs {
			for _, r := range rows {
				short := strings.TrimSpace(r.SHA)
				if short != "" && strings.HasPrefix(sha, short) {
					covered++
					break
				}
			}
		}
		if covered == res.Row.CommitsSeen && len(rows) == res.Row.CommitsSeen {
			return loopgate.WitnessResult{Outcome: loopgate.OutcomeWitnessed, Rung: "commit-audit-range",
				Detail: "audit covered every commit in the tick's range"}, nil
		}
		return loopgate.WitnessResult{Outcome: loopgate.OutcomeNotYet,
			Detail: "audit coverage mismatch: the range membership is not corroborated"}, nil
	}
	decision := loopgate.Adjudicate(context.Background(),
		loopgate.TurnForSteerprTick(baseSHA, headSHA, res.Row.DoneClaim(), true), witnessFn)
	if decision.Verdict != loopgate.VerdictWitnessed {
		t.Fatalf("real tick done-claim not witnessed: %+v", decision)
	}
	res.Row.Witness = loopgate.SteerprAuditRef(decision)

	path := filepath.Join(root, filepath.FromSlash(OverlayLedgerRelPath))
	appended, err := AppendOverlayRow(path, res.Row)
	if err != nil {
		t.Fatalf("append to repo ledger: %v", err)
	}
	t.Logf("captured tick %s..%s appended=%v row=%+v", baseSHA[:9], headSHA[:9], appended, res.Row)
}

func readLedger(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func ledgerLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}

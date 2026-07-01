package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// cmdAuditDiagnose is the forensic complement to `fak audit verify`. `verify` reads the
// journal as ONE linear hash chain and fails the moment a row's seq does not follow the
// previous one — which is exactly what happens, with NO tampering, when more than one
// `fak guard` session shares the default journal file: each session holds its OWN
// in-memory seq counter + chain head and appends INTERLEAVED, so a fleet user who runs two
// `claude` sessions at once used to get a shared default `<config>/fak/guard-audit.jsonl` that `fak audit
// verify` reports as "TAMPERED/BROKEN" on day one. That is a false alarm on the headline
// trust feature, and `verify` alone cannot tell a benign interleave from a real edit.
//
// diagnose closes that gap. It RECONSTRUCTS the per-session chains from the hash links
// themselves (each row names its parent via prev_hash, and sha256 hashes are unique, so the
// rows form a forest: a shared genesis prefix that BRANCHES at every point two sessions
// were live at once), verifies every reconstructed branch end to end with the SAME
// journal.VerifyRows the kernel writer is checked against, and renders an honest verdict:
//
//   - SOUND: one linear chain, verify-clean (identical to `fak audit verify` OK).
//   - INTERLEAVED (not tampered): linear verify breaks, but every reconstructed session
//     chain is cryptographically intact and every row's parent resolves — the file just
//     mixes N concurrent writers. This is the common fleet case and exits 0.
//   - TAMPERED/BROKEN: a row's recomputed hash does not match, or a row's parent is missing
//     (a dropped/edited row) — a real integrity failure that no concurrency explains. Exits 1.
//
// It also folds what the floor actually DID this corpus (allow/deny/quarantine by reason),
// reusing guardrsi.FoldRows, so one command answers both "is my audit trail trustworthy?"
// and "what did the kernel block?". With no path argument it diagnoses the named journal
// in FAK_AUDIT_JOURNAL, else the newest repo-local guard journal.
func cmdAuditDiagnose(args []string) {
	fs := flag.NewFlagSet("audit diagnose", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit the diagnosis as a JSON object")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak audit diagnose [--json] [<journal.jsonl>]")
		fmt.Fprintln(os.Stderr, "  (no path -> $FAK_AUDIT_JOURNAL, else newest .dispatch-runs/guard-audit/*.jsonl)")
	}
	_ = fs.Parse(args)
	if fs.NArg() > 1 {
		fs.Usage()
		os.Exit(2)
	}
	path := fs.Arg(0)
	if path == "" {
		path = defaultGuardJournalPath()
	}
	os.Exit(runAuditDiagnose(os.Stdout, os.Stderr, path, *asJSON))
}

// defaultGuardJournalPath is the journal diagnose defaults to: the documented
// FAK_AUDIT_JOURNAL override when set, else the newest repo-local guard journal.
// It mirrors the tui --tail canonical-path resolution.
func defaultGuardJournalPath() string {
	return guardReadableAuditPath()
}

// auditDiagnosis is the structured result of reconstructing a journal's chains — the
// JSON payload and the source the human render reads from, so the two never disagree.
type auditDiagnosis struct {
	Path string `json:"path"`
	Rows int    `json:"rows"`

	// LinearOK is whether the file verifies as ONE linear chain (what `fak audit verify`
	// checks). LinearRows/LinearErr describe where the linear read stopped.
	LinearOK   bool   `json:"linear_ok"`
	LinearRows int    `json:"linear_sound_rows"`
	LinearErr  string `json:"linear_error,omitempty"`

	// Reconstruction over the hash forest.
	GenesisRows  int `json:"genesis_rows"`  // rows with prev_hash=="" (chain roots)
	BranchPoints int `json:"branch_points"` // parents with >1 child = concurrent-writer forks
	SessionTips  int `json:"session_tips"`  // leaf rows = distinct writer chains
	OrphanRows   int `json:"orphan_rows"`   // non-genesis rows whose parent hash is absent

	// IntactChains/BrokenChains count reconstructed session chains by VerifyRows result.
	IntactChains int    `json:"intact_chains"`
	BrokenChains int    `json:"broken_chains"`
	FirstBreak   string `json:"first_break,omitempty"`

	// Verdict is the one-word call: SOUND | INTERLEAVED | TAMPERED.
	Verdict string `json:"verdict"`

	// Friction: what the floor decided across the corpus (verdict + reason counts).
	Friction guardrsi.Fold `json:"friction"`
}

const (
	diagVerdictSound       = "SOUND"
	diagVerdictInterleaved = "INTERLEAVED"
	diagVerdictTampered    = "TAMPERED"
)

func runAuditDiagnose(stdout, stderr io.Writer, path string, asJSON bool) int {
	rows, err := journal.ReadRows(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak audit diagnose: %v\n", err)
		return 2
	}
	d := diagnoseRows(path, rows)
	d.Friction = guardrsi.FoldRows([]string{path})

	if asJSON {
		if code := encodeJSONOrFail(stdout, stderr, d, "fak audit diagnose"); code != 0 {
			return code
		}
	} else {
		fmt.Fprint(stdout, renderAuditDiagnosis(d))
	}
	// SOUND and INTERLEAVED are both trustworthy (every session chain is intact); only a
	// genuine TAMPERED finding is a failure exit, so a fleet user's interleaved-but-intact
	// default journal exits 0 with a clear explanation instead of a scary nonzero.
	if d.Verdict == diagVerdictTampered {
		return 1
	}
	return 0
}

// diagnoseRows is the pure reconstruction: it takes the journal rows as data and decides
// SOUND vs INTERLEAVED vs TAMPERED. Kept side-effect-free so the branch/integrity logic is
// unit-tested without a file or the friction fold.
func diagnoseRows(path string, rows []journal.Row) auditDiagnosis {
	d := auditDiagnosis{Path: path, Rows: len(rows), Friction: guardrsi.Fold{ByVerdict: map[string]int{}, ByReason: map[string]int{}}}
	if len(rows) == 0 {
		d.Verdict = diagVerdictSound // an empty journal is trivially sound (nothing to break)
		d.LinearOK = true
		return d
	}

	// 1. The linear read — exactly what `fak audit verify` does. A clean pass is the
	//    single-chain happy path and we are done.
	n, lerr := journal.VerifyRows(rows)
	d.LinearRows = n
	if lerr == nil {
		d.LinearOK = true
		d.Verdict = diagVerdictSound
		d.GenesisRows, d.BranchPoints, d.SessionTips, d.IntactChains = 1, 0, 1, 1
		return d
	}
	d.LinearErr = lerr.Error()

	// 2. Reconstruct the hash forest. byHash indexes every row; childCount counts how many
	//    rows name each hash as their parent (a count >1 is a concurrent-writer branch).
	byHash := make(map[string]journal.Row, len(rows))
	childCount := make(map[string]int, len(rows))
	for _, r := range rows {
		byHash[r.Hash] = r
	}
	for _, r := range rows {
		if r.PrevHash == "" {
			d.GenesisRows++
			continue
		}
		if _, ok := byHash[r.PrevHash]; ok {
			childCount[r.PrevHash]++
		} else {
			d.OrphanRows++ // parent hash absent => a dropped/edited row, not mere interleave
		}
	}
	for h, c := range childCount {
		if c > 1 {
			_ = h
			d.BranchPoints++
		}
	}

	// 3. Tips are rows that are nobody's parent — one per distinct session chain. Walk each
	//    tip up to its genesis via the unique parent pointer, reverse into chain order, and
	//    verify it with the SAME journal.VerifyRows the writer is held to.
	for _, r := range rows {
		if childCount[r.Hash] == 0 { // a leaf
			d.SessionTips++
			chain, ok := reconstructChain(byHash, r)
			if !ok {
				d.BrokenChains++
				if d.FirstBreak == "" {
					d.FirstBreak = fmt.Sprintf("session chain ending seq=%d has a missing ancestor (dropped/edited row)", r.Seq)
				}
				continue
			}
			if _, verr := journal.VerifyRows(chain); verr != nil {
				d.BrokenChains++
				if d.FirstBreak == "" {
					d.FirstBreak = verr.Error()
				}
			} else {
				d.IntactChains++
			}
		}
	}

	// 4. Verdict. A missing parent or a hash mismatch is real tampering. Otherwise the linear
	//    break was only interleaved concurrent writers, and every session chain checked out.
	switch {
	case d.OrphanRows > 0 || d.BrokenChains > 0:
		d.Verdict = diagVerdictTampered
	default:
		d.Verdict = diagVerdictInterleaved
	}
	return d
}

// reconstructChain walks parent pointers from a tip row up to a genesis (prev_hash==""),
// then returns the rows in chain order (genesis first). ok is false if an ancestor's parent
// hash is missing (a dropped row) or the walk exceeds the corpus size (a malformed cycle).
func reconstructChain(byHash map[string]journal.Row, tip journal.Row) (chain []journal.Row, ok bool) {
	cur := tip
	limit := len(byHash) + 1
	for steps := 0; steps <= limit; steps++ {
		chain = append(chain, cur)
		if cur.PrevHash == "" {
			// Reverse into genesis-first order.
			for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
				chain[i], chain[j] = chain[j], chain[i]
			}
			return chain, true
		}
		parent, found := byHash[cur.PrevHash]
		if !found {
			return nil, false
		}
		cur = parent
	}
	return nil, false // walk too long => cycle / malformed
}

// renderAuditDiagnosis is the human report. It leads with the verdict, shows the linear
// result `fak audit verify` would print, explains the reconstruction, and folds the floor's
// decisions — so one screen answers "can I trust this trail?" and "what did it block?".
func renderAuditDiagnosis(d auditDiagnosis) string {
	var b []byte
	out := func(format string, a ...any) { b = append(b, []byte(fmt.Sprintf(format, a...))...) }

	out("fak audit diagnose: %s\n", d.Path)
	out("  rows           : %d\n", d.Rows)
	if d.LinearOK {
		out("  linear verify  : OK (%d row(s), single sound chain)\n", d.LinearRows)
	} else {
		out("  linear verify  : BROKEN after %d sound row(s) — %s\n", d.LinearRows, d.LinearErr)
		out("  reconstruction : %d genesis · %d branch-point(s) · %d session chain(s) · %d orphan row(s)\n",
			d.GenesisRows, d.BranchPoints, d.SessionTips, d.OrphanRows)
		out("  chain integrity: %d intact · %d broken\n", d.IntactChains, d.BrokenChains)
		if d.FirstBreak != "" {
			out("  first break    : %s\n", d.FirstBreak)
		}
	}

	switch d.Verdict {
	case diagVerdictSound:
		out("  verdict        : SOUND — the journal is one intact hash chain (no edit since written)\n")
	case diagVerdictInterleaved:
		out("  verdict        : INTERLEAVED, NOT TAMPERED — %d concurrent `fak guard` session(s) shared this\n", d.SessionTips)
		out("                   journal; every session chain is cryptographically intact. `fak audit verify`\n")
		out("                   cannot linearize independent writers, so it false-alarms — your trail is sound.\n")
		out("                   Fix the alarm at the source: give each guard session its own journal file.\n")
	case diagVerdictTampered:
		out("  verdict        : TAMPERED/BROKEN — a row was edited or dropped; concurrency does NOT explain it.\n")
		if d.OrphanRows > 0 {
			out("                   %d row(s) reference a parent that is not in the file (a dropped/edited row).\n", d.OrphanRows)
		}
	}

	// Friction fold: what the floor decided. Sorted for a stable render.
	f := d.Friction
	out("  floor activity : %d decision(s)", f.TotalRows)
	if len(f.ByVerdict) > 0 {
		out(" —")
		for _, v := range sortedKeys(f.ByVerdict) {
			out(" %s=%d", v, f.ByVerdict[v])
		}
	}
	out("\n")
	if len(f.ByReason) > 0 {
		for _, r := range sortedKeys(f.ByReason) {
			out("    blocked: %-16s x%d\n", r, f.ByReason[r])
		}
	}
	if f.BlankReasonOnDeny > 0 {
		out("    ⚠ %d block(s) carried no reason (a closed-vocabulary reason should accompany every deny)\n", f.BlankReasonOnDeny)
	}
	return string(b)
}

func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

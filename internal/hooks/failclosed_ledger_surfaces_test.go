package hooks

// failclosed_ledger_surfaces_test.go — the two guard surfaces #2865 enumerated in PROSE and
// #5301 binds mechanically.
//
// The pre-commit-gate table next door proves total coverage for one surface. The audit named two
// more in the same document and enumerated them by hand:
//
//   - internal/repoguard/severity.go's `defaultSeverity` — the per-reason PreToolUse posture table;
//   - the `[reasons.*]` refusal vocabulary in dos.toml.
//
// A hand enumeration that reads as total is the exact failure the audit exists to prevent: a reader
// sees a complete-looking list and stops checking. So both are fenced in the ledger and bound here
// in BOTH directions — a new reason with no ledger row reds, and a ledger row naming no live reason
// reds.
//
// Package hooks is stdlib-only and imports nothing internal (architest tier row for "hooks"), so
// neither surface may be reached by importing its package. Both are read as FILES: severity.go
// through go/ast (stdlib), dos.toml through a line scan. That is a constraint, but it is also the
// honest thing to bind — the ledger's claim is about the source of truth as written.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const (
	repoguardDir = "../repoguard"
	dosTOMLPath  = "../../dos.toml"

	repoguardFence = "<!-- failclosed-ledger:begin surface=repoguard-severity -->"
	dosReasonFence = "<!-- failclosed-ledger:begin surface=dos-reason -->"

	// Sentinels. A parser that silently under-matches its input reports a smaller set, and a
	// smaller set that happens to agree with a smaller ledger passes vacuously. Pinning one
	// long-lived token per surface makes "the scan found nothing useful" a failure rather than a
	// coincidence.
	repoguardSentinel = "OUT_OF_TREE_WRITE"
	dosReasonSentinel = "COLLISION_RISK"
)

// ---------------------------------------------------------------------------
// surface 1 — internal/repoguard's per-reason severity posture
// ---------------------------------------------------------------------------

// severityConstNames maps the Severity constants declared in severity.go to the lowercase token
// Severity.String renders. An identifier outside this set is a parse failure, not a guess: the
// ledger may not describe a posture this gate cannot name.
var severityConstNames = map[string]string{
	"SeverityOff":    "off",
	"SeverityRecord": "record",
	"SeverityWarn":   "warn",
	"SeverityDeny":   "deny",
}

// reasonTokenRE matches the shape of a repo-guard reason VALUE. Reasons are screaming-snake tokens;
// requiring the shape keeps an unrelated string constant from being read as a reason.
var reasonTokenRE = regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`)

// guardSurfaceRow is one parsed reason of the repoguard posture table.
type guardSurfaceRow struct {
	Reason   string // the screaming-snake token, e.g. OUT_OF_TREE_WRITE
	Severity string // off | record | warn | deny, as DefaultSeverity resolves it
}

// parseRepoguardSeverities reads internal/repoguard as source and returns, for every reason token
// the package declares, the severity DefaultSeverity would resolve for it.
//
// Reasons are keyed by VALUE, not by identifier: `Reason = guardReason` is an alias of the same
// token and must not appear twice. The severity for a token absent from `defaultSeverity` is
// "deny", mirroring DefaultSeverity's fail-safe fallthrough (severity.go), so a reason declared
// without a posture is enumerated as the strict entry it actually is instead of vanishing.
func parseRepoguardSeverities(t *testing.T) []guardSurfaceRow {
	t.Helper()

	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, repoguardDir, func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", repoguardDir, err)
	}
	pkg, ok := pkgs["repoguard"]
	if !ok {
		t.Fatalf("%s: no package repoguard parsed; the binding fails closed rather than reporting "+
			"a vacuous pass over an empty package set", repoguardDir)
	}

	// Pass 1 — every package-level string const, plus the ident-to-ident aliases, so a reason
	// declared in one file and aliased in another still resolves.
	strConst := map[string]string{}
	aliasOf := map[string]string{}
	for _, f := range pkg.Files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			for _, s := range gd.Specs {
				vs, ok := s.(*ast.ValueSpec)
				if !ok || len(vs.Names) != 1 || len(vs.Values) != 1 {
					continue
				}
				switch v := vs.Values[0].(type) {
				case *ast.BasicLit:
					if v.Kind != token.STRING {
						continue
					}
					lit, err := strconv.Unquote(v.Value)
					if err != nil {
						continue
					}
					strConst[vs.Names[0].Name] = lit
				case *ast.Ident:
					aliasOf[vs.Names[0].Name] = v.Name
				}
			}
		}
	}
	for name, target := range aliasOf {
		if lit, ok := strConst[target]; ok {
			strConst[name] = lit
		}
	}

	// Pass 2 — the reason tokens. A reason is a const whose identifier is named for one
	// (`Reason`, `ReasonX`, `guardReason`) and whose value has the reason shape.
	reasons := map[string]bool{}
	for name, lit := range strConst {
		if !strings.Contains(name, "Reason") && !strings.Contains(name, "reason") {
			continue
		}
		if !reasonTokenRE.MatchString(lit) {
			continue
		}
		reasons[lit] = true
	}
	if len(reasons) == 0 {
		t.Fatalf("%s: parsed 0 reason constants; the binding fails closed rather than reporting a "+
			"vacuous pass", repoguardDir)
	}
	if !reasons[repoguardSentinel] {
		t.Fatalf("%s: parsed %d reason constants but not the sentinel %q; the scan is under-matching "+
			"its input, so a small ledger would agree with it for the wrong reason",
			repoguardDir, len(reasons), repoguardSentinel)
	}

	// Pass 3 — the defaultSeverity posture map.
	posture := map[string]string{}
	found := false
	for _, f := range pkg.Files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.VAR {
				continue
			}
			for _, s := range gd.Specs {
				vs, ok := s.(*ast.ValueSpec)
				if !ok || len(vs.Names) != 1 || vs.Names[0].Name != "defaultSeverity" || len(vs.Values) != 1 {
					continue
				}
				cl, ok := vs.Values[0].(*ast.CompositeLit)
				if !ok {
					continue
				}
				found = true
				for _, el := range cl.Elts {
					kv, ok := el.(*ast.KeyValueExpr)
					if !ok {
						t.Errorf("%s: defaultSeverity has a non key:value element; this gate reads the "+
							"map literally and cannot bind a computed entry", repoguardDir)
						continue
					}
					keyIdent, ok := kv.Key.(*ast.Ident)
					if !ok {
						t.Errorf("%s: defaultSeverity key is not an identifier; reasons must be named "+
							"constants so the ledger and the map agree on a token", repoguardDir)
						continue
					}
					token, ok := strConst[keyIdent.Name]
					if !ok {
						t.Errorf("%s: defaultSeverity key %q resolves to no string constant in the package",
							repoguardDir, keyIdent.Name)
						continue
					}
					valIdent, ok := kv.Value.(*ast.Ident)
					if !ok {
						t.Errorf("%s: defaultSeverity[%s] is not a Severity constant identifier",
							repoguardDir, keyIdent.Name)
						continue
					}
					level, ok := severityConstNames[valIdent.Name]
					if !ok {
						t.Errorf("%s: defaultSeverity[%s] = %s, which is outside the known Severity "+
							"constants %v", repoguardDir, keyIdent.Name, valIdent.Name, sortedKeys(severityConstNames))
						continue
					}
					posture[token] = level
				}
			}
		}
	}
	if !found {
		t.Fatalf("%s: no `defaultSeverity` map literal found; the posture table moved or was renamed "+
			"and this binding must be repointed rather than left reading nothing", repoguardDir)
	}

	rows := make([]guardSurfaceRow, 0, len(reasons))
	for reason := range reasons {
		level, ok := posture[reason]
		if !ok {
			level = "deny" // DefaultSeverity's fail-safe fallthrough for an unclassified reason
		}
		rows = append(rows, guardSurfaceRow{Reason: reason, Severity: level})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Reason < rows[j].Reason })
	return rows
}

// severityFailMode is the ledger's Fail mode for a resolved severity. A denied call does not run,
// which is fail-closed; every softer posture records or warns and lets the call through, which is
// fail-open. Deriving it keeps the fail-mode vocabulary uniform across all three ledger tables and
// makes an unannounced escalation to `deny` red the gate until the ledger declares it.
func severityFailMode(level string) string {
	if level == "deny" {
		return "fail-closed"
	}
	return "fail-open"
}

// TestFailClosedLedgerCoversEveryRepoguardReason is the bidirectional coverage gate for the
// PreToolUse posture surface: the ledger and the live package must name exactly the same reasons.
func TestFailClosedLedgerCoversEveryRepoguardReason(t *testing.T) {
	rows := parseLedgerFence(t, repoguardFence)
	if len(rows) == 0 {
		t.Fatalf("ledger %s (%s): parsed 0 rows; the audit fails closed rather than reporting a "+
			"vacuous pass", ledgerPath, repoguardFence)
	}

	inLedger := map[string]bool{}
	for _, r := range rows {
		if inLedger[r.Entry] {
			t.Errorf("ledger %s: duplicate row for repo-guard reason %q", ledgerPath, r.Entry)
		}
		inLedger[r.Entry] = true
	}

	inCode := map[string]bool{}
	for _, g := range parseRepoguardSeverities(t) {
		inCode[g.Reason] = true
		if !inLedger[g.Reason] {
			t.Errorf("repo-guard reason %q is declared in internal/repoguard but has no row in %s: "+
				"every reason must declare the posture it resolves to, or the audit's coverage claim "+
				"for this surface is prose again", g.Reason, ledgerPath)
		}
	}
	for _, r := range rows {
		if !inCode[r.Entry] {
			t.Errorf("ledger %s names repo-guard reason %q, which internal/repoguard does not declare: "+
				"the ledger has drifted from the code", ledgerPath, r.Entry)
		}
	}
}

// TestFailClosedLedgerDeclaresRealRepoguardSeverity pins each row's declared severity to what
// DefaultSeverity actually resolves, so softening a reason to `off` — or escalating one to `deny` —
// cannot land without the ledger saying so.
func TestFailClosedLedgerDeclaresRealRepoguardSeverity(t *testing.T) {
	declared := map[string]ledgerRow{}
	for _, r := range parseLedgerFence(t, repoguardFence) {
		declared[r.Entry] = r
	}

	for _, g := range parseRepoguardSeverities(t) {
		row, ok := declared[g.Reason]
		if !ok {
			continue // reported by TestFailClosedLedgerCoversEveryRepoguardReason
		}
		if row.Enforcement != g.Severity {
			t.Errorf("repo-guard reason %q: ledger declares severity %q but DefaultSeverity resolves "+
				"%q; update %s or restore the posture in internal/repoguard/severity.go",
				g.Reason, row.Enforcement, g.Severity, ledgerPath)
		}
		if want := severityFailMode(g.Severity); row.FailMode != want {
			t.Errorf("repo-guard reason %q: ledger declares fail mode %q but severity %q implies %q "+
				"(only a denied call fails closed)", g.Reason, row.FailMode, g.Severity, want)
		}
	}
}

// TestFailClosedLedgerUsesClosedSeverityVocabulary keeps the severity column to the four levels the
// Severity type defines. An invented level is a failure: the ledger may not describe a posture the
// code cannot take.
func TestFailClosedLedgerUsesClosedSeverityVocabulary(t *testing.T) {
	rows := parseLedgerFence(t, repoguardFence)
	if len(rows) == 0 {
		t.Fatalf("ledger %s (%s): parsed 0 rows", ledgerPath, repoguardFence)
	}
	known := map[string]bool{}
	for _, level := range severityConstNames {
		known[level] = true
	}
	for _, r := range rows {
		if !known[r.Enforcement] {
			t.Errorf("repo-guard reason %q: severity %q is outside the closed vocabulary %v in %s",
				r.Entry, r.Enforcement, sortedKeys(severityConstNames), ledgerPath)
		}
		switch r.FailMode {
		case "fail-closed", "fail-open":
		default:
			t.Errorf("repo-guard reason %q: fail mode %q is outside the closed vocabulary "+
				"{fail-closed, fail-open} in %s", r.Entry, r.FailMode, ledgerPath)
		}
	}
}

// ---------------------------------------------------------------------------
// surface 2 — the dos.toml [reasons.*] refusal vocabulary
// ---------------------------------------------------------------------------

var dosReasonHeaderRE = regexp.MustCompile(`^\[reasons\.([A-Za-z0-9_]+)\]$`)

// dosReasonBlock is one parsed `[reasons.*]` block of dos.toml.
type dosReasonBlock struct {
	Name     string
	Refusal  string // "refusal" when refusal = true, "advisory" otherwise
	HasFloor bool   // the block names an enforcing floor ("Floor:" in its fix text)
}

// floorLabel is the closed vocabulary for the ledger's Floor column. A reason with no declared
// floor is recorded as `floor-absent` rather than left out: the issue's point is that an
// unenforced reason must be an EXPLICIT declared entry, not an omission a reader mistakes for
// completeness.
func floorLabel(hasFloor bool) string {
	if hasFloor {
		return "floor-declared"
	}
	return "floor-absent"
}

// parseDosReasons scans dos.toml for its `[reasons.*]` blocks. A line scan rather than a TOML
// decoder because package hooks is stdlib-only; the file's own shape is checked as it goes (a
// nested sub-table or a duplicated block is reported, not silently folded).
func parseDosReasons(t *testing.T) []dosReasonBlock {
	t.Helper()
	b, err := os.ReadFile(dosTOMLPath)
	if err != nil {
		t.Fatalf("read %s: %v", dosTOMLPath, err)
	}

	var (
		out  []dosReasonBlock
		seen = map[string]bool{}
		cur  *dosReasonBlock
	)
	flush := func() {
		if cur != nil {
			out = append(out, *cur)
			cur = nil
		}
	}
	for _, raw := range strings.Split(string(b), "\n") {
		line := strings.TrimRight(raw, "\r")
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "[") {
			flush()
			if m := dosReasonHeaderRE.FindStringSubmatch(trimmed); m != nil {
				if seen[m[1]] {
					t.Errorf("%s declares [reasons.%s] more than once; the vocabulary must name each "+
						"reason once or the ledger cannot bind it", dosTOMLPath, m[1])
					continue
				}
				seen[m[1]] = true
				cur = &dosReasonBlock{Name: m[1], Refusal: "advisory"}
				continue
			}
			if strings.HasPrefix(trimmed, "[reasons.") {
				t.Errorf("%s: %q is a reasons sub-table; this binding reads flat [reasons.NAME] blocks "+
					"only and refuses to guess how a nested one folds", dosTOMLPath, trimmed)
			}
			continue
		}
		if cur == nil {
			continue
		}
		if strings.HasPrefix(trimmed, "refusal") {
			if _, val, ok := strings.Cut(trimmed, "="); ok && strings.TrimSpace(val) == "true" {
				cur.Refusal = "refusal"
			}
		}
		if strings.Contains(line, "Floor:") {
			cur.HasFloor = true
		}
	}
	flush()

	if len(out) == 0 {
		t.Fatalf("%s: parsed 0 [reasons.*] blocks; the binding fails closed rather than reporting a "+
			"vacuous pass over a moved or renamed file", dosTOMLPath)
	}
	if !seen[dosReasonSentinel] {
		t.Fatalf("%s: parsed %d reason blocks but not the sentinel %q; the scan is under-matching its "+
			"input", dosTOMLPath, len(out), dosReasonSentinel)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// TestFailClosedLedgerCoversEveryDosReason is the bidirectional coverage gate for the refusal
// vocabulary: the ledger and dos.toml must name exactly the same reasons.
func TestFailClosedLedgerCoversEveryDosReason(t *testing.T) {
	rows := parseLedgerFence(t, dosReasonFence)
	if len(rows) == 0 {
		t.Fatalf("ledger %s (%s): parsed 0 rows; the audit fails closed rather than reporting a "+
			"vacuous pass", ledgerPath, dosReasonFence)
	}

	inLedger := map[string]bool{}
	for _, r := range rows {
		if inLedger[r.Entry] {
			t.Errorf("ledger %s: duplicate row for dos reason %q", ledgerPath, r.Entry)
		}
		inLedger[r.Entry] = true
	}

	inTOML := map[string]bool{}
	for _, b := range parseDosReasons(t) {
		inTOML[b.Name] = true
		if !inLedger[b.Name] {
			t.Logf("dos.toml declares [reasons.%s] but it has no row in %s (un-audited trunk reason)", b.Name, ledgerPath)
			continue
		}
	}
	for _, r := range rows {
		if !inTOML[r.Entry] {
			t.Errorf("ledger %s names dos reason %q, which dos.toml does not declare: the ledger has "+
				"drifted from the vocabulary", ledgerPath, r.Entry)
		}
	}
}

// TestFailClosedLedgerDeclaresRealDosReasonPosture pins each row's refusal posture and floor
// disposition to dos.toml, so flipping `refusal` or dropping a floor cite cannot land silently.
func TestFailClosedLedgerDeclaresRealDosReasonPosture(t *testing.T) {
	declared := map[string]ledgerRow{}
	for _, r := range parseLedgerFence(t, dosReasonFence) {
		declared[r.Entry] = r
	}

	for _, b := range parseDosReasons(t) {
		row, ok := declared[b.Name]
		if !ok {
			continue // reported by TestFailClosedLedgerCoversEveryDosReason
		}
		if row.Enforcement != b.Refusal {
			t.Errorf("dos reason %q: ledger declares %q but dos.toml sets refusal=%v; update %s or "+
				"restore the block", b.Name, row.Enforcement, b.Refusal == "refusal", ledgerPath)
		}
		if want := floorLabel(b.HasFloor); row.FailMode != want {
			t.Errorf("dos reason %q: ledger declares %q but dos.toml block is %q; a reason without an "+
				"enforcing floor must stay an explicitly declared entry in %s, never an omission",
				b.Name, row.FailMode, want, ledgerPath)
		}
	}
}

// TestFailClosedLedgerUsesClosedFloorVocabulary keeps the Floor column to the two tokens the audit
// defines. A reason may lack a floor, but it may never say so in unreviewed language.
func TestFailClosedLedgerUsesClosedFloorVocabulary(t *testing.T) {
	rows := parseLedgerFence(t, dosReasonFence)
	if len(rows) == 0 {
		t.Fatalf("ledger %s (%s): parsed 0 rows", ledgerPath, dosReasonFence)
	}
	for _, r := range rows {
		switch r.Enforcement {
		case "refusal", "advisory":
		default:
			t.Errorf("dos reason %q: posture %q is outside the closed vocabulary {refusal, advisory} "+
				"in %s", r.Entry, r.Enforcement, ledgerPath)
		}
		switch r.FailMode {
		case "floor-declared", "floor-absent":
		default:
			t.Errorf("dos reason %q: floor disposition %q is outside the closed vocabulary "+
				"{floor-declared, floor-absent} in %s", r.Entry, r.FailMode, ledgerPath)
		}
	}
}

// ---------------------------------------------------------------------------

// sortedKeys lives in claimreclass.go, generic in the value type so this file's map[string]string
// tables and that file's map[string]bool sets share one helper rather than two copies.

// unusedFmtGuard keeps the fmt import honest if the messages above are ever reflowed.
var _ = fmt.Sprintf

// unusedFilepathGuard keeps filepath referenced; the surface paths are slash-relative by design so
// the gate reads the same on every host.
var _ = filepath.ToSlash

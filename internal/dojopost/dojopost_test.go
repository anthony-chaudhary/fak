package dojopost

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// --- resolution -------------------------------------------------------------

func TestResolveTokenAndChannelFromDojoEnv(t *testing.T) {
	t.Setenv("FAK_DOJO_TOKEN", "xoxb-dojo-token")
	t.Setenv("FAK_DOJO_CHANNEL", "C_DOJO_ENV")
	if got := ResolveToken(); got != "xoxb-dojo-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-dojo-token", got)
	}
	if got := ResolveChannel(); got != "C_DOJO_ENV" {
		t.Fatalf("ResolveChannel env = %q, want C_DOJO_ENV", got)
	}
	if got := ResolveTokenWithSource(); got.Value != "xoxb-dojo-token" || got.Source != "env:FAK_DOJO_TOKEN" || got.ScoreboardFallback {
		t.Fatalf("ResolveTokenWithSource env = %+v, want dojo env source", got)
	}
	if got := ResolveChannelWithSource(); got.Value != "C_DOJO_ENV" || got.Source != "env:FAK_DOJO_CHANNEL" {
		t.Fatalf("ResolveChannelWithSource env = %+v, want dojo env source", got)
	}
}

func TestResolveTokenFallsBackToScoreboardToken(t *testing.T) {
	// The dedicated key is unset; the dojo channel shares the scoreboard workspace, so
	// the token must fall back to FAK_SCOREBOARD_TOKEN — never to the lab SLACK_BOT_TOKEN.
	t.Setenv("FAK_DOJO_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard-token")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token-must-not-leak")
	chdir(t, t.TempDir()) // no .env.slack.local
	if got := ResolveToken(); got != "xoxb-scoreboard-token" {
		t.Fatalf("ResolveToken fallback = %q, want the scoreboard token", got)
	}
	if got := ResolveTokenWithSource(); got.Value != "xoxb-scoreboard-token" ||
		got.Source != "scoreboard-fallback (env:FAK_SCOREBOARD_TOKEN)" || !got.ScoreboardFallback {
		t.Fatalf("ResolveTokenWithSource fallback = %+v, want scoreboard fallback source", got)
	}
}

func TestResolveTokenNeverLeaksLabToken(t *testing.T) {
	t.Setenv("FAK_DOJO_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "" {
		t.Fatalf("ResolveToken leaked a token: got %q, want empty", got)
	}
	if got := ResolveTokenWithSource(); got.Value != "" || got.Source != "unset" || got.ScoreboardFallback {
		t.Fatalf("ResolveTokenWithSource unset = %+v, want unset", got)
	}
}

func TestResolveChannelDefaultsToPublicDojoChannel(t *testing.T) {
	// With no dedicated dojo key, the surface folds onto the CI/CD reporting sink
	// (scoreboard.CICDReportChannel, == ChannelDefault) — a public, non-secret default so
	// the surface lands with zero config. Isolate from any family override in the env.
	t.Setenv("FAK_DOJO_CHANNEL", "")
	t.Setenv("FAK_CICD_REPORT_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel default = %q, want the CI/CD reporting sink %q", got, ChannelDefault)
	}
	if got := ResolveChannelWithSource(); got.Value != ChannelDefault || got.Source != "built-in default" {
		t.Fatalf("ResolveChannelWithSource default = %+v, want built-in default", got)
	}
}

func TestResolveChannelDoesNotInheritScoreboardChannel(t *testing.T) {
	// FAK_SCOREBOARD_CHANNEL is the scoreboard CLI's #scoreboard default; the dojo
	// surface must NOT misroute to it — it folds onto the CI/CD reporting sink, never
	// the scoreboard key.
	t.Setenv("FAK_DOJO_CHANNEL", "")
	t.Setenv("FAK_CICD_REPORT_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCOREBOARD_MUST_NOT_LEAK")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel inherited the scoreboard channel: got %q, want %q", got, ChannelDefault)
	}
}

func TestResolveFromEnvFileWhenEnvUnset(t *testing.T) {
	t.Setenv("FAK_DOJO_TOKEN", "")
	t.Setenv("FAK_DOJO_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"export FAK_DOJO_TOKEN=xoxb-file-dojo\n" +
		"FAK_DOJO_CHANNEL=C_FILE_DOJO\n" +
		"SLACK_BOT_TOKEN=xoxb-lab-token-must-not-leak\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, sub)

	if got := ResolveToken(); got != "xoxb-file-dojo" {
		t.Fatalf("ResolveToken file = %q, want xoxb-file-dojo", got)
	}
	if got := ResolveChannel(); got != "C_FILE_DOJO" {
		t.Fatalf("ResolveChannel file = %q, want C_FILE_DOJO", got)
	}
}

// --- folds ------------------------------------------------------------------

func TestRollupFromReportMeasuredRun(t *testing.T) {
	r := dojo.Report{
		Commit:       "abcdef1234567890",
		LeverCount:   1,
		EpisodeCount: 2,
		Measured:     2,
		Calibrated:   1,
		MeanCalibErr: 0.341,
		Grade:        "C",
		NextAction:   "inspect the cold-write regression before changing policy",
		Episodes: []dojo.Episode{
			{Lever: "resume-posture", Metric: "cold_write_share", Claimed: 0.85, Realized: 0.40, CalibErr: 0.53, Verdict: dojo.VerdictOverClaim, Grade: "D", Provenance: dojo.Observed, Sample: 40},
			{Lever: "resume-posture", Metric: "posture_accuracy", Claimed: 1.0, Realized: 0.98, CalibErr: 0.02, Verdict: dojo.VerdictCalibrated, Grade: "A", Provenance: dojo.Observed, Sample: 1000},
		},
	}
	got := RollupFromReport(r, 8).Text()

	// The lead carries the aggregate and the (truncated) commit.
	if !strings.Contains(got, "mean calib-err 0.341") || !strings.Contains(got, "grade C") {
		t.Fatalf("rollup lead missing aggregate:\n%s", got)
	}
	if !strings.Contains(got, "@abcdef123456") { // 12-char short commit
		t.Fatalf("rollup lead missing short commit:\n%s", got)
	}
	// Worst-first: the OVER_CLAIM cold_write_share (calib-err 0.53) must precede the
	// CALIBRATED posture_accuracy (0.02).
	cold := strings.Index(got, "cold_write_share")
	acc := strings.Index(got, "posture_accuracy")
	if cold < 0 || acc < 0 || cold > acc {
		t.Fatalf("episodes not worst-first (cold=%d acc=%d):\n%s", cold, acc, got)
	}
	// Provenance is carried through (conflation honesty).
	if !strings.Contains(got, "OVER_CLAIM") || !strings.Contains(got, "OBSERVED") {
		t.Fatalf("rollup dropped verdict/provenance:\n%s", got)
	}
	for _, want := range []string{
		"operator: inspect the cold-write regression",
		"current: 1 lever(s), 2 episode(s), 2 measured, 0 unmeasured, 1 calibrated",
		"worst lever: `resume-posture`",
		"worst metric `cold_write_share`",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rollup missing operator-friendly line %q:\n%s", want, got)
		}
	}
}

func TestRollupFromReportUnmeasuredSurfacesReason(t *testing.T) {
	r := dojo.Report{
		Grade:    "n/a",
		Measured: 0,
		Reason:   "dojo run incomplete — no episode had ground truth to score against",
	}
	got := RollupFromReport(r, 8).Text()
	if !strings.Contains(got, "no episode had ground truth") {
		t.Fatalf("unmeasured rollup must surface the reason:\n%s", got)
	}
}

func TestTrendFromLedgerOrdersRecentFirstAndTrends(t *testing.T) {
	rows := []dojo.LedgerRow{
		{Schema: dojo.LedgerSchema, Date: "2026-06-25", Commit: "c1", GeneratedAt: "2026-06-25T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.70, Grade: "F", Calibrated: 2, Measured: 6},
		{Schema: dojo.LedgerSchema, Date: "2026-06-26", Commit: "c2", GeneratedAt: "2026-06-26T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.50, Grade: "D", Calibrated: 2, Measured: 6},
		{Schema: dojo.LedgerSchema, Date: "2026-06-27", Commit: "c3", GeneratedAt: "2026-06-27T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.34, Grade: "C", Calibrated: 2, Measured: 6},
	}
	got := TrendFromLedger(rows, 3).Text()

	// The latest row's grade leads.
	if !strings.Contains(got, "grade C") {
		t.Fatalf("trend lead must carry the latest grade:\n%s", got)
	}
	// 0.50 -> 0.34 is an improvement; the summary must say so.
	if !strings.Contains(got, "improved") {
		t.Fatalf("trend must report the improvement direction:\n%s", got)
	}
	// Most-recent-first: 2026-06-27 precedes 2026-06-25 in the body.
	newest := strings.Index(got, "2026-06-27")
	oldest := strings.Index(got, "2026-06-25")
	if newest < 0 || oldest < 0 || newest > oldest {
		t.Fatalf("trend rows not most-recent-first (new=%d old=%d):\n%s", newest, oldest, got)
	}
	for _, want := range []string{
		"current: 3 lever(s), 7 episode(s), 6 measured, 1 unmeasured, 2 calibrated",
		"operator: claims moved closer to billed reality",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("trend missing operator-friendly line %q:\n%s", want, got)
		}
	}
}

func TestTrendFromLedgerEmptyIsHonest(t *testing.T) {
	got := TrendFromLedger(nil, 6).Text()
	if !strings.Contains(got, "no dojo history yet") {
		t.Fatalf("empty ledger must yield an honest card:\n%s", got)
	}
}

func TestRollupClampsMaxEpisodesAndPreservesWorstFirst(t *testing.T) {
	r := dojo.Report{
		Commit:       "abcdef1234567890",
		LeverCount:   3,
		EpisodeCount: 4,
		Measured:     4,
		Calibrated:   1,
		MeanCalibErr: 0.450,
		Grade:        "D",
		Episodes: []dojo.Episode{
			{Lever: "lever-a", Metric: "m1", Claimed: 0.9, Realized: 0.85, CalibErr: 0.05, Verdict: dojo.VerdictCalibrated, Grade: "A", Provenance: dojo.Witnessed, Sample: 100},
			{Lever: "lever-b", Metric: "m2", Claimed: 0.9, Realized: 0.20, CalibErr: 0.70, Verdict: dojo.VerdictOverClaim, Grade: "F", Provenance: dojo.Observed, Sample: 50},
			{Lever: "lever-c", Metric: "m3", Claimed: 0.8, Realized: 0.40, CalibErr: 0.40, Verdict: dojo.VerdictOverClaim, Grade: "D", Provenance: dojo.Observed, Sample: 80},
			{Lever: "lever-d", Metric: "m4", Claimed: 0.9, Realized: 0.60, CalibErr: 0.30, Verdict: dojo.VerdictUnderClaim, Grade: "C", Provenance: dojo.Witnessed, Sample: 60},
		},
	}

	post := RollupFromReport(r, 2)
	text := post.Text()

	// Should show worst two: m2 (0.70) and m3 (0.40).
	m2Idx := strings.Index(text, "m2")
	m3Idx := strings.Index(text, "m3")
	if m2Idx < 0 || m3Idx < 0 || m2Idx > m3Idx {
		t.Fatalf("expected worst-first order with m2 before m3:\n%s", text)
	}
	if strings.Contains(text, "m1") || strings.Contains(text, "m4") {
		t.Fatalf("expected clamped output to omit m1 and m4:\n%s", text)
	}
	if !strings.Contains(text, "…and 2 more episode(s) (worst-first; see `fak dojo run`)") {
		t.Fatalf("expected overflow summary line in text:\n%s", text)
	}
}

func TestPostFormattingDeterministic(t *testing.T) {
	r := dojo.Report{
		Commit:       "1234567890123456",
		LeverCount:   1,
		EpisodeCount: 2,
		Measured:     2,
		Calibrated:   1,
		MeanCalibErr: 0.150,
		Grade:        "B",
		Episodes: []dojo.Episode{
			{Lever: "l1", Metric: "m1", Claimed: 0.8, Realized: 0.7, CalibErr: 0.1, Verdict: dojo.VerdictCalibrated, Grade: "B", Provenance: dojo.Witnessed, Sample: 20},
			{Lever: "l2", Metric: "m2", Claimed: 0.9, Realized: 0.5, CalibErr: 0.4, Verdict: dojo.VerdictOverClaim, Grade: "D", Provenance: dojo.Observed, Sample: 30},
		},
	}

	p1 := RollupFromReport(r, 5)
	p2 := RollupFromReport(r, 5)

	if p1.Text() != p2.Text() {
		t.Fatalf("RollupFromReport non-deterministic Text: %q vs %q", p1.Text(), p2.Text())
	}
	if !reflect.DeepEqual(p1.Blocks(), p2.Blocks()) {
		t.Fatalf("RollupFromReport non-deterministic Blocks")
	}

	rows := []dojo.LedgerRow{
		{Schema: dojo.LedgerSchema, Date: "2026-06-25", Commit: "c1", GeneratedAt: "2026-06-25T01:00:00Z", LeverCount: 1, EpisodeCount: 1, MeanCalibErr: 0.50, Grade: "D", Calibrated: 0, Measured: 1},
		{Schema: dojo.LedgerSchema, Date: "2026-06-26", Commit: "c2", GeneratedAt: "2026-06-26T01:00:00Z", LeverCount: 1, EpisodeCount: 1, MeanCalibErr: 0.20, Grade: "B", Calibrated: 1, Measured: 1},
	}

	t1 := TrendFromLedger(rows, 2)
	t2 := TrendFromLedger(rows, 2)

	if t1.Text() != t2.Text() {
		t.Fatalf("TrendFromLedger non-deterministic Text: %q vs %q", t1.Text(), t2.Text())
	}
	if !reflect.DeepEqual(t1.Blocks(), t2.Blocks()) {
		t.Fatalf("TrendFromLedger non-deterministic Blocks")
	}
}

func TestCommentHygieneAndNoFormulaicNoise(t *testing.T) {
	fset := token.NewFileSet()
	files := []string{"dojopost.go", "render.go"}

	for _, filename := range files {
		content, err := os.ReadFile(filename)
		if err != nil {
			t.Fatalf("failed to read %s: %v", filename, err)
		}

		node, err := parser.ParseFile(fset, filename, content, parser.ParseComments)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", filename, err)
		}

		codeLines := countNonEmptyLines(content)
		commentLines := 0
		formulaicCount := 0
		hasFiller := false

		for _, cg := range node.Comments {
			for _, c := range cg.List {
				commentLines += strings.Count(c.Text, "\n") + 1
			}
			isForm, isFill := checkFormulaicComment(cg)
			if isForm {
				formulaicCount++
				t.Logf("%s: detected formulaic comment: %q", filename, strings.TrimSpace(cg.Text()))
			}
			if isFill {
				hasFiller = true
			}
		}

		commentRatio := float64(commentLines) / float64(codeLines)
		if codeLines > 30 && commentRatio > 0.35 {
			t.Errorf("%s: comment bloat ratio %.2f exceeds 0.35 (comments: %d, code: %d)",
				filename, commentRatio, commentLines, codeLines)
		}

		if formulaicCount > 0 || hasFiller {
			t.Errorf("%s: formulaic comments detected: count=%d, filler=%v",
				filename, formulaicCount, hasFiller)
		}

		// Verify all exported symbols have non-tautological documentation
		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(d.Name.Name) {
					if !isSubstantiveDoc(d.Name.Name, d.Doc) {
						t.Errorf("%s: exported func %s missing substantive doc", filename, d.Name.Name)
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if !isSubstantiveDoc(s.Name.Name, doc) {
								t.Errorf("%s: exported type %s missing substantive doc", filename, s.Name.Name)
							}
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								doc := s.Doc
								if doc == nil {
									doc = d.Doc
								}
								if !isSubstantiveDoc(name.Name, doc) {
									t.Errorf("%s: exported var/const %s missing substantive doc", filename, name.Name)
								}
							}
						}
					}
				}
			}
		}
	}
}

func countNonEmptyLines(b []byte) int {
	scanner := bufio.NewScanner(strings.NewReader(string(b)))
	n := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) != "" {
			n++
		}
	}
	return n
}

func checkFormulaicComment(cg *ast.CommentGroup) (bool, bool) {
	if cg == nil {
		return false, false
	}
	text := strings.TrimSpace(cg.Text())
	lower := strings.ToLower(text)

	hasMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.HasPrefix(lower, "invariant") ||
		strings.HasPrefix(lower, "guard") ||
		strings.HasPrefix(lower, "contract") ||
		strings.HasPrefix(lower, "fail-closed")

	if !hasMarker {
		return false, false
	}

	words := strings.Fields(lower)
	if len(words) <= 3 {
		return true, true
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.25 || keywordCount >= 3 {
		return true, true
	}

	return true, false
}

func isSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
	if doc == nil {
		return false
	}
	text := strings.TrimSpace(doc.Text())
	if text == "" {
		return false
	}
	words := strings.Fields(text)
	if len(words) < 3 {
		return false
	}
	firstWord := strings.Trim(strings.ToLower(words[0]), ":,.-()")
	nameLower := strings.ToLower(name)
	if firstWord == nameLower && len(words) <= 4 {
		// e.g. "Post is a post"
		fillers := map[string]bool{
			"is": true, "a": true, "the": true, "an": true, "for": true, "of": true,
		}
		meaningful := 0
		for _, w := range words[1:] {
			wl := strings.ToLower(strings.Trim(w, ":,.-()"))
			if !fillers[wl] && wl != nameLower {
				meaningful++
			}
		}
		if meaningful < 2 {
			return false
		}
	}
	return true
}

// --- helpers ----------------------------------------------------------------

func chdir(t *testing.T, dir string) {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(prev) })
}

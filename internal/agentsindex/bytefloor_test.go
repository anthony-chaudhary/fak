package agentsindex

import (
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// updateAgentsMDFloor makes TestAgentsMDByteFloorAtHEAD the one sanctioned writer of
// FloorBaselineFile (RegenerateFloorCommand). It re-pins from the live AGENTS.md; the
// same commit must carry the AGENTS.md change that justifies the new number.
var updateAgentsMDFloor = flag.Bool("update-agents-md-floor", false,
	"re-pin "+FloorBaselineFile+" from the live AGENTS.md")

// Two regrown paragraphs, one per section: the "two findings" of the ratchet tests.
const (
	alphaFinding = "Alpha finding: a rule that could have lived one hop away but now rides every turn-1 read.\n"
	betaFinding  = "Beta finding: another paragraph every agent pays for before it has done any work at all.\n"
	thirdFinding = "Third finding: the next reasonable-looking paragraph, the way the file regrew 2.4x.\n"
)

// floorDoc is an AGENTS.md-shaped source: a preamble, two sections, and a fenced shell
// comment at column 0 that must never be mistaken for a heading.
func floorDoc(alphaExtra, betaExtra string) string {
	return "# AGENTS.md\n\nLede line.\n\n" +
		"## Alpha rules (enforced)\n\nAlpha body.\n" + alphaExtra +
		"\n```bash\n# not a heading\nfak commit --path x\n```\n" +
		"\n## Beta: build\n\nBeta body.\n" + betaExtra
}

// measure is MeasureFloor for fixtures that must measure cleanly.
func measure(t *testing.T, src string) []FloorFinding {
	t.Helper()
	fs, err := MeasureFloor([]byte(src))
	if err != nil {
		t.Fatalf("MeasureFloor: %v", err)
	}
	return fs
}

// pinFloor regenerates a baseline for src and round-trips it through the strict parser,
// exactly as RegenerateFloorCommand followed by the gate would.
func pinFloor(t *testing.T, src string) FloorBaseline {
	t.Helper()
	base, err := ParseFloorBaseline(strings.NewReader(FormatFloorBaseline(measure(t, src))))
	if err != nil {
		t.Fatalf("a regenerated baseline does not parse: %v", err)
	}
	return base
}

func wantFloorErr(t *testing.T, err error, reason string) *FloorError {
	t.Helper()
	var fe *FloorError
	if !errors.As(err, &fe) {
		t.Fatalf("want *FloorError %s, got %T: %v", reason, err, err)
	}
	if fe.Reason != reason {
		t.Fatalf("Reason = %q, want %q (%v)", fe.Reason, reason, fe)
	}
	return fe
}

func findingCounts(t *testing.T, src string) map[string]int {
	t.Helper()
	out := map[string]int{}
	for _, f := range measure(t, src) {
		out[f.Key()] = f.Count
	}
	return out
}

// TestAgentsMDByteFloorAtHEAD is the enforcement half: the real AGENTS.md must sit
// inside its committed floor. This is the test that reds a commit that grows AGENTS.md
// without re-pinning, or trims it past the slack without banking the win.
func TestAgentsMDByteFloorAtHEAD(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	root, ok := FindRoot(wd)
	if !ok {
		t.Skipf("no dos.toml above %s; skipping the live AGENTS.md byte floor", wd)
	}
	raw, err := os.ReadFile(filepath.Join(root, FileName))
	if err != nil {
		t.Fatalf("read %s: %v (the floor fails closed on a missing doc)", FileName, err)
	}
	measured, err := MeasureFloor(raw)
	if err != nil {
		t.Fatal(err)
	}
	basePath := filepath.Join(root, filepath.FromSlash(FloorBaselineFile))
	if *updateAgentsMDFloor {
		if err := os.WriteFile(basePath, []byte(FormatFloorBaseline(measured)), 0o644); err != nil {
			t.Fatalf("re-pin %s: %v", FloorBaselineFile, err)
		}
		t.Logf("re-pinned %s at %d bytes; update the figure in docs/context-budget/agents-md-floor.md", FloorBaselineFile, measured[0].Count)
	}
	f, err := os.Open(basePath)
	if err != nil {
		t.Fatalf("open committed floor: %v", err)
	}
	defer f.Close()
	base, err := ParseFloorBaseline(f)
	if err != nil {
		t.Fatalf("committed floor %s is corrupt: %v", FloorBaselineFile, err)
	}
	if len(measured) < 6 {
		t.Fatalf("real AGENTS.md measured into %d findings; a broken parse must not pass as a small file", len(measured))
	}
	t.Logf("%s: %d bytes, floor %d, slack %d, %d sections plus the preamble",
		FileName, measured[0].Count, base.Ceiling(), FloorSlackBytes, len(measured)-2)
	if err := CheckFloor(measured, base); err != nil {
		t.Fatal(err)
	}
}

// TestAgentsMDFloorGrowthPastFloorFails witnesses the refusal and its message: the byte
// count, the overage, the section that regrew, and the command that re-pins. A bare
// "too big" is a gate nobody acts on.
func TestAgentsMDFloorGrowthPastFloorFails(t *testing.T) {
	base := pinFloor(t, floorDoc(alphaFinding, betaFinding))
	grown := floorDoc(alphaFinding, betaFinding+thirdFinding)

	fe := wantFloorErr(t, CheckFloor(measure(t, grown), base), ReasonFloorExceeded)
	if fe.Measured != len(grown) || fe.Measured-fe.Ceiling != len(thirdFinding) {
		t.Fatalf("Measured=%d Ceiling=%d; want %d over by %d", fe.Measured, fe.Ceiling, len(grown), len(thirdFinding))
	}
	if len(fe.Regrown) != 1 || fe.Regrown[0].Slug != "beta" || fe.Regrown[0].Growth() != len(thirdFinding) {
		t.Fatalf("Regrown = %+v; want exactly beta at +%d", fe.Regrown, len(thirdFinding))
	}
	msg := fe.Error()
	for _, want := range []string{
		"is " + strconv.Itoa(len(grown)) + " bytes",
		strconv.Itoa(len(thirdFinding)) + " bytes over",
		"floor of " + strconv.Itoa(fe.Ceiling),
		"Beta: build",
		"+" + strconv.Itoa(len(thirdFinding)) + " B",
		RegenerateFloorCommand,
		"one hop away",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("growth refusal does not carry %q:\n%s", want, msg)
		}
	}
}

// TestAgentsMDFloorAtFloorPasses pins the band: the floor itself admits, one byte over
// refuses, exactly slack below admits, one byte past the slack refuses as STALE.
func TestAgentsMDFloorAtFloorPasses(t *testing.T) {
	const pad, slack = 400, 100
	at := func(n int) []FloorFinding {
		return measure(t, floorDoc(strings.Repeat("x", n)+"\n", betaFinding))
	}
	base := pinFloor(t, floorDoc(strings.Repeat("x", pad)+"\n", betaFinding))
	for _, tc := range []struct {
		name string
		n    int
		want string // "" admits
	}{
		{"at the floor admits", pad, ""},
		{"one byte over refuses", pad + 1, ReasonFloorExceeded},
		{"exactly slack below admits", pad - slack, ""},
		{"one byte past the slack refuses", pad - slack - 1, ReasonFloorStale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkFloorAgainst(at(tc.n), base, slack)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("want admit, got %v", err)
				}
				return
			}
			wantFloorErr(t, err, tc.want)
		})
	}
}

// TestAgentsMDFloorFixingOneOfTwoTightens is contract clause (b). Two findings are
// pinned. Fixing one past the slack reds as STALE until the win is banked; banking it
// lowers the floor by exactly the fixed bytes; and re-adding the fixed paragraph, which
// the OLD floor admitted, is now refused. A third finding over the original two-finding
// floor is refused even though its key was already in the baseline: the baseline counts
// bytes, it is not an allowlist of paths.
func TestAgentsMDFloorFixingOneOfTwoTightens(t *testing.T) {
	two := floorDoc(alphaFinding, betaFinding)
	base := pinFloor(t, two)
	fixed := floorDoc(alphaFinding, "")
	slack := len(betaFinding) - 1 // the fix is larger than the slack, so it must be banked

	wantFloorErr(t, checkFloorAgainst(measure(t, fixed), base, slack), ReasonFloorStale)

	tight := pinFloor(t, fixed)
	if got, want := tight.Ceiling(), base.Ceiling()-len(betaFinding); got != want {
		t.Fatalf("banked floor = %d, want %d (tightened by exactly the fixed bytes)", got, want)
	}
	betaKey, alphaKey := KindSectionBytes+"\t"+FileName+"#beta", KindSectionBytes+"\t"+FileName+"#alpha-rules"
	if tight[betaKey] != base[betaKey]-len(betaFinding) || tight[alphaKey] != base[alphaKey] {
		t.Fatalf("banked sections: alpha %d->%d, beta %d->%d; only beta should tighten",
			base[alphaKey], tight[alphaKey], base[betaKey], tight[betaKey])
	}

	if err := checkFloorAgainst(measure(t, two), base, slack); err != nil {
		t.Fatalf("the original doc must still pass its own floor: %v", err)
	}
	fe := wantFloorErr(t, checkFloorAgainst(measure(t, two), tight, slack), ReasonFloorExceeded)
	if len(fe.Regrown) == 0 || fe.Regrown[0].Slug != "beta" {
		t.Fatalf("re-growth attributed to %+v; want beta", fe.Regrown)
	}

	three := floorDoc(alphaFinding+thirdFinding, betaFinding)
	fe = wantFloorErr(t, checkFloorAgainst(measure(t, three), base, slack), ReasonFloorExceeded)
	if len(fe.Regrown) != 1 || fe.Regrown[0].Slug != "alpha-rules" || fe.Regrown[0].Growth() != len(thirdFinding) {
		t.Fatalf("third finding attributed to %+v; want alpha-rules at +%d", fe.Regrown, len(thirdFinding))
	}
}

// TestParseFloorBaselineRejectsCorruptRow is contract clause (c): every malformed row is
// a hard error naming its line, never a skip. A skipped DOC_BYTES row would read as no
// ceiling; a skipped SECTION_BYTES row would misattribute someone else's refusal.
func TestParseFloorBaselineRejectsCorruptRow(t *testing.T) {
	good := FormatFloorBaseline(measure(t, floorDoc(alphaFinding, betaFinding)))
	if _, err := ParseFloorBaseline(strings.NewReader(good)); err != nil {
		t.Fatalf("control: a regenerated baseline must parse: %v", err)
	}
	lines := strings.Split(strings.TrimSuffix(good, "\n"), "\n")
	docRow, secRow := -1, -1
	for i, l := range lines {
		if strings.HasPrefix(l, KindDocBytes+"\t") {
			docRow = i
		}
		if secRow < 0 && strings.HasPrefix(l, KindSectionBytes+"\t") {
			secRow = i
		}
	}
	if docRow < 0 || secRow < 0 {
		t.Fatalf("control baseline lacks rows:\n%s", good)
	}
	replace := func(idx int, row string) (string, int) {
		cp := append([]string(nil), lines...)
		cp[idx] = row
		return strings.Join(cp, "\n") + "\n", idx + 1
	}
	for _, tc := range []struct {
		name string
		idx  int
		row  string
		want string
	}{
		{"two fields", secRow, KindSectionBytes + "\t" + FileName + "#alpha-rules", "want 3 tab-separated fields"},
		{"four fields", secRow, KindSectionBytes + "\t" + FileName + "#alpha-rules\t10\textra", "want 3 tab-separated fields"},
		{"spaces not tabs", docRow, KindDocBytes + " " + FileName + " 10", "want 3 tab-separated fields"},
		{"non-integer count", docRow, KindDocBytes + "\t" + FileName + "\t52,301", "not a non-negative integer"},
		{"negative count", secRow, KindSectionBytes + "\t" + FileName + "#alpha-rules\t-5", "not a non-negative integer"},
		{"unknown kind", secRow, "LINE_BYTES\t" + FileName + "#alpha-rules\t10", "not a floor finding kind"},
		{"wrong doc path", docRow, KindDocBytes + "\tCLAUDE.md\t10", "path must be"},
		{"empty section slug", secRow, KindSectionBytes + "\t" + FileName + "#\t10", "path must be"},
		{"section of another doc", secRow, KindSectionBytes + "\tCLAUDE.md#alpha-rules\t10", "path must be"},
		{"unproducible slug", secRow, KindSectionBytes + "\t" + FileName + "#Not A Slug!\t10", "not a heading slug"},
		{"implausible count", secRow, KindSectionBytes + "\t" + FileName + "#alpha-rules\t9223372036854775807", "exceeds"},
		{"duplicate key", secRow, lines[docRow], "duplicate key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src, lineNo := replace(tc.idx, tc.row)
			base, err := ParseFloorBaseline(strings.NewReader(src))
			if err == nil {
				t.Fatalf("corrupt row %q parsed (as %v); it must be a hard error", tc.row, base)
			}
			if base != nil {
				t.Errorf("a failed parse returned a partial baseline %v", base)
			}
			if !strings.Contains(err.Error(), "line "+strconv.Itoa(lineNo)+":") {
				t.Errorf("error does not name line %d: %v", lineNo, err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not say %q", err, tc.want)
			}
		})
	}

	t.Run("hand-edited ceiling breaks the partition", func(t *testing.T) {
		src, lineNo := replace(docRow, KindDocBytes+"\t"+FileName+"\t999999")
		_, err := ParseFloorBaseline(strings.NewReader(src))
		if err == nil || !strings.Contains(err.Error(), "line "+strconv.Itoa(lineNo)+":") || !strings.Contains(err.Error(), "partition") {
			t.Fatalf("a hand-raised DOC_BYTES must be refused naming line %d, got %v", lineNo, err)
		}
	})
	t.Run("no ceiling row", func(t *testing.T) {
		cp := append(append([]string(nil), lines[:docRow]...), lines[docRow+1:]...)
		_, err := ParseFloorBaseline(strings.NewReader(strings.Join(cp, "\n") + "\n"))
		if err == nil || !strings.Contains(err.Error(), "without a ceiling") {
			t.Fatalf("a baseline with no DOC_BYTES row must be refused, got %v", err)
		}
	})
}

// TestAgentsMDFloorKeysStableUnderLineInsertion is contract clause (a): inserting a line
// shifts every later heading's line number but renumbers no key. Only the section that
// received the line changes count, by exactly the inserted bytes, and the refusal names
// that section.
func TestAgentsMDFloorKeysStableUnderLineInsertion(t *testing.T) {
	src := floorDoc(alphaFinding, betaFinding)
	before := findingCounts(t, src)
	const inserted = "An inserted line near the top of Alpha.\n"
	after := strings.Replace(src, "Alpha body.\n", "Alpha body.\n"+inserted, 1)
	got := findingCounts(t, after)

	if len(got) != len(before) {
		t.Fatalf("key set changed size %d -> %d after a line insertion", len(before), len(got))
	}
	for key, n := range before {
		m, ok := got[key]
		if !ok {
			t.Fatalf("key %q vanished after a line insertion (renumbered?)", key)
		}
		want := n
		switch key {
		case KindDocBytes + "\t" + FileName, KindSectionBytes + "\t" + FileName + "#alpha-rules":
			want = n + len(inserted)
		}
		if m != want {
			t.Errorf("%q: %d -> %d, want %d", key, n, m, want)
		}
	}

	var betaBefore, betaAfter int
	for _, f := range measure(t, src) {
		if f.Path == FileName+"#beta" {
			betaBefore = f.Line
		}
	}
	for _, f := range measure(t, after) {
		if f.Path == FileName+"#beta" {
			betaAfter = f.Line
		}
	}
	if betaAfter != betaBefore+1 {
		t.Fatalf("beta heading line %d -> %d; the fixture must actually shift it", betaBefore, betaAfter)
	}

	fe := wantFloorErr(t, CheckFloor(measure(t, after), pinFloor(t, src)), ReasonFloorExceeded)
	if len(fe.Regrown) != 1 || fe.Regrown[0].Slug != "alpha-rules" {
		t.Fatalf("insertion attributed to %+v; want only alpha-rules", fe.Regrown)
	}
}

// TestAgentsMDFloorPartitionsAndNormalizesEOL proves the counted findings are a faithful
// partition (preamble + every section's own bytes == DOC_BYTES), that a fenced `# ...`
// line is not a section, and that a CRLF checkout measures the same bytes as the
// committed LF blob.
func TestAgentsMDFloorPartitionsAndNormalizesEOL(t *testing.T) {
	src := floorDoc(alphaFinding, betaFinding)
	fs := measure(t, src)
	if len(fs) != 4 { // DOC_BYTES, preamble, alpha-rules, beta
		t.Fatalf("measured %d findings, want 4 (a fenced comment became a heading?): %+v", len(fs), fs)
	}
	sum := 0
	for _, f := range fs[1:] {
		sum += f.Count
	}
	if fs[0].Count != len(src) || sum != len(src) {
		t.Fatalf("DOC_BYTES=%d, sections sum=%d, file=%d; the findings must partition the file", fs[0].Count, sum, len(src))
	}
	crlf := findingCounts(t, strings.ReplaceAll(src, "\n", "\r\n"))
	for key, n := range findingCounts(t, src) {
		if crlf[key] != n {
			t.Errorf("%q: LF %d vs CRLF %d; the floor must count committed-blob bytes", key, n, crlf[key])
		}
	}
}

// TestAgentsMDFloorFailsClosed: an empty doc measures 0 bytes and refuses as STALE
// rather than greening on "I measured nothing", even against a floor small enough that
// 0 bytes would sit inside the slack band.
func TestAgentsMDFloorFailsClosed(t *testing.T) {
	base := pinFloor(t, "# AGENTS.md\n\n## Tiny\n\nx\n")
	if base.Ceiling() > FloorSlackBytes {
		t.Fatalf("fixture floor %d must sit inside the %d-byte slack for this test to mean anything", base.Ceiling(), FloorSlackBytes)
	}
	empty, err := MeasureFloor(nil)
	if err != nil {
		t.Fatal(err)
	}
	wantFloorErr(t, CheckFloor(empty, base), ReasonFloorStale)
}

// TestAgentsMDFloorDocPinsTheCeiling keeps the reviewable page honest: the doc a
// reviewer reads must carry the same ceiling the baseline gates, the regeneration
// command, and both refusal tokens.
func TestAgentsMDFloorDocPinsTheCeiling(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	root, ok := FindRoot(wd)
	if !ok {
		t.Skipf("no dos.toml above %s", wd)
	}
	f, err := os.Open(filepath.Join(root, filepath.FromSlash(FloorBaselineFile)))
	if err != nil {
		t.Fatalf("open committed floor: %v", err)
	}
	defer f.Close()
	base, err := ParseFloorBaseline(f)
	if err != nil {
		t.Fatalf("committed floor is corrupt: %v", err)
	}
	doc, err := os.ReadFile(filepath.Join(root, "docs", "context-budget", "agents-md-floor.md"))
	if err != nil {
		t.Fatalf("read the floor page: %v", err)
	}
	for _, want := range []string{
		strconv.Itoa(base.Ceiling()),
		RegenerateFloorCommand,
		filepath.Base(FloorBaselineFile), // the page sits beside the baseline and links it relatively
		ReasonFloorExceeded,
		ReasonFloorStale,
	} {
		if !strings.Contains(string(doc), want) {
			t.Errorf("docs/context-budget/agents-md-floor.md does not carry %q; the page drifted from the gate", want)
		}
	}
}

// TestAgentsMDFloorCheckFloorUsesFloorSlackBytes binds the public gate to the committed
// slack. The band tests inject a slack, so without this a CheckFloor that ignored
// FloorSlackBytes (and so never demanded a banked win) would stay green.
func TestAgentsMDFloorCheckFloorUsesFloorSlackBytes(t *testing.T) {
	const pad = FloorSlackBytes + 200
	at := func(n int) []FloorFinding {
		return measure(t, floorDoc(strings.Repeat("x", n)+"\n", betaFinding))
	}
	base := pinFloor(t, floorDoc(strings.Repeat("x", pad)+"\n", betaFinding))
	if err := CheckFloor(at(pad-FloorSlackBytes), base); err != nil {
		t.Fatalf("a trim of exactly FloorSlackBytes must admit: %v", err)
	}
	fe := wantFloorErr(t, CheckFloor(at(pad-FloorSlackBytes-1), base), ReasonFloorStale)
	if fe.Slack != FloorSlackBytes {
		t.Fatalf("Slack = %d, want FloorSlackBytes = %d", fe.Slack, FloorSlackBytes)
	}
	for _, want := range []string{
		"bytes, " + strconv.Itoa(FloorSlackBytes+1) + " below",
		"floor of " + strconv.Itoa(base.Ceiling()),
		RegenerateFloorCommand,
	} {
		if !strings.Contains(fe.Error(), want) {
			t.Errorf("stale refusal does not carry %q:\n%s", want, fe.Error())
		}
	}
}

// TestAgentsMDFloorAttributionRanksAndCaps pins how a refusal names sections: largest
// growth first, a heading absent from the baseline marked new, and at most
// maxRegrownShown rows with the remainder counted rather than dropped silently.
func TestAgentsMDFloorAttributionRanksAndCaps(t *testing.T) {
	base := pinFloor(t, floorDoc("", ""))
	fe := wantFloorErr(t, CheckFloor(measure(t, floorDoc(alphaFinding, betaFinding+thirdFinding)), base), ReasonFloorExceeded)
	if len(fe.Regrown) != 2 || fe.Regrown[0].Slug != "beta" || fe.Regrown[1].Slug != "alpha-rules" {
		t.Fatalf("Regrown = %+v; want beta (larger growth) before alpha-rules", fe.Regrown)
	}

	base = pinFloor(t, floorDoc(alphaFinding, betaFinding))
	fe = wantFloorErr(t, CheckFloor(measure(t, floorDoc(alphaFinding, betaFinding)+"## Gamma\n\n"+thirdFinding), base), ReasonFloorExceeded)
	if len(fe.Regrown) != 1 || fe.Regrown[0].Slug != "gamma" || !fe.Regrown[0].New || fe.Regrown[0].Pinned != 0 {
		t.Fatalf("Regrown = %+v; want gamma marked New", fe.Regrown)
	}
	if !strings.Contains(fe.Error(), `new heading "gamma"`) {
		t.Errorf("refusal does not flag the new heading:\n%s", fe.Error())
	}

	var doc, grown strings.Builder
	doc.WriteString("# AGENTS.md\n")
	grown.WriteString("# AGENTS.md\n")
	for i := 0; i < maxRegrownShown+2; i++ {
		head := "\n## Section " + string(rune('a'+i)) + "\n\n"
		doc.WriteString(head)
		grown.WriteString(head + "one more line\n")
	}
	fe = wantFloorErr(t, CheckFloor(measure(t, grown.String()), pinFloor(t, doc.String())), ReasonFloorExceeded)
	if len(fe.Regrown) != maxRegrownShown+2 {
		t.Fatalf("Regrown has %d sections, want %d", len(fe.Regrown), maxRegrownShown+2)
	}
	msg := fe.Error()
	if rows := strings.Count(msg, " B  L"); rows != maxRegrownShown {
		t.Errorf("refusal shows %d section rows, want %d:\n%s", rows, maxRegrownShown, msg)
	}
	if !strings.Contains(msg, "... and 2 more grown section(s)") {
		t.Errorf("refusal drops the remainder silently:\n%s", msg)
	}
}

// TestAgentsMDFloorRefusesSharedHeadingSlug: Parse numbers repeated slugs by position,
// so two headings that slug alike would shift keys when a third is added above them and
// can even collide (Step, Step 2, Step (again) all reach step-2), which would make the
// regenerate command write a baseline its own parser refuses. MeasureFloor refuses the
// pair instead, naming both lines.
func TestAgentsMDFloorRefusesSharedHeadingSlug(t *testing.T) {
	for _, src := range []string{
		"# AGENTS.md\n\n## Notes\n\na\n\n## Notes\n\nb\n",
		"# AGENTS.md\n\n## Step\n\na\n\n## Step 2\n\nb\n\n## Step (again)\n\nc\n",
	} {
		_, err := MeasureFloor([]byte(src))
		if err == nil {
			t.Fatalf("headings sharing a slug measured cleanly:\n%s", src)
		}
		if !strings.Contains(err.Error(), "line 3") || !strings.Contains(err.Error(), "both slug to") {
			t.Errorf("error does not name the first heading's line and the shared slug: %v", err)
		}
	}
}

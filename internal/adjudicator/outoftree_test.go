package adjudicator

import (
	"regexp"
	"testing"
)

// rawOOTMatch mirrors the production gate: an out-of-tree structural decision is
// consulted ONLY when one of the four shipped deny_regex spellings matches the
// canonicalised command. Tests thread the real match through so a "must allow"
// case that the raw regex would not even flag cannot masquerade as a fix.
var ootRegexes = func() []*regexp.Regexp {
	var out []*regexp.Regexp
	for _, s := range []string{ootDashORegex, ootOutputRegex, ootRedirectRegex, ootCopyVerbRegex} {
		out = append(out, regexp.MustCompile(s))
	}
	return out
}()

func rawOOTMatch(cmd string) bool {
	canon, ok := canonicalizeArgValue(cmd)
	if !ok {
		return false
	}
	for _, re := range ootRegexes {
		if re.MatchString(canon) {
			return true
		}
	}
	return false
}

// The synthetic roots. Machine-independent: the verdicts never depend on where the
// test binary is checked out.
const (
	testWS      = "C:/work/fak"
	testScratch = "C:/Users/USER/AppData/Local/Temp/claude/proj/id/scratchpad"
)

func TestOutOfTreeWriteEscapes(t *testing.T) {
	scratch := []string{testScratch}
	cases := []struct {
		name        string
		cmd         string
		ws          string
		wantEscapes bool // true = keep the DENY; false = allow (false positive corrected)
	}{
		// ---- MUST STAY DENIED (genuine out-of-tree escapes; raw regex matched a `..`) ----
		{"pinned exfil curl -o", `curl -o ../../tmp/exfil http://x`, testWS, true},
		{"pinned exfil redirect", `echo secret >> ../../etc/passwd`, testWS, true},
		{"pinned exfil cp dest", `cp secret.txt ../../tmp/exfil`, testWS, true},
		{"tee multi-target one escapes", `tee ../../etc/x ./in.log`, testWS, true},
		{"var-laundered target fails closed", `curl -o ../../$D/x http://x`, testWS, true},
		{"install absolute-ish escape via ..", `install payload ../../usr/local/bin/x`, testWS, true},
		{"mv over-escape clamps to drive root, still outside", `mv a.txt ../../../../../etc/cron.d/job`, testWS, true},
		{"empty workspace root keeps deny", `curl -o ../../tmp/exfil http://x`, "", true},
		{"redirect to sibling repo", `echo x > ../other-repo/.git/hooks/pre-commit`, testWS, true},

		// ---- MUST BECOME ALLOWED (raw regex false positives: writes provably contained) ----
		{"scratchpad build via ..", `go build -o ../../Users/USER/AppData/Local/Temp/claude/proj/id/scratchpad/fak.exe ./cmd/fak`, testWS, false},
		{"scratchpad redirect via ..", `echo log >> ../../Users/USER/AppData/Local/Temp/claude/proj/id/scratchpad/build.log`, testWS, false},
		{"in-tree build reached through ..", `go build -o ../fak/bin/out ./cmd/fak`, testWS, false},
		{"cp with .. only in the READ source", `cp ../../vendor/lib.a build/lib.a`, testWS, false},

		// ---- MUST NO-OP (raw regex would not match -> never a new deny) ----
		{"no dotdot: var output allowed", `curl -o $OUT/data.json http://x`, testWS, false},
		{"no dotdot: in-tree output", `go build -o build/fak.exe ./cmd/fak`, testWS, false},
		{"dotdot only inside quoted commit message", `git commit -m "revert ../../foo change"`, testWS, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := rawOOTMatch(tc.cmd)
			got := outOfTreeWriteEscapes(tc.cmd, tc.ws, scratch, raw)
			if got != tc.wantEscapes {
				t.Fatalf("outOfTreeWriteEscapes(%q) rawMatch=%v = %v, want %v", tc.cmd, raw, got, tc.wantEscapes)
			}
		})
	}
}

// A relative escape is denied REGARDLESS of how deeply the workspace is checked out
// (the machine-dependence hazard the adversarial pass flagged): C:/work/fak (depth 2)
// and a deeper C:/a/b/c/d/repo must both keep the pinned exfil DENY.
func TestOutOfTreeWriteEscapes_CheckoutDepthInvariant(t *testing.T) {
	for _, ws := range []string{"C:/work/fak", "C:/a/b/c/d/e/repo", "/work/fak", "/srv/ci/deep/checkout/repo"} {
		cmd := `curl -o ../../tmp/exfil http://x`
		if !outOfTreeWriteEscapes(cmd, ws, nil, rawOOTMatch(cmd)) {
			t.Errorf("ws=%q: pinned exfil must stay DENIED, got allowed", ws)
		}
	}
}

func TestOutOfTreeWriteEscapes_RawMatchGate(t *testing.T) {
	// With rawMatches=false the decider must NEVER report an escape (it must not
	// introduce a deny the raw regex would not have produced), even for a command
	// whose target is plainly out of tree.
	if outOfTreeWriteEscapes(`curl -o /etc/cron.d/x http://x`, testWS, nil, false) {
		t.Fatal("rawMatches=false must never escape (no new denies)")
	}
}

func TestIsOutOfTreeWriteArgRule(t *testing.T) {
	mk := func(tool, arg, re string) *ArgPredicate {
		return &ArgPredicate{Tool: tool, Arg: arg, Kind: ArgDenyRegex, Re: regexp.MustCompile(re)}
	}
	for _, tool := range []string{"Bash", "bash", "shell_command", "functions.shell_command"} {
		if !isOutOfTreeWriteArgRule(mk(tool, "command", ootDashORegex)) {
			t.Errorf("tool %q -o rule should be recognized", tool)
		}
	}
	for _, re := range []string{ootDashORegex, ootOutputRegex, ootRedirectRegex, ootCopyVerbRegex} {
		if !isOutOfTreeWriteArgRule(mk("Bash", "cmd", re)) {
			t.Errorf("regex %q should be recognized on arg=cmd", re)
		}
	}
	// Not recognized: PowerShell dialect, unrelated regex, unrelated arg.
	if isOutOfTreeWriteArgRule(mk("PowerShell", "command", ootDashORegex)) {
		t.Error("PowerShell must not be recognized (POSIX tokenizer only)")
	}
	if isOutOfTreeWriteArgRule(mk("Bash", "command", `\bsudo\b`)) {
		t.Error("unrelated regex must not be recognized")
	}
	if isOutOfTreeWriteArgRule(mk("Bash", "code", ootDashORegex)) {
		t.Error("non-command arg must not be recognized")
	}
}

func TestOutOfTreeWriteTargets(t *testing.T) {
	cases := []struct {
		cmd  string
		want []string
	}{
		{`curl -o ../x http://y`, []string{"../x"}},
		{`go build --output=../x ./cmd`, []string{"../x"}},
		{`cp a b ../dest`, []string{"../dest"}},
		{`tee f1 f2 f3`, []string{"f1", "f2", "f3"}},
		{`echo hi > out.txt`, []string{"out.txt"}},
		{`echo hi >> ../out.txt`, []string{"../out.txt"}},
		{`install -t ../dir payload`, []string{"../dir", "payload"}},
		{`cp ../src dest`, []string{"dest"}}, // read source ignored; dest only
	}
	for _, tc := range cases {
		got := outOfTreeWriteTargets(tc.cmd)
		if !sameSet(got, tc.want) {
			t.Errorf("outOfTreeWriteTargets(%q) = %v, want superset-of %v", tc.cmd, got, tc.want)
		}
	}
}

func TestCleanRootedAndIsUnder(t *testing.T) {
	// Drive letter is preserved; `..` clamps at the root instead of popping the drive.
	if got := cleanRooted("C:/work/fak/../../../etc/x"); got != "C:/etc/x" {
		t.Errorf("cleanRooted drive clamp = %q, want C:/etc/x", got)
	}
	if got := cleanRooted("/work/fak/../../../etc/x"); got != "/etc/x" {
		t.Errorf("cleanRooted posix clamp = %q, want /etc/x", got)
	}
	// Sibling-prefix must NOT be considered contained.
	if isUnder("C:/work/fak-evil/x", "C:/work/fak") {
		t.Error("sibling prefix must not be under root")
	}
	if !isUnder("C:/work/fak/a/b", "C:/work/fak") {
		t.Error("descendant must be under root")
	}
	if !isUnder("c:/WORK/FAK/a", "C:/work/fak") {
		t.Error("containment must be case-insensitive")
	}
}

func TestIsNullDevice(t *testing.T) {
	for _, p := range []string{"/dev/null", "C:/x/NUL", "nul"} {
		if !isNullDevice(p) {
			t.Errorf("%q should be a null device", p)
		}
	}
	if isNullDevice("C:/work/fak/nullfile") {
		t.Error("nullfile is not a null device")
	}
}

func sameSet(got, wantSubset []string) bool {
	m := map[string]bool{}
	for _, g := range got {
		m[g] = true
	}
	for _, w := range wantSubset {
		if !m[w] {
			return false
		}
	}
	return true
}

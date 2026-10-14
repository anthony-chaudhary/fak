package architest

import (
	"go/version"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// goToolchainSecurityFloors are the minimum patched toolchain releases that carry the
// reachable standard-library fixes security-audit reported on 2026-10-09 (12 findings
// under go1.26.6, fixed in go1.26.9). A toolchain on the 1.26 line must be at least
// go1.26.9; once Go 1.27 exists, every subsequent stable line must additionally be at
// least go1.27.2. The two floors are named so a reader can see which release line each
// binds without re-deriving them from the audit transcript.
const (
	go126SecurityFloor = "go1.26.9"
	go127SecurityFloor = "go1.27.2"
)

// goToolchainMeetsSecurityFloor reports whether a go.mod `toolchain` directive names a
// release that carries the security fixes above. It fails closed on anything that cannot
// be shown to reach a floor: a malformed directive, a prerelease, or a language-only
// version ("go1.26", which pins no patch and so cannot carry a specific fix) is refused.
//
// go/version is used rather than a numeric-prefix scan because its parser is strict: it
// rejects malformed suffixes such as "go1.26.9foo" that fmt.Sscanf would silently accept
// as the prefix go1.26.9, and it orders prereleases correctly ("go1.27rc1" < "go1.27.2").
func goToolchainMeetsSecurityFloor(directive string) bool {
	if !version.IsValid(directive) {
		return false
	}
	// A language-only directive ("go1.26") does not pin a patch release, so it cannot be
	// shown to carry a specific fix even though it sorts below its own patch releases.
	if version.Lang(directive) == directive {
		return false
	}
	if version.Compare(directive, go126SecurityFloor) < 0 {
		return false
	}
	// On the 1.26 line the go1.26.9 floor is the binding one; any later release line must
	// also clear go1.27.2, so go1.27.0 and go1.27.1 (which predate the 1.27 security fixes)
	// are refused rather than admitted by their numeric prefix.
	if version.Compare(directive, "go1.27") >= 0 && version.Compare(directive, go127SecurityFloor) < 0 {
		return false
	}
	return true
}

func TestPinnedToolchainIncludesGo126SecurityFixes(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	data, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}

	var directive string
	for line := range strings.Lines(string(data)) {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "toolchain" {
			directive = fields[1]
			break
		}
	}
	if directive == "" {
		t.Fatal("go.mod has no toolchain directive; security-audit must scan the toolchain the repository ships")
	}
	if !goToolchainMeetsSecurityFloor(directive) {
		t.Fatalf("go.mod toolchain %s does not carry the security fixes (need >= %s on the 1.26 line, >= %s on later stable lines)", directive, go126SecurityFloor, go127SecurityFloor)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestToolchainIncludesGoSecurityFixes(t *testing.T) {
	for _, tc := range []struct {
		directive string
		want      bool
		why       string
	}{
		{go126SecurityFloor, true, "the 1.26 floor itself"},
		{"go1.26.10", true, "a patched 1.26 release above the floor"},
		{"go1.26.99", true, "any later 1.26 patch"},
		{go127SecurityFloor, true, "the 1.27 floor itself"},
		{"go1.27.3", true, "a patched 1.27 release above the floor"},
		{"go1.28.0", true, "a later stable line clears go1.27.2 trivially"},
		{"go1.26.9-custom", true, "a well-formed custom suffix still reaches the floor"},
		{"go2.0.0", true, "a later major clears both floors"},
		{"go1.26.8", false, "one patch below the 1.26 floor"},
		{"go1.25.20", false, "an older release line entirely"},
		{"go1.27.0", false, "a 1.27 release predating the 1.27 fixes"},
		{"go1.27.1", false, "a 1.27 release predating the 1.27 fixes"},
		{"go1.27rc1", false, "a prerelease is not a stable release"},
		{"go1.26", false, "a language-only directive pins no patch"},
		{"go1.26.9foo", false, "a malformed suffix a numeric-prefix scan would admit"},
		{"", false, "an absent or empty directive"},
	} {
		if got := goToolchainMeetsSecurityFloor(tc.directive); got != tc.want {
			t.Errorf("goToolchainMeetsSecurityFloor(%q) = %v, want %v (%s)", tc.directive, got, tc.want, tc.why)
		}
	}
}

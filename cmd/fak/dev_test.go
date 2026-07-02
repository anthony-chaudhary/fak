package main

import (
	"strings"
	"testing"
)

// The C2 (#2231, epic #2228) namespace contract for `fak dev`. resolveDevVerb is
// the whole decision surface (main() only rewrites os.Args on exit == -1), so
// these tests pin the contract without process spawns:
//
//   - a dev-tier verb (any alias spelling) dispatches with its args intact;
//   - a frontdoor verb refuses — one spelling per frontdoor verb;
//   - a hidden seam is NOT advertised: it takes the unknown path, no suggestion;
//   - an unknown token gets a tier-aware did-you-mean;
//   - bare / -h / --help print the dev-tier listing (dev verbs only).

func resolveDev(t *testing.T, argv ...string) (string, []string, int, string, string) {
	t.Helper()
	var out, errb strings.Builder
	v, rest, code := resolveDevVerb(argv, &out, &errb)
	return v, rest, code, out.String(), errb.String()
}

func TestResolveDevVerbDispatchesDevTier(t *testing.T) {
	v, rest, code, _, _ := resolveDev(t, "sweep", "--dir", "x")
	if code != -1 || v != "sweep" {
		t.Fatalf("resolveDevVerb(sweep) = (%q, %v, %d), want dispatch of sweep", v, rest, code)
	}
	if len(rest) != 2 || rest[0] != "--dir" || rest[1] != "x" {
		t.Fatalf("args not passed through intact: %v", rest)
	}
	// An alias spelling of a dev verb dispatches too — under its own token, which
	// the dispatch switch routes identically.
	if v, _, code, _, _ := resolveDev(t, "benchloop"); code != -1 || v != "benchloop" {
		t.Errorf("resolveDevVerb(benchloop) = (%q, %d), want dispatch", v, code)
	}
	// Case-insensitive like the rest of the verb machinery.
	if v, _, code, _, _ := resolveDev(t, "SWEEP"); code != -1 || v != "sweep" {
		t.Errorf("resolveDevVerb(SWEEP) = (%q, %d), want lowercased dispatch", v, code)
	}
}

func TestResolveDevVerbRefusesFrontdoor(t *testing.T) {
	_, _, code, _, errs := resolveDev(t, "guard")
	if code != 2 {
		t.Fatalf("resolveDevVerb(guard) exit = %d, want 2", code)
	}
	if !strings.Contains(errs, "'fak guard'") || !strings.Contains(errs, "frontdoor") {
		t.Errorf("frontdoor refusal must name the one true spelling; got: %s", errs)
	}
}

func TestResolveDevVerbHidesHiddenSeams(t *testing.T) {
	_, _, code, _, errs := resolveDev(t, "guard-stophook")
	if code != 2 {
		t.Fatalf("resolveDevVerb(guard-stophook) exit = %d, want 2", code)
	}
	if !strings.Contains(errs, "unknown verb") {
		t.Errorf("a hidden seam must take the unknown path, got: %s", errs)
	}
	if strings.Contains(errs, "did you mean") {
		t.Errorf("a hidden seam must not be suggested back: %s", errs)
	}
}

func TestResolveDevVerbSuggestsTierAware(t *testing.T) {
	_, _, code, _, errs := resolveDev(t, "swep")
	if code != 2 {
		t.Fatalf("resolveDevVerb(swep) exit = %d, want 2", code)
	}
	if !strings.Contains(errs, "did you mean 'fak dev sweep'?") {
		t.Errorf("dev-tier near-miss must suggest the dev spelling; got: %s", errs)
	}
	// A near-miss of a FRONTDOOR verb points at the bare spelling.
	_, _, _, _, errs = resolveDev(t, "guardd")
	if !strings.Contains(errs, "did you mean 'fak guard'?") {
		t.Errorf("frontdoor near-miss must suggest the bare spelling; got: %s", errs)
	}
}

func TestDevListingIsDevTierOnly(t *testing.T) {
	_, _, code, out, _ := resolveDev(t)
	if code != 0 {
		t.Fatalf("bare `fak dev` exit = %d, want 0 (the listing)", code)
	}
	for _, want := range []string{"  sweep\t", "  commit\t", "  scorecard\t"} {
		if !strings.Contains(out, strings.ReplaceAll(want, "\t", "")) {
			t.Errorf("dev listing missing dev verb %q", strings.TrimSpace(want))
		}
	}
	for _, ln := range strings.Split(out, "\n") {
		f := strings.Fields(ln)
		if len(f) > 0 && (f[0] == "guard" || f[0] == "serve" || f[0] == "guard-stophook") {
			t.Errorf("dev listing leaked a non-dev verb line: %q", ln)
		}
	}
	if !strings.Contains(out, "fak dev <verb> [args...]") {
		t.Errorf("listing must show the usage line; got head: %.120s", out)
	}
	// --help prints the same listing.
	_, _, code, out2, _ := resolveDev(t, "--help")
	if code != 0 || out2 != out {
		t.Errorf("`fak dev --help` must print the same listing (exit %d)", code)
	}
}

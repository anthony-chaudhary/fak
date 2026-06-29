package compute

// Offline, GPU-free preflight for the flat C ABI header cuda_backend.h (#17).
//
// cuda_backend.h once broke the paid fak-cuda GPU build: it used uint8_t WITHOUT
// #include <stdint.h>. A lax toolchain (WSL's micromamba nvcc) pulled the type in
// transitively via <stddef.h>, so it built locally; a strict datacenter toolchain
// (the GCP L4's nvcc) does not, so it failed with "identifier uint8_t is undefined" —
// and that failure was only discovered AFTER provisioning and paying for a GPU VM.
//
// These tests are the durable guard that catches that class — a type used without the
// header that defines it — before a VM is ever launched. They are deterministic, need
// no GPU, no nvcc, and no network, run in the default `go test ./internal/compute/`
// (so they sit in `make ci` / the pre-push gate for free), and additionally run the
// issue's exact `gcc -fsyntax-only` standalone parse as a superset wherever a host C
// compiler exists (CI's hosted Ubuntu, WSL). The pure-Go check is the portable witness
// that runs even where no C compiler is installed (e.g. a Windows dev box).

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

const cudaHeaderFile = "cuda_backend.h"

// typeDefiningHeaders maps a C fixed-width / stddef type to the set of #includes, any ONE
// of which defines it. A header that USES the type in code but #includes none of them is
// the exact #17 portability bug: it compiles only on a toolchain that happens to pull the
// definition in transitively.
var typeDefiningHeaders = map[string][]string{
	"int8_t":    {"stdint.h", "cstdint", "inttypes.h"},
	"uint8_t":   {"stdint.h", "cstdint", "inttypes.h"},
	"int16_t":   {"stdint.h", "cstdint", "inttypes.h"},
	"uint16_t":  {"stdint.h", "cstdint", "inttypes.h"},
	"int32_t":   {"stdint.h", "cstdint", "inttypes.h"},
	"uint32_t":  {"stdint.h", "cstdint", "inttypes.h"},
	"int64_t":   {"stdint.h", "cstdint", "inttypes.h"},
	"uint64_t":  {"stdint.h", "cstdint", "inttypes.h"},
	"intptr_t":  {"stdint.h", "cstdint", "inttypes.h"},
	"uintptr_t": {"stdint.h", "cstdint", "inttypes.h"},
	"size_t":    {"stddef.h", "cstddef", "stdlib.h", "stdint.h"},
	"ptrdiff_t": {"stddef.h", "cstddef"},
}

var (
	blockCommentRe  = regexp.MustCompile(`(?s)/\*.*?\*/`)
	lineCommentRe   = regexp.MustCompile(`//[^\n]*`)
	includeRe       = regexp.MustCompile(`(?m)^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]`)
	stdintIncludeRe = regexp.MustCompile(`#[ \t]*include[ \t]*<[ \t]*stdint\.h[ \t]*>`)
)

// stripCComments blanks /* ... */ and // ... comments so a type token that appears only in
// prose (cuda_backend.h's own header comment literally mentions "uint8_t") is never mistaken
// for a use. Replaced with a space to preserve token boundaries.
func stripCComments(src string) string {
	src = blockCommentRe.ReplaceAllString(src, " ")
	src = lineCommentRe.ReplaceAllString(src, " ")
	return src
}

// includedHeaders is the set of header basenames the source #includes (e.g. "stdint.h").
func includedHeaders(src string) map[string]bool {
	got := map[string]bool{}
	for _, m := range includeRe.FindAllStringSubmatch(src, -1) {
		got[filepath.Base(m[1])] = true
	}
	return got
}

// preflightHeaderTypes returns, sorted, one "<type> needs #include <hdr>" defect for every
// fixed-width / stddef type used in CODE whose defining header is not #included. An empty
// result means the header parses standalone for this class. This is the pure-Go, toolchain-
// free reproduction of the #17 bug class.
func preflightHeaderTypes(src string) []string {
	includes := includedHeaders(src)
	code := stripCComments(src)
	types := make([]string, 0, len(typeDefiningHeaders))
	for t := range typeDefiningHeaders {
		types = append(types, t)
	}
	sort.Strings(types)

	var defects []string
	for _, typ := range types {
		if !regexp.MustCompile(`\b` + regexp.QuoteMeta(typ) + `\b`).MatchString(code) {
			continue // type not used in code
		}
		satisfied := false
		for _, h := range typeDefiningHeaders[typ] {
			if includes[h] {
				satisfied = true
				break
			}
		}
		if !satisfied {
			defects = append(defects, typ+" needs #include <"+typeDefiningHeaders[typ][0]+">")
		}
	}
	return defects
}

// findHostCC returns the first available host C compiler, or "" if none is installed.
func findHostCC() string {
	if cc := os.Getenv("CC"); cc != "" {
		if p, err := exec.LookPath(cc); err == nil {
			return p
		}
	}
	for _, cand := range []string{"cc", "gcc", "clang"} {
		if p, err := exec.LookPath(cand); err == nil {
			return p
		}
	}
	return ""
}

// strictStandaloneParse runs the issue's exact technique: compile the header ALONE under a
// strict toolchain (gcc -x c -std=c11 -fsyntax-only -Wall -Werror, mirroring build_cuda.sh's
// `check`). Returns nil if it parses clean, the compiler's diagnostics otherwise. Returns
// (false, "") when no compiler is installed so the caller can skip rather than fail.
func strictStandaloneParse(t *testing.T, src string) (ran bool, diag string) {
	t.Helper()
	cc := findHostCC()
	if cc == "" {
		return false, ""
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "standalone_cuda_backend.h")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatalf("write temp header: %v", err)
	}
	out, err := exec.Command(cc, "-x", "c", "-std=c11", "-fsyntax-only", "-Wall", "-Werror", path).CombinedOutput()
	if err != nil {
		return true, string(out)
	}
	return true, ""
}

func readCUDAHeader(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(cudaHeaderFile)
	if err != nil {
		t.Fatalf("read %s (test runs in the package dir): %v", cudaHeaderFile, err)
	}
	return string(b)
}

// TestCUDAHeaderPreflightFixedHeaderPasses: the current (fixed) cuda_backend.h must pass the
// standalone type-include preflight — and, where a C compiler exists, the strict gcc parse.
func TestCUDAHeaderPreflightFixedHeaderPasses(t *testing.T) {
	src := readCUDAHeader(t)

	if defects := preflightHeaderTypes(src); len(defects) != 0 {
		t.Fatalf("current %s must parse standalone but the preflight flagged: %v\n"+
			"(a fixed-width/stddef type is used without #including its defining header — the #17 class)",
			cudaHeaderFile, defects)
	}

	if ran, diag := strictStandaloneParse(t, src); ran {
		if diag != "" {
			t.Fatalf("strict standalone parse of %s failed under the host C compiler:\n%s", cudaHeaderFile, diag)
		}
	} else {
		t.Logf("strict gcc -fsyntax-only superset skipped: no host C compiler on PATH (pure-Go preflight still ran)")
	}
}

// TestCUDAHeaderPreflightMissingIncludeFails: a cuda_backend.h with <stdint.h> removed — the
// exact regression that broke the paid GPU build — must be caught. This is the deterministic
// fails-on-bad half of the "Done when": derived from the real header so it can never drift
// out of sync with the types the header actually uses.
func TestCUDAHeaderPreflightMissingIncludeFails(t *testing.T) {
	src := readCUDAHeader(t)
	if !stdintIncludeRe.MatchString(src) {
		t.Fatalf("%s no longer #includes <stdint.h> — the negative fixture cannot be derived; "+
			"update this test if the header's include set changed", cudaHeaderFile)
	}
	bad := stdintIncludeRe.ReplaceAllString(src, "/* <stdint.h> removed: #17 negative fixture */")

	defects := preflightHeaderTypes(bad)
	if len(defects) == 0 {
		t.Fatalf("a %s missing <stdint.h> must be flagged by the preflight, but it reported none "+
			"(uint8_t/int8_t are used by the q4k/q8/AWQ prototypes and need <stdint.h>)", cudaHeaderFile)
	}
	joined := strings.Join(defects, "; ")
	if !strings.Contains(joined, "uint8_t") && !strings.Contains(joined, "int8_t") {
		t.Fatalf("preflight must name the now-undefined fixed-width type, got: %v", defects)
	}

	if ran, diag := strictStandaloneParse(t, bad); ran {
		if diag == "" {
			t.Fatalf("strict standalone parse should FAIL on a %s missing <stdint.h>, but the host "+
				"C compiler accepted it (it may be pulling the type in transitively — the very lax "+
				"behavior that masked #17)", cudaHeaderFile)
		}
	} else {
		t.Logf("strict gcc -fsyntax-only superset skipped: no host C compiler on PATH (pure-Go preflight still caught it)")
	}
}

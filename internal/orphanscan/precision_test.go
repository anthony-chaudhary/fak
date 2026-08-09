package orphanscan

// precision_test.go — Arm 2 of #3169: the four known precision/recall edge classes,
// fixtured so the boundary is PINNED rather than assumed.
//
// The scan's whole claim is precision over recall (see the package doc): it would rather
// miss a real orphan than cry wolf. That claim was asserted but never tested at its
// edges. #3169 enumerated four paths where the syntactic pass could plausibly be wrong —
// free funcs adapted to an interface, string-keyed/reflection dispatch, build-tag
// interactions, and generated-code references — and none of them had a fixture. An
// unfixtured limit is a rumour: nobody can tell a deliberate accepted trade-off from a
// bug nobody noticed, and the next person to touch the reference counter has no way to
// know which behaviours are load-bearing.
//
// Each case below is labelled with the verdict it pins:
//
//	clean       — the class does NOT misbehave; the identifier is visible after all, so
//	              the worry itself was wrong. Pinned so a future refactor cannot quietly
//	              turn it into a real false positive.
//	accepted-FP — the scan DOES flag something reachable. Accepted because the remedy is
//	              the visible //orphanscan:keep escape hatch, and because narrowing the
//	              scan to avoid it would cost true positives.
//	accepted-FN — the scan does NOT flag something that is dead on any single build. The
//	              deliberate cost of parsing every file regardless of //go:build.
//
// Every fixture also carries a `controlOrphan` func: a plain, unambiguous true positive
// sitting in the same package as the edge case. Asserting it is STILL flagged is the
// precision-first invariant from #3169's third Arm-2 checkbox — no exclusion may
// silently swallow a true positive alongside the case it was written for.

import (
	"reflect"
	"testing"
)

// controlOrphanSrc is the true-positive control every precision fixture carries. It is a
// plain unexported func with no call site anywhere, so it must be flagged in EVERY case
// below; if a case's want list ever loses it, an exclusion has grown too wide.
const controlOrphanSrc = `
// controlOrphan is a deliberate, unambiguous orphan. It must be flagged in every case.
func controlOrphan() {}
`

func TestPrecisionRecallClasses(t *testing.T) {
	cases := []struct {
		name    string
		verdict string            // clean | accepted-FP | accepted-FN
		files   map[string]string // filename -> source (controlOrphanSrc is appended to the first-listed anchor)
		anchor  string            // the file the control orphan is appended to
		want    []string          // exact flagged names, sorted
		why     string            // the documented reason this verdict is the intended one
	}{
		{
			// CLASS 1 — free func satisfying an interface through a func-typed adapter.
			// #3169 suspected a false positive here. It is not one: converting the func to
			// the adapter type (`HandlerFunc(handleThing)`) is itself an identifier
			// reference, so the syntactic pass sees the wiring. The worry only holds for a
			// func adapted from ANOTHER package — and an unexported name cannot be.
			name:    "free-func-satisfies-interface",
			verdict: "clean",
			anchor:  "adapter.go",
			files: map[string]string{
				"adapter.go": `package fixture

type Handler interface{ Serve() }

// HandlerFunc adapts a receiver-less func to Handler, the http.HandlerFunc shape.
type HandlerFunc func()

func (h HandlerFunc) Serve() { h() }

// handleThing is receiver-less and is never CALLED — it is only ever converted to the
// adapter type. That conversion is an identifier reference, so it is not an orphan.
func handleThing() {}

// Route is exported API and is the only thing that mentions handleThing.
func Route() Handler { return HandlerFunc(handleThing) }
`,
			},
			want: []string{"controlOrphan"},
			why:  "the adapter conversion HandlerFunc(handleThing) is an identifier reference, so interface satisfaction via a func-typed adapter is visible to the syntactic pass",
		},
		{
			// CLASS 2 — string-keyed / reflection dispatch. This one IS a real false
			// positive, and it is the class the //orphanscan:keep hatch exists for: when
			// the only occurrence of the name is inside a string literal, no identifier
			// reference exists to count.
			name:    "string-keyed-dispatch-is-flagged",
			verdict: "accepted-FP",
			anchor:  "registry.go",
			files: map[string]string{
				"registry.go": `package fixture

// Dispatch resolves an op through a string-keyed table an external generator owns, so
// the op's identifier never appears at a call site.
func Dispatch(op string) { _ = op }

// Boot names the op the only way this shape can: as a string.
func Boot() { Dispatch("stringKeyedOp") }

// stringKeyedOp is reachable at run time and invisible to a syntactic scan.
func stringKeyedOp() {}
`,
			},
			want: []string{"controlOrphan", "stringKeyedOp"},
			why:  "a name that appears only inside a string literal has no identifier reference to count; ACCEPTED because the remedy is the visible //orphanscan:keep hatch, and inferring string-literal wiring would cost true positives",
		},
		{
			// CLASS 2b — the same shape with the escape hatch applied. This is the half
			// that makes the accepted-FP acceptable: the remedy actually works, and it
			// does not leak onto the neighbouring control orphan.
			name:    "string-keyed-dispatch-keep-hatch-suppresses",
			verdict: "clean",
			anchor:  "registry.go",
			files: map[string]string{
				"registry.go": `package fixture

func Dispatch(op string) { _ = op }

func Boot() { Dispatch("stringKeyedOp") }

//orphanscan:keep dispatched by string key from a generated table
func stringKeyedOp() {}
`,
			},
			want: []string{"controlOrphan"},
			why:  "//orphanscan:keep suppresses exactly the func it annotates and nothing else, so the accepted false positive has a working, greppable remedy",
		},
		{
			// CLASS 3 — build-tag interactions. go/parser does not evaluate //go:build at
			// all, so a def and its only use under DISJOINT constraints still pool into
			// one reference set. On any single platform the func is dead; the scan does
			// not say so. Accepted false negative, in the precision-first direction.
			name:    "build-tag-disjoint-def-and-use",
			verdict: "accepted-FN",
			anchor:  "unix.go",
			files: map[string]string{
				"unix.go": `//go:build linux

package fixture

// linuxOnlyHelper is defined only on linux.
func linuxOnlyHelper() int { return 1 }
`,
				"windows.go": `//go:build windows

package fixture

// UseOnWindows is the ONLY reference to linuxOnlyHelper and can never be compiled
// together with it. The parser sees both files anyway.
func UseOnWindows() int { return linuxOnlyHelper() }
`,
			},
			want: []string{"controlOrphan"},
			why:  "parser.ParseFile ignores //go:build, so cross-tag references count and a func dead on every individual platform is not flagged; ACCEPTED because evaluating constraints would require a per-platform scan and can only ADD false positives",
		},
		{
			// CLASS 3b — the same mechanism seen from its useful side: a `//go:build
			// ignore` standalone tool in the package dir still contributes references, so
			// a helper used only by that tool is not flagged.
			name:    "build-ignore-file-reference-counts",
			verdict: "accepted-FN",
			anchor:  "lib.go",
			files: map[string]string{
				"lib.go": `package fixture

// usedByIgnoredTool is referenced only from a //go:build ignore file.
func usedByIgnoredTool() int { return 2 }
`,
				"gen.go": `//go:build ignore

package main

func main() { _ = usedByIgnoredTool() }
`,
			},
			want: []string{"controlOrphan"},
			why:  "an ignore-tagged file is never built with the package, but its identifiers still count as references; ACCEPTED in the same direction as the disjoint-tag case",
		},
		{
			// CLASS 4 — funcs referenced only from generated code. Within the package the
			// generated file's identifiers ARE counted (they are pooled into refs before
			// the generated-file `continue`), so a func wired only by a generator is not
			// flagged. Correct, and pinned.
			name:    "generated-file-reference-counts",
			verdict: "clean",
			anchor:  "hand.go",
			files: map[string]string{
				"hand.go": `package fixture

// wiredByGenerator has no hand-written call site; the generated table below is its
// only reference.
func wiredByGenerator() int { return 3 }
`,
				"zz_generated_table.go": `// Code generated by fakgen. DO NOT EDIT.

package fixture

var Table = map[string]func() int{"wired": wiredByGenerator}
`,
			},
			want: []string{"controlOrphan"},
			why:  "a generated file contributes its identifiers to the reference set before it is skipped as a definition source, so same-package generated wiring is seen",
		},
		{
			// CLASS 4b — the cross-package boundary #3169 asked to fixture. ScanDir is
			// package-local, so generated code in a DIFFERENT package is invisible. That
			// can only matter for a name the other package could actually write, i.e. an
			// EXPORTED one — and exported names are never candidates. The boundary is
			// therefore structurally safe, not merely untested. Pinned here so the
			// argument is checkable rather than asserted.
			name:    "cross-package-generated-reference-needs-exported-name",
			verdict: "clean",
			anchor:  "api.go",
			files: map[string]string{
				"api.go": `package fixture

// ExportedForOtherPackages is what a generator in another package would have to call.
// It is exported, so it is never a candidate no matter what references it.
func ExportedForOtherPackages() int { return 4 }

// unexportedNeverCrossesPackages cannot be named from another package at all, so no
// cross-package generated reference can exist for it. It is flagged on its own merits.
func unexportedNeverCrossesPackages() int { return 5 }
`,
			},
			want: []string{"controlOrphan", "unexportedNeverCrossesPackages"},
			why:  "the package-local scan cannot see another package's generated code, but only an EXPORTED name is reachable from there and exported names are already excluded; an unexported name has no cross-package reference to miss",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, src := range tc.files {
				if name == tc.anchor {
					src += controlOrphanSrc
				}
				writeGo(t, dir, name, src)
			}
			got := scanNames(t, dir)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("[%s] orphans = %v, want %v\n  reason this is the intended verdict: %s",
					tc.verdict, got, tc.want, tc.why)
			}
			// The precision-first invariant (#3169): whatever the class does, the plain
			// true positive sitting beside it is still reported. An exclusion that also
			// silences controlOrphan has grown wider than the case it was written for.
			var sawControl bool
			for _, n := range got {
				if n == "controlOrphan" {
					sawControl = true
				}
			}
			if !sawControl {
				t.Fatalf("[%s] control true positive was swallowed by this class; precision-first invariant broken", tc.verdict)
			}
			t.Logf("%s [%s]: %s", tc.name, tc.verdict, tc.why)
		})
	}
}

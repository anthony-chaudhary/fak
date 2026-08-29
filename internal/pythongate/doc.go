// Package pythongate is the NEW-PYTHON-TOOL ratchet: the durable gate that makes the
// project's de-Python push gradual and self-sustaining.
//
// # The ratchet
//
// The tree carries hundreds of tracked tools/*.py helpers. Rewriting them all to Go at
// once is neither safe nor necessary; what matters is that the Python count only ever
// goes DOWN. So this gate does not ban Python — it bans NEW Python. It compares the
// current tracked tools/*.py set (via `git ls-files tools/*.py`) against a frozen
// allowlist (baseline.go, the grandfathered set captured the day the ratchet shipped)
// and refuses any path that is not grandfathered. A new tool must therefore be written
// in Go, following the house pattern (a new internal/<name>/ package plus a cmd/fak/
// shell); a stray tools/*.py reds the trunk via TestNoNewPythonTools.
//
// A Python ratchet that banned Python would be self-defeating, so the gate itself is Go.
//
// # The gradual de-Python policy
//
//   - Existing tracked tools/*.py are GRANDFATHERED: allowed because they predate the
//     ratchet. They are not retroactively broken.
//   - A NEW tools/*.py is refused with the reason NEW_PYTHON_TOOL. Port the logic to Go.
//   - The baseline only ever SHRINKS. When a grandfathered .py is legitimately ported to
//     Go and DELETED, it leaves the tracked set and its baseline entry is removed too.
//     That is the ratchet tightening: the allowlist can never grow, so the Python surface
//     monotonically decreases over time.
//
// # The one narrow widening case: a shared module, not a new tool
//
// The allowlist gains a row only for a file that is not a new TOOL at all: a shared
// module extracted OUT of already-grandfathered tools/*.py callers, whose behavior
// already exists in Go. tools/fleet_regdir.py is the shape — the Python twin of
// internal/accountprobe/regdir.go, imported in-process by six grandfathered fleet
// modules that each carried a divergent copy of the same registry-dir fallback. No new
// capability escaped into Python: six copies of existing Python logic became one, and
// the Go reader remains the authority the module mirrors. All three preconditions hold,
// and the commit body has to argue them:
//
//   - every consumer is itself grandfathered, and the import is in-process — a Go-only
//     resolver would mean a process spawn per registry read, not a port;
//   - the behavior already has a Go implementation, so the module is a mirror of that
//     authority rather than a second one;
//   - the rows are APPENDED to the slice literal by hand, never swept in by a full
//     regeneration (which stays reserved for tightening), so the diff is exactly the
//     widen being argued for and nothing else.
//
// Anything else — a new operator-facing script, or a capability with no Go home — is
// refused: write it in Go. Even the sanctioned widen is temporary; when the callers are
// ported the module leaves with them and the recipe below drops the rows.

// # The narrower test-companion contract

// Historical gate policy admitted tests of grandfathered modules because a test adds no
// operator-facing Python capability. testcompanions.go makes that policy explicit without
// widening baseline.go: every reviewed row names its exact sibling module and introducing
// commit, and the live gate admits it only while the sibling is tracked, grandfathered, and
// imported in-process by the test. There is deliberately no wildcard for *_test.py; a new
// test still reds until its provenance is reviewed. A Python test for shell, Go, or a new
// Python capability does not qualify and must be ported or removed.
//
// # Regenerating the baseline
//
// You regenerate baseline.go only to TIGHTEN it after a port-and-delete — never to
// re-admit a new tool. After porting tools/foo.py to Go and `git rm`-ing it, refreeze the
// (now smaller) allowlist from the tracked set:
//
//	{
//	  printf '// Code generated from `git ls-files tools/*.py`. DO NOT EDIT by hand.\n'
//	  printf '// Regenerate with the recipe in doc.go when a grandfathered tool is PORTED-AND-DELETED.\n\n'
//	  printf 'package pythongate\n\n'
//	  printf 'var grandfathered = []string{\n'
//	  git ls-files 'tools/*.py' | sort | sed 's/.*/\t"&",/'
//	  printf '}\n'
//	} > internal/pythongate/baseline.go
//	gofmt -w internal/pythongate/baseline.go
//
// Because the recipe reads the live tracked set, the regenerated allowlist is always a
// subset of the previous one (the deleted file is gone) — the ratchet can only get
// stricter, never looser.
package pythongate

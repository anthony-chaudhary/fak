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

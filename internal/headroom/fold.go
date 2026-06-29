package headroom

import (
	"fmt"
	"strings"
)

// globalDupMin: a line must occur at least this many times across the WHOLE blob
// (not just consecutively) before global folding touches it. Three keeps the floor
// conservative — a line seen twice is left alone.
const globalDupMin = 3

// minFoldLen: only fold lines at least this many bytes long. Short structural lines
// (`}`, `ok`, `--`) are cheap and folding them hurts readability more than it saves;
// the win is in repeated substantive lines — warnings, errors, stack frames, paths.
const minFoldLen = 8

// foldGlobalDuplicates collapses lines that recur NON-CONSECUTIVELY across the blob:
// the dual of normalizeLines' consecutive run-collapse. Real test / lint / build
// output repeats the SAME message scattered among other lines (one warning per file,
// a stack frame echoed per failure) — waste the consecutive pass cannot see. It keeps
// the FIRST occurrence of each foldable line IN PLACE (order preserved), annotates it
// with how many more times it recurs, and elides the later occurrences. That is
// model-readable ("this line shows up N more times") and reversible — the gate pins
// the original in the shared CAS (the CCR promise).
//
// It returns the folded bytes and whether anything changed (false ⇒ caller keeps the
// input and claims no codec). Lossy-but-bounded, like the consecutive collapse: a
// folded view drops the INTERLEAVING of repeated lines, so it is a benign-result
// compression, not a structural transform of code under edit (the gate only ever runs
// it on results it already chose to compress, never on poison).
func foldGlobalDuplicates(b []byte) ([]byte, bool) {
	s := string(b)
	lines := strings.Split(s, "\n")

	// Pass 1: count occurrences of each foldable line.
	counts := make(map[string]int, len(lines))
	for _, ln := range lines {
		if foldableLine(ln) {
			counts[ln]++
		}
	}
	// Nothing recurs enough to fold? Bail before allocating the output.
	anyFoldable := false
	for _, c := range counts {
		if c >= globalDupMin {
			anyFoldable = true
			break
		}
	}
	if !anyFoldable {
		return b, false
	}

	// Pass 2: emit each line in order; keep the first occurrence of a folded line
	// (plus a recurrence marker), drop the rest.
	out := make([]string, 0, len(lines))
	emitted := make(map[string]bool, len(counts))
	changed := false
	for _, ln := range lines {
		if foldableLine(ln) && counts[ln] >= globalDupMin {
			if emitted[ln] {
				changed = true // a later occurrence: elide it
				continue
			}
			emitted[ln] = true
			out = append(out, ln)
			out = append(out, fmt.Sprintf("… (×%d more identical, elided) …", counts[ln]-1))
			continue
		}
		out = append(out, ln)
	}
	if !changed {
		return b, false
	}
	return []byte(strings.Join(out, "\n")), true
}

// foldableLine reports whether a line is a candidate for global folding: non-blank,
// long enough to be worth folding, and not itself an elision marker emitted by the
// consecutive-collapse pass (folding a marker would be nonsense).
func foldableLine(ln string) bool {
	if len(ln) < minFoldLen {
		return false
	}
	if strings.Contains(ln, "identical lines elided") || strings.Contains(ln, "more identical, elided") {
		return false
	}
	return true
}

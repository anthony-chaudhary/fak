package agent

// anthropic_elide_crossturn.go — cross-turn verbatim-span dedup, a distinct dedup LEVEL layered
// onto the same byte-splice request transform as head+tail elision (anthropic_elide.go).
//
// Head+tail elision shrinks a SINGLE oversized tool_result with no comparison to any other block.
// Cross-turn dedup is orthogonal: a coding agent re-displays the same file bytes many times per
// session (cat → sed → git diff → cat), and fak's whole-cell dedup elsewhere can neither catch a
// span that is a sub-region of a differently-bounded later block nor run on the tool_use stream at
// all. This pass folds a contiguous LINE span in a later tool_result that appeared verbatim in a
// STRICTLY-EARLIER tool_result down to a one-line pointer naming the earlier turn + line range.
//
// Two invariants make it cache-safe and correct by construction:
//
//   - Strictly-earlier-only + keep-earliest. A block is only ever matched against blocks that
//     precede it on the wire, and the pointer names the EARLIEST occurrence of the run (nothing
//     earlier contains it, so the earliest occurrence is itself never folded for that run and its
//     bytes stay verbatim). Because a block's rewrite depends ONLY on strictly-earlier blocks'
//     ORIGINAL content, appending a turn never mutates an earlier turn's folded bytes — the
//     prefix-monotonicity property dedup(blocks[:k]) == dedup(full)[:k] (TestCrossTurnDedup).
//   - Whole-value re-marshal. Like elideStringEdit, a fold rebuilds the ENTIRE tool_result text
//     VALUE in decoded line-space and emits one spliceEdit replacing the whole value's byte range.
//     This sidesteps JSON-escape/offset mapping entirely; the protected cache prefix is untouched
//     because only blocks strictly after it are ever rewritten (proven by verifySplicedBody).
//
// Residual (accepted, lossless): if a proper sub-run of a referenced run had an even-earlier
// occurrence, that sub-run is itself folded within the referenced block, relocating its bytes to
// their own earliest home (reachable via the pointer chain). No bytes are lost; line numbers name
// the referenced block's ORIGINAL numbering, which is frozen regardless of downstream folding.

import (
	"encoding/json"
	"fmt"
	"strings"
)

const (
	// crossTurnMinDupLines is the min-span floor in LINES: a run shorter than this is never folded,
	// so the one-line pointer can never cost more structure than it saves.
	crossTurnMinDupLines = 8
	// crossTurnMinDupBytes is the min-span floor in BYTES: the folded run must be at least this many
	// raw bytes, guaranteeing the ~80-byte pointer is a clear net win (a hard len() guard confirms
	// the actual saving per value before any edit is emitted).
	crossTurnMinDupBytes = 240
	// crossTurnMaxLocsPerLine caps how many earlier occurrences of a single line the index tracks, so
	// a degenerate body of identical lines cannot make matching quadratic. Earliest occurrences are
	// kept (they are the ones a pointer prefers to reference); excess later ones are dropped. This is
	// a best-effort bound — a missed fold is always safe, a wrong fold is not, and none can occur.
	crossTurnMaxLocsPerLine = 64
)

// crossTurnPointerf renders the frozen, ABSOLUTE in-band pointer that stands in for a folded run:
// it names the earlier turn and that turn's original 1-based line range, so the model (and a human
// reading the wire) can locate the verbatim bytes. trailingNL preserves whether the folded span
// ended the block with a newline.
func crossTurnPointerf(nLines, turn, lineFrom, lineTo int, trailingNL bool) string {
	s := fmt.Sprintf("…[fak dedup: %d lines identical to output shown earlier, turn %d, lines %d-%d]…", nLines, turn, lineFrom, lineTo)
	if trailingNL {
		s += "\n"
	}
	return s
}

// dedupLoc is one indexed occurrence of a line: its block sequence (into the ordered block list)
// and line index within that block's ORIGINAL lines.
type dedupLoc struct{ blk, line int }

// hashLine is FNV-1a over the line's bytes — a fast bucket key for candidate match starts. A hash
// collision is harmless: every candidate is confirmed by real string comparison before it can fold.
func hashLine(s string) uint64 {
	const offset64 = 14695981039346656037
	const prime64 = 1099511628211
	h := uint64(offset64)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime64
	}
	return h
}

// splitLinesKeepNL splits s into physical lines, each RETAINING its trailing "\n" (the final line
// keeps a trailing "\n" only if s ended with one). Concatenating the result reproduces s byte-for-
// byte, so a fold that replaces a contiguous line range and re-marshals the whole value is exact.
// Splitting on the single byte 0x0A is UTF-8-safe (a newline is never part of a multibyte rune).
func splitLinesKeepNL(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i+1])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

// spanBytes sums the byte lengths of a contiguous line range (newlines included).
func spanBytes(lines []string) int {
	n := 0
	for _, l := range lines {
		n += len(l)
	}
	return n
}

// dedupBlockLines is the pure, deterministic core of cross-turn dedup. Given ordered blocks (each a
// []string of lines) and their turn numbers, it returns each block's folded rendering and a per-
// block "changed" flag. Every block is matched ONLY against strictly-earlier blocks, and the index
// records each earlier block's ORIGINAL lines — so a block's output is a function of the blocks
// before it alone. That is exactly the prefix-monotonicity invariant the witness test asserts:
// dedupBlockLines(blocks[:k]) == dedupBlockLines(blocks)[:k] for every k.
//
// The caller decides which folded outputs to actually splice in (only rewrite-eligible blocks —
// never the protected prefix, a cache_control block, or the recent working set); non-eligible
// blocks are still indexed here as sources, which is why this function folds every block uniformly.
func dedupBlockLines(blocks [][]string, turns []int) (out []string, changed []bool) {
	out = make([]string, len(blocks))
	changed = make([]bool, len(blocks))
	lineIx := make(map[uint64][]dedupLoc)
	for bi := range blocks {
		cur := blocks[bi]
		var sb strings.Builder
		did := false
		a := 0
		for a < len(cur) {
			// Find the longest contiguous run starting at cur[a] that also appears contiguously in
			// some strictly-earlier block. Candidates are earlier occurrences of the first line;
			// iteration is earliest-first, so the first candidate achieving the maximal extension is
			// the earliest occurrence of the run (the block whose bytes are guaranteed verbatim).
			bestLen, bestBlk, bestLine := 0, -1, -1
			for _, loc := range lineIx[hashLine(cur[a])] {
				src := blocks[loc.blk]
				l := 0
				for a+l < len(cur) && loc.line+l < len(src) && cur[a+l] == src[loc.line+l] {
					l++
				}
				if l > bestLen {
					bestLen, bestBlk, bestLine = l, loc.blk, loc.line
				}
			}
			if bestBlk >= 0 && bestLen >= crossTurnMinDupLines && spanBytes(cur[a:a+bestLen]) >= crossTurnMinDupBytes {
				trailingNL := strings.HasSuffix(cur[a+bestLen-1], "\n")
				sb.WriteString(crossTurnPointerf(bestLen, turns[bestBlk], bestLine+1, bestLine+bestLen, trailingNL))
				a += bestLen
				did = true
			} else {
				sb.WriteString(cur[a])
				a++
			}
		}
		out[bi] = sb.String()
		changed[bi] = did
		// Only now index THIS block's original lines, so it can be a source for later blocks but was
		// itself matched only against earlier ones.
		for li, ln := range cur {
			h := hashLine(ln)
			if len(lineIx[h]) < crossTurnMaxLocsPerLine {
				lineIx[h] = append(lineIx[h], dedupLoc{blk: bi, line: li})
			}
		}
	}
	return out, changed
}

// toolTextSite is one tool_result text VALUE located in the request body: its owning message index,
// the absolute byte span of the JSON string value, the decoded lines, and whether the block carries
// its own cache_control (which bars rewriting it, though it may still be a source).
type toolTextSite struct {
	msgIndex   int
	valAbs     int
	valBytes   []byte
	lines      []string
	blockHasCC bool
}

// collectToolTextSites enumerates every tool_result text value across ALL user messages in wire
// order, decoding each to lines. It indexes protected / cache_control / recent-window values too,
// because they are valid SOURCES a later fold may point back to; the caller applies the rewrite-
// eligibility band separately. Value byte spans are located by KEY (objectValueSpan / arrayElement
// Spans) with the exact same offset arithmetic as collectResultElisionEdits, so a site's valAbs
// coincides with the head+tail path's edit start for the same value (used for overlap resolution).
func collectToolTextSites(raw []byte, elems []json.RawMessage, spans []elementSpan) []toolTextSite {
	var sites []toolTextSite
	for i, el := range elems {
		var m struct {
			Role string `json:"role"`
		}
		if json.Unmarshal(el, &m) != nil || m.Role != "user" {
			continue
		}
		forEachToolResultBlock(spans[i].start, el, func(blk json.RawMessage, blkBase int) {
			blockHasCC := rawHasCacheControl(blk) || toolResultContentHasCacheControl(blk)
			cStart, cEnd, ok := objectValueSpan(blk, "content")
			if !ok {
				return
			}
			cVal := blk[cStart:cEnd]
			addString := func(valAbs int, valBytes []byte) {
				var s string
				if len(valBytes) == 0 || valBytes[0] != '"' || json.Unmarshal(valBytes, &s) != nil {
					return
				}
				sites = append(sites, toolTextSite{
					msgIndex:   i,
					valAbs:     valAbs,
					valBytes:   valBytes,
					lines:      splitLinesKeepNL(s),
					blockHasCC: blockHasCC,
				})
			}
			switch {
			case len(cVal) > 0 && cVal[0] == '"': // bare-string content
				addString(blkBase+cStart, cVal)
			case len(cVal) > 0 && cVal[0] == '[': // array of blocks — index each text block
				inner, innerSpans, ok := arrayElementSpans(cVal)
				if !ok {
					return
				}
				for k, ib := range inner {
					var tb struct {
						Type string `json:"type"`
					}
					if json.Unmarshal(ib, &tb) != nil || tb.Type != "text" {
						continue
					}
					tStart, tEnd, ok := objectValueSpan(ib, "text")
					if !ok {
						continue
					}
					addString(blkBase+cStart+innerSpans[k].start+tStart, ib[tStart:tEnd])
				}
			}
		})
	}
	return sites
}

// collectCrossTurnDedupEdits builds the whole-value splice edits that fold cross-turn duplicate
// spans. It runs dedupBlockLines over every tool_result text value (all indexed as sources) and
// emits an edit only for a value that (a) is rewrite-eligible — strictly after the protected prefix
// [pfxEnd], before the recent window [lastEligible), and carrying no cache_control at message or
// block depth — and (b) actually got shorter after re-marshal. The returned edits are disjoint (one
// per distinct value) and every edit lies strictly after the protected prefix, so applying them
// keeps the cached head byte-identical.
func collectCrossTurnDedupEdits(raw []byte, elems []json.RawMessage, spans []elementSpan, pfxEnd, lastEligible int) (edits []spliceEdit, shed int) {
	sites := collectToolTextSites(raw, elems, spans)
	if len(sites) < 2 {
		return nil, 0
	}
	blocksLines := make([][]string, len(sites))
	turns := make([]int, len(sites))
	for si, s := range sites {
		blocksLines[si] = s.lines
		turns[si] = s.msgIndex
	}
	folded, changed := dedupBlockLines(blocksLines, turns)
	for si, s := range sites {
		if !changed[si] {
			continue
		}
		// Rewrite-eligibility: identical band to head+tail elision. Sources may be anywhere earlier,
		// but a fold is only spliced into an old, un-cached, non-resident block.
		if !(pfxEnd < s.msgIndex && s.msgIndex < lastEligible) {
			continue
		}
		if s.blockHasCC || messageHasCacheControlForElide(elems[s.msgIndex]) {
			continue
		}
		newVal, err := json.Marshal(folded[si])
		if err != nil || len(newVal) >= len(s.valBytes) {
			continue // no net win — leave the value untouched
		}
		edits = append(edits, spliceEdit{start: s.valAbs, end: s.valAbs + len(s.valBytes), repl: newVal})
		shed += len(s.valBytes) - len(newVal)
	}
	return edits, shed
}

// spliceEditOverlapsAny reports whether e's [start,end) byte range intersects any edit in others.
// Two whole-value edits for the SAME value share an identical range (overlap); edits for distinct
// values are disjoint. The elision merge uses this to drop a head+tail edit superseded by a
// cross-turn fold on the same value before applySpliceEdits (which would otherwise bail on overlap).
func spliceEditOverlapsAny(e spliceEdit, others []spliceEdit) bool {
	for _, o := range others {
		if e.start < o.end && o.start < e.end {
			return true
		}
	}
	return false
}

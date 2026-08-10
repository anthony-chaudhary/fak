package trajectory

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"sort"
	"strings"
)

// MINING — segment a recorded trajectory into named, reusable workload subpatterns.
//
// The corpus is the SAME [Turn] stream the Recorder already folds out of the kernel's
// abi.Event fan-out and ExportTo/ImportFrom already round-trip; the miner adds no
// parallel event model. Anything that can produce Turns (a live Recorder, a JSONL
// export, the scrubbed-chat adapter in chatexport.go) can be mined.
//
// PRIVACY IS THE DEFAULT, NOT AN OPTION. A [Report] carries tool names, closed-vocabulary
// verdict/reason labels, counts, turn ranges, and sha256 digests — never [Turn.Query],
// never [Turn.Labels], never an args or result body. Those are the three channels a raw
// prompt, a user utterance, or a secret can actually ride on, and the report type has no
// field that reaches them unless the caller explicitly sets MineOptions.Excerpts.

// MineSchemaVersion is the stable id of the [Report] wire schema. New fields are
// additive (omitempty) so an older reader keeps parsing; a breaking change bumps it.
const MineSchemaVersion = "fak.trajectory-mine/1"

// SubpatternID is a stable catalog identifier for one recognized workload motif.
// The IDs are namespaced (`wp.sub.*`) so they can be adopted by the versioned pattern
// catalog (#6210) without a rename: this package OWNS the detectors and the evidence,
// the catalog owns the prose definitions and the composition edges.
type SubpatternID string

const (
	// SubReadEditVerify is the canonical inspect-then-change-then-check loop:
	// one or more read/search turns, then one or more edit turns, then an exec turn.
	SubReadEditVerify SubpatternID = "sp.read-edit-verify"
	// SubSearchFanout is broad orientation: a run of search-family calls with
	// distinct argument digests, i.e. genuinely different probes rather than a replay.
	SubSearchFanout SubpatternID = "sp.search-fanout"
	// SubRetryAfterRefusal is adaptation to a kernel refusal: a DENY/QUARANTINE/WITNESS
	// verdict followed shortly by another call of the SAME tool.
	SubRetryAfterRefusal SubpatternID = "sp.retry-after-refusal"
	// SubRedundantReplay is wasted work: the same tool re-called with an identical
	// argument digest several times inside a short span.
	SubRedundantReplay SubpatternID = "sp.redundant-replay"
	// SubBlindEditRun is a run of edits with no exec turn between them — changes
	// accumulating without a verification step.
	SubBlindEditRun SubpatternID = "sp.blind-edit-run"
	// SubCacheWarmReplay is a run of turns served from the cache rather than recomputed.
	SubCacheWarmReplay SubpatternID = "sp.cache-warm-replay"
)

// Detector thresholds. They are constants, not options: a subpattern whose boundary
// moves per call is not a stable catalog id, and the report has to mean the same thing
// across two runs for the aggregate counts to be comparable.
const (
	minFanoutTurns  = 3  // search-family calls (and distinct digests) before a fan-out is named
	minReplayTurns  = 3  // identical tool+digest calls before a replay is named
	maxReplaySpan   = 12 // ...and how close together they must sit to count as one replay
	minEditRunTurns = 3  // consecutive edits before an unverified run is named
	minCacheRun     = 3  // consecutive cache-served turns before a warm replay is named
	retryWindow     = 3  // turns after a refusal in which a same-tool call reads as a retry
	defaultExcerpt  = 80 // runes per opt-in excerpt
)

// MineOptions configures a mining pass. The zero value is the privacy-safe default:
// no content leaves the corpus.
type MineOptions struct {
	// Excerpts opts IN to attaching raw Query text to each segment. It is off by
	// default and there is deliberately no env/config path that turns it on — a caller
	// that wants user content has to name it in code and own that decision.
	Excerpts bool
	// ExcerptRunes caps each excerpt's length. <= 0 uses defaultExcerpt. Ignored
	// unless Excerpts is set.
	ExcerptRunes int
}

// Segment is one recognized subpattern occurrence: which trace, which turn range
// (inclusive, 1-based [Turn.Seq] values), and WHY the detector fired. Reason is built
// only from counts and closed-vocabulary labels, so it is safe to print verbatim.
type Segment struct {
	TraceID    string       `json:"trace_id"`
	Subpattern SubpatternID `json:"subpattern"`
	StartSeq   int          `json:"start_seq"`
	EndSeq     int          `json:"end_seq"`
	Turns      int          `json:"turns"`
	Reason     string       `json:"reason"`
	Tools      []string     `json:"tools"`              // distinct tool names in the span, sorted
	SpanDigest string       `json:"span_digest"`        // sha256 identity of the span, content-free
	Excerpts   []string     `json:"excerpts,omitempty"` // OPT-IN ONLY (MineOptions.Excerpts)
}

// Aggregate is the cross-trace rollup for one subpattern — the "which motifs does this
// fleet actually run" answer that a single trace cannot give.
type Aggregate struct {
	Subpattern SubpatternID `json:"subpattern"`
	Segments   int          `json:"segments"`
	Traces     int          `json:"traces"`
	Turns      int          `json:"turns"`
}

// Report is the deterministic result of a mining pass. Marshaling the same corpus twice
// yields byte-identical JSON: every slice is ordered by an explicit total order and no
// map iteration reaches the output.
type Report struct {
	Version string `json:"version"`
	Traces  int    `json:"traces"`
	Turns   int    `json:"turns"`
	// Redacted is true when the report carries no corpus content (the default).
	Redacted        bool        `json:"redacted"`
	Segments        []Segment   `json:"segments"`
	Aggregates      []Aggregate `json:"aggregates"`
	AbstainedTraces []string    `json:"abstained_traces"`
	// OverlapDropped counts candidate segments the overlap policy discarded. It is the
	// audit trail for "why is this turn range named X and not Y".
	OverlapDropped int `json:"overlap_dropped"`
}

// Mine segments every turn this Recorder holds. Convenience over [MineTurns].
func (r *Recorder) Mine(opt MineOptions) Report { return MineTurns(r.Turns(), opt) }

// MineTurns segments an ordered turn corpus into named subpatterns and rolls the result
// up across traces.
//
// Turns are grouped by TraceID in first-encounter order and sorted by Seq within a
// trace, so an out-of-order JSONL export mines the same as an in-order one. A trace no
// detector fires on ABSTAINS — it is named in AbstainedTraces and contributes no
// segment. Forcing every trace into some bucket is how a taxonomy starts lying; an
// unrecognized trajectory is a real and reportable outcome.
func MineTurns(turns []Turn, opt MineOptions) Report {
	rep := Report{
		Version:         MineSchemaVersion,
		Turns:           len(turns),
		Redacted:        !opt.Excerpts,
		Segments:        []Segment{},
		Aggregates:      []Aggregate{},
		AbstainedTraces: []string{},
	}
	order, byTrace := groupTurns(turns)
	rep.Traces = len(order)

	for _, id := range order {
		trace := byTrace[id]
		accepted, dropped := resolveOverlaps(runDetectors(trace))
		rep.OverlapDropped += dropped
		if len(accepted) == 0 {
			rep.AbstainedTraces = append(rep.AbstainedTraces, id)
			continue
		}
		for _, c := range accepted {
			rep.Segments = append(rep.Segments, segmentOf(id, trace, c, opt))
		}
	}
	rep.Aggregates = aggregate(rep.Segments)
	return rep
}

// groupTurns buckets a flat corpus by trace, preserving first-encounter trace order and
// sorting each trace by Seq. Turns with no TraceID have no analysis home and are dropped
// (the same rule turnFromEvent applies at record time).
func groupTurns(turns []Turn) ([]string, map[string][]Turn) {
	order := []string{}
	byTrace := map[string][]Turn{}
	for _, t := range turns {
		if t.TraceID == "" {
			continue
		}
		if _, seen := byTrace[t.TraceID]; !seen {
			order = append(order, t.TraceID)
		}
		byTrace[t.TraceID] = append(byTrace[t.TraceID], t)
	}
	for id, ts := range byTrace {
		sort.SliceStable(ts, func(i, j int) bool { return ts[i].Seq < ts[j].Seq })
		byTrace[id] = ts
	}
	return order, byTrace
}

// ---------------------------------------------------------------------------
// Tool families — the closed vocabulary the sequence detectors reason over.
// ---------------------------------------------------------------------------

type toolFamily string

const (
	famRead   toolFamily = "read"
	famSearch toolFamily = "search"
	famEdit   toolFamily = "edit"
	famExec   toolFamily = "exec"
	famOther  toolFamily = "other"
)

// familyTable maps a lowercased tool name to its family. It is deliberately a CLOSED
// table rather than a fuzzy matcher: an unknown tool lands in famOther and simply does
// not participate in a sequence motif, which abstains instead of guessing. Note that a
// shell tool is classed famExec even when the agent used it to search — the miner sees
// the tool NAME only, never the argument body, which is exactly the privacy boundary.
var familyTable = map[string]toolFamily{
	"read": famRead, "notebookread": famRead, "readmcpresource": famRead, "view": famRead,
	"grep": famSearch, "glob": famSearch, "search": famSearch, "websearch": famSearch,
	"codesearch": famSearch, "find": famSearch, "select-string": famSearch,
	"edit": famEdit, "write": famEdit, "multiedit": famEdit, "notebookedit": famEdit,
	"apply_patch": famEdit, "applypatch": famEdit, "str_replace_editor": famEdit,
	"bash": famExec, "powershell": famExec, "shell": famExec, "run": famExec,
	"bashoutput": famExec, "exec": famExec, "terminal": famExec,
}

func familyOf(tool string) toolFamily {
	if f, ok := familyTable[strings.ToLower(strings.TrimSpace(tool))]; ok {
		return f
	}
	return famOther
}

// isLookFamily is "the agent was gathering information" — read or search.
func isLookFamily(tool string) bool {
	f := familyOf(tool)
	return f == famRead || f == famSearch
}

// isRefusal reports whether a verdict label is a kernel refusal the agent must react
// to. The labels are the closed set verdictName emits.
func isRefusal(verdict string) bool {
	switch verdict {
	case "DENY", "QUARANTINE", "WITNESS":
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// Detectors
// ---------------------------------------------------------------------------

// candidate is a detector hit before the overlap policy runs. start/end are INDICES
// into the trace slice, not Seq values.
type candidate struct {
	id     SubpatternID
	prio   int
	start  int
	end    int
	reason string
}

// detector binds a catalog id to its matcher and its overlap priority (lower wins).
type detector struct {
	id    SubpatternID
	prio  int
	match func(trace []Turn, id SubpatternID, prio int) []candidate
}

// detectors is the registered detector set, in a FIXED order. Order plus prio is what
// makes the overlap resolution total (and therefore the report deterministic).
var detectors = []detector{
	{SubReadEditVerify, 0, detectReadEditVerify},
	{SubRetryAfterRefusal, 0, detectRetryAfterRefusal},
	{SubSearchFanout, 1, detectSearchFanout},
	{SubRedundantReplay, 1, detectRedundantReplay},
	{SubBlindEditRun, 2, detectBlindEditRun},
	{SubCacheWarmReplay, 2, detectCacheWarmReplay},
}

func runDetectors(trace []Turn) []candidate {
	var out []candidate
	for _, d := range detectors {
		out = append(out, d.match(trace, d.id, d.prio)...)
	}
	return out
}

// detectReadEditVerify walks for the minimal window `look+ edit+ exec+`: a run of
// read/search turns, immediately followed by a run of edits, immediately followed by a
// run of exec turns. Requiring contiguity is what keeps the claim honest — an edit ten
// turns after an unrelated read is not one loop.
func detectReadEditVerify(trace []Turn, id SubpatternID, prio int) []candidate {
	var out []candidate
	i := 0
	for i < len(trace) {
		if !isLookFamily(trace[i].Tool) {
			i++
			continue
		}
		look := i
		for look < len(trace) && isLookFamily(trace[look].Tool) {
			look++
		}
		edit := look
		for edit < len(trace) && familyOf(trace[edit].Tool) == famEdit {
			edit++
		}
		if edit == look { // no edit followed the look run
			i = look
			continue
		}
		exec := edit
		for exec < len(trace) && familyOf(trace[exec].Tool) == famExec {
			exec++
		}
		if exec == edit { // edits were never followed by a verification turn
			i = edit
			continue
		}
		out = append(out, candidate{id, prio, i, exec - 1, fmt.Sprintf(
			"%d look, %d edit, %d exec turn(s) in order", look-i, edit-look, exec-edit)})
		i = exec
	}
	return out
}

// detectSearchFanout names a run of at least minFanoutTurns search-family calls whose
// argument digests are mostly DISTINCT. The distinctness test is what separates
// orientation from a replay: repeating one grep is SubRedundantReplay, not fan-out.
func detectSearchFanout(trace []Turn, id SubpatternID, prio int) []candidate {
	var out []candidate
	i := 0
	for i < len(trace) {
		if familyOf(trace[i].Tool) != famSearch {
			i++
			continue
		}
		j := i
		seen := map[string]bool{}
		for j < len(trace) && familyOf(trace[j].Tool) == famSearch {
			seen[trace[j].ArgsDigest] = true
			j++
		}
		if j-i >= minFanoutTurns && len(seen) >= minFanoutTurns {
			out = append(out, candidate{id, prio, i, j - 1, fmt.Sprintf(
				"%d consecutive search-family call(s) with %d distinct arg digest(s)", j-i, len(seen))})
		}
		i = j
	}
	return out
}

// detectRetryAfterRefusal pairs a refusal with the next same-tool call inside
// retryWindow turns. The reason quotes the kernel's own verdict/reason labels, which are
// a closed vocabulary (see verdictName / abi.ReasonName) and carry no call content.
func detectRetryAfterRefusal(trace []Turn, id SubpatternID, prio int) []candidate {
	var out []candidate
	for i := range trace {
		if !isRefusal(trace[i].Verdict) || trace[i].Tool == "" {
			continue
		}
		for j := i + 1; j < len(trace) && j <= i+retryWindow; j++ {
			if trace[j].Tool != trace[i].Tool {
				continue
			}
			reason := trace[i].Reason
			if reason == "" {
				reason = "NONE"
			}
			out = append(out, candidate{id, prio, i, j, fmt.Sprintf(
				"%s(%s) retried with the same tool %d turn(s) later", trace[i].Verdict, reason, j-i)})
			break
		}
	}
	return out
}

// detectRedundantReplay finds a tool re-called with an IDENTICAL argument digest at
// least minReplayTurns times inside a maxReplaySpan window. Identity comes from the
// digest the recorder already computed, so the detector never sees the arguments.
func detectRedundantReplay(trace []Turn, id SubpatternID, prio int) []candidate {
	type replayKey struct{ tool, digest string }
	groups := map[replayKey][]int{}
	var keys []replayKey
	for i, t := range trace {
		if t.Tool == "" || t.ArgsDigest == "" {
			continue
		}
		k := replayKey{t.Tool, t.ArgsDigest}
		if _, seen := groups[k]; !seen {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], i)
	}
	// Iterate a sorted key slice, never the map: map order would make the report
	// non-deterministic exactly where two candidates tie.
	sort.Slice(keys, func(a, b int) bool {
		if keys[a].tool != keys[b].tool {
			return keys[a].tool < keys[b].tool
		}
		return keys[a].digest < keys[b].digest
	})
	var out []candidate
	for _, k := range keys {
		idx := groups[k]
		if len(idx) < minReplayTurns {
			continue
		}
		first, last := idx[0], idx[len(idx)-1]
		if last-first+1 > maxReplaySpan {
			continue
		}
		out = append(out, candidate{id, prio, first, last, fmt.Sprintf(
			"identical tool+arg digest repeated %d time(s) within a %d-turn span", len(idx), last-first+1)})
	}
	return out
}

// detectBlindEditRun names minEditRunTurns or more consecutive edits with no exec turn
// between them: changes piling up without a verification step.
func detectBlindEditRun(trace []Turn, id SubpatternID, prio int) []candidate {
	var out []candidate
	i := 0
	for i < len(trace) {
		if familyOf(trace[i].Tool) != famEdit {
			i++
			continue
		}
		j := i
		for j < len(trace) && familyOf(trace[j].Tool) == famEdit {
			j++
		}
		if j-i >= minEditRunTurns {
			out = append(out, candidate{id, prio, i, j - 1, fmt.Sprintf(
				"%d consecutive edit-family call(s) with no verification turn", j-i)})
		}
		i = j
	}
	return out
}

// detectCacheWarmReplay names a run of turns the kernel served from cache.
func detectCacheWarmReplay(trace []Turn, id SubpatternID, prio int) []candidate {
	warm := func(t Turn) bool { return t.CacheHit || t.Materialized == "HIT" }
	var out []candidate
	i := 0
	for i < len(trace) {
		if !warm(trace[i]) {
			i++
			continue
		}
		j := i
		for j < len(trace) && warm(trace[j]) {
			j++
		}
		if j-i >= minCacheRun {
			out = append(out, candidate{id, prio, i, j - 1, fmt.Sprintf(
				"%d consecutive cache-served turn(s)", j-i)})
		}
		i = j
	}
	return out
}

// ---------------------------------------------------------------------------
// Overlap policy
// ---------------------------------------------------------------------------

// resolveOverlaps applies THE overlap policy: within one trace a turn belongs to at
// most one named segment. Candidates are ranked by (priority asc, span desc, start asc,
// id asc) and accepted greedily; a candidate that overlaps an already-accepted span is
// dropped and counted. Returns the accepted set in start order plus the drop count.
//
// The ranking is a total order, so the outcome never depends on detector registration
// luck or map iteration. The policy is deliberately winner-takes-the-turns rather than
// "report both": two names over one turn range make the aggregate counts double-count
// the same work, which is worse than losing the weaker name.
func resolveOverlaps(cands []candidate) ([]candidate, int) {
	ranked := append([]candidate(nil), cands...)
	sort.Slice(ranked, func(i, j int) bool {
		a, b := ranked[i], ranked[j]
		if a.prio != b.prio {
			return a.prio < b.prio
		}
		if sa, sb := a.end-a.start, b.end-b.start; sa != sb {
			return sa > sb
		}
		if a.start != b.start {
			return a.start < b.start
		}
		return a.id < b.id
	})
	var accepted []candidate
	dropped := 0
	for _, c := range ranked {
		clash := false
		for _, a := range accepted {
			if c.start <= a.end && a.start <= c.end {
				clash = true
				break
			}
		}
		if clash {
			dropped++
			continue
		}
		accepted = append(accepted, c)
	}
	sort.Slice(accepted, func(i, j int) bool {
		if accepted[i].start != accepted[j].start {
			return accepted[i].start < accepted[j].start
		}
		return accepted[i].id < accepted[j].id
	})
	return accepted, dropped
}

// ---------------------------------------------------------------------------
// Projection to the report
// ---------------------------------------------------------------------------

// segmentOf projects a candidate into the exported [Segment]. This is the ONE place a
// Turn becomes report bytes, so it is the one place the privacy boundary has to hold:
// it reads Tool, Verdict, ArgsDigest, ResultDigest and Seq, and reaches Query only
// behind the explicit opt-in.
func segmentOf(traceID string, trace []Turn, c candidate, opt MineOptions) Segment {
	span := trace[c.start : c.end+1]
	seg := Segment{
		TraceID:    traceID,
		Subpattern: c.id,
		StartSeq:   span[0].Seq,
		EndSeq:     span[len(span)-1].Seq,
		Turns:      len(span),
		Reason:     c.reason,
		Tools:      distinctTools(span),
		SpanDigest: spanDigest(traceID, span),
	}
	if opt.Excerpts {
		seg.Excerpts = excerptsOf(span, opt.ExcerptRunes)
	}
	return seg
}

func distinctTools(span []Turn) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, t := range span {
		if t.Tool == "" || seen[t.Tool] {
			continue
		}
		seen[t.Tool] = true
		out = append(out, t.Tool)
	}
	sort.Strings(out)
	return out
}

// spanDigest is a content-free identity for a span: it hashes the trace id and each
// turn's seq/tool/verdict/args-digest/result-digest. Two runs over the same corpus
// produce the same digest, and the digest reveals nothing the report does not already
// print — it exists so a downstream consumer can dedupe or join without content.
func spanDigest(traceID string, span []Turn) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\n", traceID)
	for _, t := range span {
		fmt.Fprintf(h, "%d\x00%s\x00%s\x00%s\x00%s\n", t.Seq, t.Tool, t.Verdict, t.ArgsDigest, t.ResultDigest)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// excerptsOf is the OPT-IN content path. It is the only function in this file that
// reads Turn.Query, and nothing calls it unless MineOptions.Excerpts is set.
func excerptsOf(span []Turn, runes int) []string {
	if runes <= 0 {
		runes = defaultExcerpt
	}
	out := []string{}
	for _, t := range span {
		if t.Query == "" {
			continue
		}
		r := []rune(t.Query)
		if len(r) > runes {
			r = r[:runes]
		}
		out = append(out, string(r))
	}
	return out
}

// aggregate rolls segments up per subpattern, ordered by segment count desc then id asc
// so the ranking is stable across runs.
func aggregate(segs []Segment) []Aggregate {
	type acc struct {
		segments, turns int
		traces          map[string]bool
	}
	byID := map[SubpatternID]*acc{}
	var order []SubpatternID
	for _, s := range segs {
		a, ok := byID[s.Subpattern]
		if !ok {
			a = &acc{traces: map[string]bool{}}
			byID[s.Subpattern] = a
			order = append(order, s.Subpattern)
		}
		a.segments++
		a.turns += s.Turns
		a.traces[s.TraceID] = true
	}
	out := make([]Aggregate, 0, len(order))
	for _, id := range order {
		a := byID[id]
		out = append(out, Aggregate{Subpattern: id, Segments: a.segments, Traces: len(a.traces), Turns: a.turns})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Segments != out[j].Segments {
			return out[i].Segments > out[j].Segments
		}
		return out[i].Subpattern < out[j].Subpattern
	})
	return out
}

// Render writes the human readout. It prints exactly what the JSON carries — the human
// and machine surfaces share one privacy boundary, so a reader cannot be shown content
// the JSON withholds.
func (rep Report) Render(w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%s  traces=%d turns=%d segments=%d overlap_dropped=%d redacted=%t\n",
		rep.Version, rep.Traces, rep.Turns, len(rep.Segments), rep.OverlapDropped, rep.Redacted); err != nil {
		return err
	}
	for _, a := range rep.Aggregates {
		if _, err := fmt.Fprintf(w, "  %-28s segments=%d traces=%d turns=%d\n",
			a.Subpattern, a.Segments, a.Traces, a.Turns); err != nil {
			return err
		}
	}
	for _, s := range rep.Segments {
		if _, err := fmt.Fprintf(w, "  %s turns %d-%d %s [%s] — %s\n",
			s.TraceID, s.StartSeq, s.EndSeq, s.Subpattern, strings.Join(s.Tools, ","), s.Reason); err != nil {
			return err
		}
	}
	for _, id := range rep.AbstainedTraces {
		if _, err := fmt.Fprintf(w, "  %s ABSTAIN no subpattern matched\n", id); err != nil {
			return err
		}
	}
	return nil
}

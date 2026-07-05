package guardrsi

import (
	"sort"
	"strings"
)

// Fleet correlation is the cross-trace complement of the per-trace LivelockDetector
// above. LivelockDetector answers "is THIS one agent stuck repeating the same
// refused call?"; Correlate answers "are N DIFFERENT agents all hitting the same
// shared cause right now?" — the signal the blast-radius containment epic (#2712)
// needs to turn "N agents each rediscover the bug" into "1 agent records it, the
// rest read it". It produces a candidate the W1 knownbad record seam (#2713) can
// then persist; it writes nothing itself.
const (
	// FleetCorrelateSchema tags a correlation candidate so a downstream reader (the
	// gateway forward, a ledger writer) can filter for exactly this producer's rows,
	// the same schema-tag idiom LivelockSchema uses.
	FleetCorrelateSchema = "guardrsi.fleet-correlate/1"

	// KnownBadCandidateEvent is the event a promoted candidate carries — the
	// cross-trace analogue of LivelockEvent (which is per-trace, per-session).
	KnownBadCandidateEvent = "FLEET_KNOWN_BAD_CANDIDATE"

	// DefaultCorrelateK is the number of DISTINCT traces at/above which a shared
	// FailureHash promotes to a known-bad candidate. It mirrors
	// DefaultLivelockThreshold's "third strike" shape but counts fleet breadth
	// (distinct traces) rather than repeats within one trace.
	DefaultCorrelateK = 3

	// DefaultCorrelateWindowSecs is how long a fleet observation stays eligible to
	// correlate. Observations older than this at `now` do not count toward K, so a
	// candidate reflects a currently-shared failure, not a stale coincidence.
	DefaultCorrelateWindowSecs int64 = 900
)

// FleetObservation is one trace's report that a specific shared failure —
// identified by the content-free FailureHash a LivelockEnvelope already carries —
// was seen over a declared tree at time TS. It is the cross-trace input the
// gateway feeds from its existing livelock observation point. The correlator never
// reads a clock, so TS and `now` are supplied as data.
type FleetObservation struct {
	TraceID     string
	FailureHash string
	Reason      string
	TreeGlobs   []string
	TS          int64
}

// KnownBadCandidate is the promoted output: a shared FailureHash that at least K
// DISTINCT traces hit inside the window. It carries the failure's reason class and
// the deduped, sorted union of the correlated traces' declared trees as the
// signature tree — exactly the shape the W1 knownbad record seam (#2713) needs. It
// is a candidate, not a ledger row: it only states "N traces share this cause" and
// writes nothing.
type KnownBadCandidate struct {
	Schema         string   `json:"schema"`
	Event          string   `json:"event"`
	FailureHash    string   `json:"failure_hash"`
	ReasonClass    string   `json:"reason_class,omitempty"`
	TreeGlobs      []string `json:"tree_globs,omitempty"`
	TraceIDs       []string `json:"trace_ids"`
	DistinctTraces int      `json:"distinct_traces"`
	WindowSecs     int64    `json:"window_secs"`
	FirstSeenUnix  int64    `json:"first_seen_unix"`
	LastSeenUnix   int64    `json:"last_seen_unix"`
}

// correlateAgg accumulates one FailureHash's in-window evidence during the fold.
type correlateAgg struct {
	reason    string
	traces    map[string]struct{}
	traceList []string // distinct traces in first-seen order (stable pre-sort)
	trees     map[string]struct{}
	first     int64
	last      int64
}

// Correlate folds a batch of cross-trace fleet observations into known-bad
// candidates: one candidate per FailureHash that at least k DISTINCT traces
// emitted inside the [now-windowSecs, now] window. It is pure — `now` and every TS
// are data, so the same inputs always yield the same candidates.
//
// The distinct-trace rule is load-bearing: k repeats from a SINGLE trace is a
// per-trace livelock (already handled by LivelockDetector), not a fleet-wide shared
// cause, so it returns nothing. Observations older than the window are dropped
// before counting, so a stale coincidence never promotes.
//
// A candidate carries the reason class (the first in-window observation of that
// hash wins — the FailureHash already encodes the reason, so they agree) and the
// deduped, sorted union of the correlated traces' declared trees as the signature
// tree. Candidates are returned sorted by FailureHash for a deterministic order.
// A non-positive k or windowSecs falls back to the package defaults.
func Correlate(obs []FleetObservation, k int, windowSecs, now int64) []KnownBadCandidate {
	if k <= 0 {
		k = DefaultCorrelateK
	}
	if windowSecs <= 0 {
		windowSecs = DefaultCorrelateWindowSecs
	}
	cutoff := now - windowSecs

	byHash := map[string]*correlateAgg{}
	var order []string // first-seen order of hashes, kept stable before the final sort

	for _, o := range obs {
		trace := strings.TrimSpace(o.TraceID)
		hash := strings.TrimSpace(o.FailureHash)
		if trace == "" || hash == "" {
			continue // an anonymous or hash-less observation cannot be correlated
		}
		if o.TS < cutoff {
			continue // older than the window — does not count toward K
		}
		a := byHash[hash]
		if a == nil {
			a = &correlateAgg{
				reason: strings.TrimSpace(o.Reason),
				traces: map[string]struct{}{},
				trees:  map[string]struct{}{},
				first:  o.TS,
				last:   o.TS,
			}
			byHash[hash] = a
			order = append(order, hash)
		}
		if a.reason == "" {
			a.reason = strings.TrimSpace(o.Reason)
		}
		if _, seen := a.traces[trace]; !seen {
			a.traces[trace] = struct{}{}
			a.traceList = append(a.traceList, trace)
		}
		for _, g := range o.TreeGlobs {
			if g = strings.TrimSpace(g); g != "" {
				a.trees[g] = struct{}{}
			}
		}
		if o.TS < a.first {
			a.first = o.TS
		}
		if o.TS > a.last {
			a.last = o.TS
		}
	}

	var out []KnownBadCandidate
	for _, hash := range order {
		a := byHash[hash]
		if len(a.traces) < k {
			continue
		}
		traces := append([]string(nil), a.traceList...)
		sort.Strings(traces)
		trees := make([]string, 0, len(a.trees))
		for g := range a.trees {
			trees = append(trees, g)
		}
		sort.Strings(trees)
		out = append(out, KnownBadCandidate{
			Schema:         FleetCorrelateSchema,
			Event:          KnownBadCandidateEvent,
			FailureHash:    hash,
			ReasonClass:    a.reason,
			TreeGlobs:      trees,
			TraceIDs:       traces,
			DistinctTraces: len(a.traces),
			WindowSecs:     windowSecs,
			FirstSeenUnix:  a.first,
			LastSeenUnix:   a.last,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].FailureHash < out[j].FailureHash })
	return out
}

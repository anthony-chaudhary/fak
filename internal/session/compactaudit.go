package session

// compactaudit.go — mine native Codex rollout JSONL for COMPACTION HEALTH (#4763).
//
// The operator question this answers is "did compaction fire, hold, and reduce
// resident context?" — which is NOT answerable from the quantities operators reach
// for first. Rollout JSONL is append-only, so file bytes only ever grow; provider
// token usage is reported cumulatively, so it is monotonic across a session. A
// perfectly healthy session that compacted 27 times is still tens of MB and still
// reports hundreds of millions of cumulative tokens. Reading either as "compaction
// never fired" is an observability failure, not a compaction failure.
//
// The one quantity that actually answers the question is RESIDENT context:
// `event_msg/token_count.info.last_token_usage.input_tokens` — what is in the window
// right now. (`total_token_usage` is the cumulative counter and will read as >100% of
// the window; it is recorded here only so the report can show the two side by side and
// say which is which.) A fire is healthy when the resident count drops across it and
// stays down; the file being large is orthogonal.
//
// Two rollout facts shape the scanner:
//
//   - A fire emits a PAIR: a top-level `type:"compacted"` row and an
//     `event_msg/context_compacted` row, typically milliseconds apart. They are one
//     fire and are counted once. Two events of the SAME kind inside the window are a
//     genuine duplicate and are flagged.
//   - Codex emits a zero-valued `token_count` row between the pair. Pairing a fire to
//     the *adjacent* sample would therefore read post-fire resident as 0 and score
//     every fire as a miracle. Pre/post bind to the nearest NON-ZERO sample instead.
//
// Where adjacency is ambiguous or a witness is absent, a fire is classified with a
// typed confidence + reason rather than silently asserted as a failure — a live audit
// found 23 pairs whose next non-zero sample was not lower, and telemetry adjacency
// artifacts are an expected cause of those.
//
// Rows are read head-bounded (see readRollupRow): the scanner never materializes a
// whole transcript, and the megabyte-scale rows — `compacted.replacement_history`,
// `function_call_output`, `session_meta.base_instructions` — are truncated to a bounded
// head and discarded. Prompt and tool-output bodies are never retained or emitted.

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// Fire-classification thresholds. They are named constants (not magic numbers at the
// comparison site) because an operator reading an anomaly needs to know the bar it
// failed against.
const (
	// CompactCeilingLateFraction — a fire whose pre-fire resident had already reached
	// this fraction of the model context window fired LATE: it ran to the ceiling.
	CompactCeilingLateFraction = 0.95
	// CompactCeilingApproachFraction — a session whose peak resident reached this
	// fraction of the window was under real pressure, so zero fires is an anomaly
	// rather than simply a short session.
	CompactCeilingApproachFraction = 0.90
	// CompactOversizedResidualRatio — post/pre above this means the fire shed little:
	// it technically fired but left most of the window resident.
	CompactOversizedResidualRatio = 0.50
	// CompactFastReboundTurns — refilling to CompactReboundFraction of the pre-fire
	// resident within this many turns means the shed bought almost no headroom.
	CompactFastReboundTurns = 3
	// CompactReboundFraction — the share of pre-fire resident that counts as rebounded.
	CompactReboundFraction = 0.90
	// CompactPairWindow — two fire events within this window are the same fire (the
	// compacted/context_compacted pair), not two fires.
	CompactPairWindow = 2 * time.Second
	// CompactAdjacencyTurns — a pre/post witness further than this many turns from the
	// fire is too far to bind confidently; the fire is classified low-confidence rather
	// than scored as fact.
	CompactAdjacencyTurns = 2
)

// Typed anomaly tokens. A closed vocabulary: an operator ranks sessions by these, so
// they are stable identifiers, not prose.
const (
	AnomalyNoFireAboveCeiling = "NO_FIRE_ABOVE_CEILING"
	AnomalyLateFire           = "LATE_FIRE"
	AnomalyIneffectiveFire    = "INEFFECTIVE_FIRE"
	AnomalyOversizedResidual  = "OVERSIZED_RESIDUAL"
	AnomalyFastRebound        = "FAST_REBOUND"
	AnomalyDuplicateFireEvent = "DUPLICATE_FIRE_EVENT"
	AnomalyMissingPreWitness  = "MISSING_PRE_WITNESS"
	AnomalyMissingPostWitness = "MISSING_POST_WITNESS"
	// AnomalyWedgedAtCeiling — the session FIRED repeatedly yet resident context never came
	// down off the ceiling: every measured fire left post-fire resident still above the
	// late-fire fraction of the window. This is the signature an oversized single item produces
	// (a large image or paste the shedder cannot drop): compaction keeps firing but the window
	// walk can never seat a kept window under budget, so the session sails at the top firing
	// uselessly. It is distinct from INEFFECTIVE_FIRE (one fire that did not reduce resident) and
	// from NO_FIRE_ABOVE_CEILING (never fired at all): here the fires HAPPEN and still do not help.
	AnomalyWedgedAtCeiling = "WEDGED_AT_CEILING"
)

// CompactWedgeMinFires — the minimum number of measured fires before a session's persistent
// high-ceiling residency is read as a WEDGE rather than one late fire. Two consecutive fires that
// both leave resident above the ceiling is the smallest pattern that distinguishes "stuck" from a
// single ill-timed cut.
const CompactWedgeMinFires = 2

// Typed confidence + reason for a fire's pre/post binding.
const (
	CompactConfidenceHigh = "high"
	CompactConfidenceLow  = "low"
	CompactConfidenceNone = "none"

	CompactReasonOK                 = "OK"
	CompactReasonAdjacencyAmbiguous = "ADJACENCY_AMBIGUOUS"
	CompactReasonTelemetryMissing   = "TELEMETRY_MISSING"
)

// Session verdicts — the headline the human report leads with, so a large append-only
// file is never read as a failure.
const (
	VerdictFiredAndHeld     = "FIRED_AND_HELD"
	VerdictFiredWithAnomaly = "FIRED_WITH_ANOMALIES"
	VerdictNoFireBounded    = "NO_FIRE_BOUNDED"
	VerdictNoFireAtCeiling  = "NO_FIRE_ABOVE_CEILING"
	VerdictTelemetryMissing = "NO_TELEMETRY"
	// VerdictWedgedAtCeiling — fired repeatedly and still never came off the ceiling (the
	// oversized-item wedge). Ranked as the worst FIRED outcome: unlike FIRED_WITH_ANOMALIES it is
	// not a one-off artifact but a session that is structurally stuck.
	VerdictWedgedAtCeiling = "WEDGED_AT_CEILING"
)

// CompactFire is one compaction event joined to its resident-context witnesses.
type CompactFire struct {
	Index int       `json:"index"`
	At    time.Time `json:"at"`
	Turn  int       `json:"turn"`

	// PreTokens/PostTokens are RESIDENT context (last_token_usage.input_tokens) at the
	// nearest non-zero sample either side of the fire; 0 means "no witness", which is
	// reported as an anomaly rather than as a real zero.
	PreTokens  int `json:"pre_tokens"`
	PostTokens int `json:"post_tokens"`
	Shed       int `json:"shed_tokens"`

	// ResidualRatio is post/pre — how much of the window survived the fire.
	// CeilingRatio is pre/window — how close to the ceiling it fired.
	ResidualRatio float64 `json:"residual_ratio"`
	CeilingRatio  float64 `json:"ceiling_ratio"`
	ContextWindow int     `json:"context_window"`

	// ReboundTurns/ReboundSeconds measure how fast resident context returned to
	// CompactReboundFraction of pre-fire. 0 = never rebounded within the session.
	ReboundTurns   int     `json:"rebound_turns"`
	ReboundSeconds float64 `json:"rebound_seconds"`

	Confidence string   `json:"confidence"`
	Reason     string   `json:"reason"`
	Anomalies  []string `json:"anomalies,omitempty"`

	// Regrowth is this fire's post-fire trajectory and content-class attribution
	// (#4768) — how fast the window refilled, out of what, and how the observation
	// ended. See compactregrowth.go.
	Regrowth *CompactRegrowth `json:"regrowth,omitempty"`
}

// CompactSessionReport is one rollout file's compaction health.
type CompactSessionReport struct {
	SessionID string `json:"session_id"`
	Path      string `json:"path"`
	Model     string `json:"model"`
	Cwd       string `json:"cwd"`

	// The three quantities the report exists to keep apart.
	Bytes                 int64 `json:"rollout_bytes"`           // append-only: grows forever
	CumulativeInputTokens int   `json:"cumulative_input_tokens"` // monotonic: grows forever
	PeakResidentTokens    int   `json:"peak_resident_tokens"`    // the real occupancy signal
	FinalResidentTokens   int   `json:"final_resident_tokens"`

	ContextWindow int `json:"context_window"`
	Turns         int `json:"turns"`
	ToolCalls     int `json:"tool_calls"`
	TokenSamples  int `json:"token_samples"`

	Fires          []CompactFire `json:"fires"`
	FireCount      int           `json:"fire_count"`
	PairedEvents   int           `json:"paired_events"`   // deduped compacted/context_compacted halves
	DuplicateFires int           `json:"duplicate_fires"` // genuine same-kind repeats

	Verdict   string   `json:"verdict"`
	Anomalies []string `json:"anomalies,omitempty"`
}

// CompactAggregate is the fleet-wide roll-up across many rollouts.
type CompactAggregate struct {
	Sessions          int   `json:"sessions"`
	Bytes             int64 `json:"rollout_bytes"`
	Fires             int   `json:"fires"`
	MeasuredFires     int   `json:"measured_fires"` // fires with both witnesses
	CompactedSessions int   `json:"compacted_sessions"`

	MedianPreTokens     int     `json:"median_pre_tokens"`
	MedianPostTokens    int     `json:"median_post_tokens"`
	MedianShedTokens    int     `json:"median_shed_tokens"`
	MedianResidualRatio float64 `json:"median_residual_ratio"`

	AnomalyCounts map[string]int `json:"anomaly_counts"`
	VerdictCounts map[string]int `json:"verdict_counts"`

	// Regrowth is the corpus-wide rebound/attribution roll-up (#4768); nil when no
	// fire carried post-fire telemetry.
	Regrowth *CompactRegrowthRollup `json:"regrowth,omitempty"`
}

// maxRollupRowHead bounds how much of one JSONL row the scanner holds. Every field
// this audit reads (timestamp, type, payload.type, token counts, model, id) sits in
// the first bytes of its row; the rows that overrun this — replacement_history,
// function_call_output, base_instructions — carry only prompt/tool-output bodies,
// which this audit must not read. Overrun is discarded, not buffered.
const maxRollupRowHead = 128 << 10

// rollupRow is the head-parse of one rollout row. Payload stays raw so only the row
// kinds this audit cares about pay the cost of decoding it.
type rollupRow struct {
	Timestamp string          `json:"timestamp"`
	Type      string          `json:"type"`
	Payload   json.RawMessage `json:"payload"`
}

type rollupTokenPayload struct {
	Type string `json:"type"`
	Info struct {
		LastTokenUsage struct {
			InputTokens int `json:"input_tokens"`
			// CachedInputTokens is the provider's cache-read share of this request's
			// input — the #4768 join that prices regrowth net of reuse.
			CachedInputTokens int `json:"cached_input_tokens"`
		} `json:"last_token_usage"`
		TotalTokenUsage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"total_token_usage"`
		ModelContextWindow int `json:"model_context_window"`
	} `json:"info"`
}

type rollupMetaPayload struct {
	ID  string `json:"id"`
	Cwd string `json:"cwd"`
}

type rollupTurnContextPayload struct {
	Model string `json:"model"`
}

type rollupTypedPayload struct {
	Type string `json:"type"`
}

// readRollupRow reads one line, keeping at most maxRollupRowHead bytes and DISCARDING
// the remainder. This is what makes the scan streaming and body-blind: a 4 MB
// replacement_history row costs a bounded head and is never held. truncated reports
// whether the row overran, so the caller can fall back to a prefix probe instead of
// unmarshalling an incomplete document. rowLen is the FULL row length including the
// discarded tail — the #4768 attribution measures rows by length, never by content.
func readRollupRow(br *bufio.Reader) (head []byte, truncated bool, rowLen int64, err error) {
	var buf []byte
	for {
		chunk, e := br.ReadSlice('\n')
		rowLen += int64(len(chunk))
		if len(buf)+len(chunk) <= maxRollupRowHead {
			buf = append(buf, chunk...)
		} else if room := maxRollupRowHead - len(buf); room > 0 {
			buf = append(buf, chunk[:room]...)
			truncated = true
		} else {
			truncated = true
		}
		if e == nil {
			return buf, truncated, rowLen, nil
		}
		if errors.Is(e, bufio.ErrBufferFull) {
			continue // keep draining this over-long row; the excess is dropped above
		}
		if errors.Is(e, io.EOF) {
			if len(buf) == 0 {
				return nil, truncated, rowLen, io.EOF
			}
			return buf, truncated, rowLen, nil
		}
		return nil, truncated, rowLen, e
	}
}

// typeValueAt returns the nth `"type":"..."` value in a row head. Codex writes the
// top-level type first and payload.type second, so this recovers the kind of a row too
// long to unmarshal — the only place the scanner leans on key order, and only for rows
// whose tail it deliberately dropped.
func typeValueAt(head []byte, n int) string {
	s := string(head)
	for i := 0; i < n; i++ {
		k := strings.Index(s, `"type":"`)
		if k < 0 {
			return ""
		}
		s = s[k+len(`"type":"`):]
		if i == n-1 {
			if e := strings.IndexByte(s, '"'); e >= 0 {
				return s[:e]
			}
			return ""
		}
	}
	return ""
}

// firstStringField recovers one leading string field from a row head that was too long
// to unmarshal. It matters most for `timestamp` on an over-long `compacted` row: a fire
// with no clock cannot be matched to its context_compacted twin inside the pair window,
// which would double-count every fire in exactly the big sessions this audit exists to
// judge. The field sits in the row's first bytes, well inside the head bound.
func firstStringField(head []byte, key string) string {
	pat := `"` + key + `":"`
	k := strings.Index(string(head), pat)
	if k < 0 {
		return ""
	}
	s := string(head)[k+len(pat):]
	if e := strings.IndexByte(s, '"'); e >= 0 {
		return s[:e]
	}
	return ""
}

// pendingFire tracks a fire awaiting its post-fire witness and rebound crossing.
type pendingFire struct {
	idx       int
	haveePost bool
	rebounded bool
}

// ScanCompactRollout streams one Codex rollout and reports its compaction health.
// size is the rollout's byte length, recorded so the report can show append-only bytes
// beside resident context and refuse the "big file = broken" read.
func ScanCompactRollout(r io.Reader, path string, size int64) (CompactSessionReport, error) {
	rep, _, err := scanCompactRollout(r, path, size, RegrowthReplayOptions{})
	return rep, err
}

// ScanCompactRolloutReplay is ScanCompactRollout with the #5254 counterfactual dedup replay
// armed: opt.Fold is run over the tool-result bodies of every post-fire window and the returned
// RegrowthReplayStat scores what that mechanism would have collapsed. A zero-value opt (nil Fold)
// is exactly ScanCompactRollout. Bodies live in memory for one window and are never persisted.
func ScanCompactRolloutReplay(r io.Reader, path string, size int64, opt RegrowthReplayOptions) (CompactSessionReport, RegrowthReplayStat, error) {
	return scanCompactRollout(r, path, size, opt)
}

func scanCompactRollout(r io.Reader, path string, size int64, opt RegrowthReplayOptions) (CompactSessionReport, RegrowthReplayStat, error) {
	rep := CompactSessionReport{Path: path, Bytes: size}
	br := bufio.NewReaderSize(r, 64<<10)

	var (
		turn        int
		toolCalls   int
		lastNonZero int
		lastNZTurn  int
		lastNZTime  time.Time
		haveNonZero bool

		lastFireKind string
		lastFireTime time.Time
		haveFire     bool

		pending []*pendingFire
	)
	tracker := newRegrowthTracker()
	replay := newRegrowthReplay(opt)
	tracker.replay = replay

	// fire records one compaction fire of `kind` at `ts` and, only when that call
	// actually appended a NEW fire (recordFire coalesces a same-fire pair reported by
	// two rows), hands the fire to the regrowth tracker. Both fire-bearing rows a
	// rollout can carry — the top-level `compacted` row and the `context_compacted`
	// event_msg — are the SAME rule, so recording a fire and sampling its regrowth
	// stay welded together: a second fire row spelled differently cannot land in the
	// report while going unseen by the tracker, which is what would leave the regrowth
	// series silently short of the fire count it is supposed to explain.
	fire := func(kind string, ts time.Time) {
		nFires := len(rep.Fires)
		rep, pending, lastFireKind, lastFireTime, haveFire = recordFire(
			rep, pending, kind, ts, turn,
			lastNonZero, lastNZTurn, lastNZTime, haveNonZero,
			lastFireKind, lastFireTime, haveFire)
		if len(rep.Fires) > nFires {
			tracker.onFire(len(rep.Fires)-1, ts, turn, toolCalls)
		}
	}

	for {
		head, truncated, rowLen, err := readRollupRow(br)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return rep, replayStatOf(replay), fmt.Errorf("%s: %w", path, err)
		}
		line := strings.TrimSpace(string(head))
		if line == "" {
			continue
		}

		var row rollupRow
		topType := ""
		payloadType := ""
		rawTS := ""
		parsed := false
		if !truncated && json.Unmarshal([]byte(line), &row) == nil {
			topType = row.Type
			rawTS = row.Timestamp
			parsed = true
		} else {
			// Over-long row: recover the kind + clock from the head. These are body rows
			// (replacement_history / function_call_output / base_instructions); we need
			// their kind and timestamp, never their content.
			topType = typeValueAt(head, 1)
			payloadType = typeValueAt(head, 2)
			rawTS = firstStringField(head, "timestamp")
		}

		ts := time.Time{}
		if rawTS != "" {
			if t, e := time.Parse(time.RFC3339Nano, rawTS); e == nil {
				ts = t
			}
		}

		switch topType {
		case "session_meta":
			if parsed && len(row.Payload) > 0 {
				var m rollupMetaPayload
				if json.Unmarshal(row.Payload, &m) == nil {
					rep.SessionID = m.ID
					rep.Cwd = m.Cwd
				}
			} else {
				// session_meta carries base_instructions (the whole system prompt), so it
				// routinely overruns the head bound. id/cwd lead the payload, ahead of it.
				rep.SessionID = firstStringField(head, "id")
				rep.Cwd = firstStringField(head, "cwd")
			}
		case "turn_context":
			if parsed && len(row.Payload) > 0 {
				var tc rollupTurnContextPayload
				if json.Unmarshal(row.Payload, &tc) == nil && tc.Model != "" {
					rep.Model = tc.Model
				}
			}
		case "response_item":
			if parsed && len(row.Payload) > 0 {
				var p rollupTypedPayload
				if json.Unmarshal(row.Payload, &p) == nil {
					payloadType = p.Type
				}
			}
			if payloadType == "function_call" || payloadType == "custom_tool_call" {
				toolCalls++
			}
			tracker.observeResponseItem(parsed, row.Payload, head, rowLen)
		case "compacted":
			fire("compacted", ts)
			// The compacted row's replacement_history is the summary the compactor
			// injects into the fresh window; attribute it there.
			tracker.observeCompacted(rowLen)
		case "event_msg":
			if parsed && len(row.Payload) > 0 {
				var p rollupTypedPayload
				if json.Unmarshal(row.Payload, &p) == nil {
					payloadType = p.Type
				}
			}
			switch payloadType {
			case "task_started":
				turn++
			case "context_compacted":
				fire("context_compacted", ts)
			case "token_count":
				if !parsed || len(row.Payload) == 0 {
					continue
				}
				var tp rollupTokenPayload
				if json.Unmarshal(row.Payload, &tp) != nil {
					continue
				}
				rep.TokenSamples++
				if w := tp.Info.ModelContextWindow; w > 0 {
					rep.ContextWindow = w
				}
				if c := tp.Info.TotalTokenUsage.InputTokens; c > rep.CumulativeInputTokens {
					rep.CumulativeInputTokens = c
				}
				resident := tp.Info.LastTokenUsage.InputTokens
				if resident <= 0 {
					// The zero row Codex emits between the fire pair. Binding a fire to
					// it would report post-fire resident as 0; skip it entirely.
					continue
				}
				rep.FinalResidentTokens = resident
				if resident > rep.PeakResidentTokens {
					rep.PeakResidentTokens = resident
				}
				lastNonZero, lastNZTurn, lastNZTime, haveNonZero = resident, turn, ts, true
				resolvePending(&rep, pending, resident, turn, ts)
				tracker.observeSample(resident, resident, tp.Info.LastTokenUsage.CachedInputTokens, ts, turn, toolCalls)
			}
		}
	}

	rep.Turns = turn
	rep.ToolCalls = toolCalls
	rep.FireCount = len(rep.Fires)
	tracker.finalize(&rep)
	finalizeCompactReport(&rep)
	return rep, replayStatOf(replay), nil
}

// replayStatOf reports the replay's score, counting this rollout, or the zero stat when the
// replay was never armed.
func replayStatOf(rp *regrowthReplay) RegrowthReplayStat {
	if rp == nil {
		return RegrowthReplayStat{}
	}
	st := rp.stat
	st.Rollouts = 1
	return st
}

// recordFire admits a fire event, folding the compacted/context_compacted PAIR into a
// single fire. A second event of a DIFFERENT kind inside CompactPairWindow is the
// known pair — counted once. A second event of the SAME kind inside the window is a
// genuine duplicate and is flagged on the fire it repeats.
func recordFire(
	rep CompactSessionReport, pending []*pendingFire,
	kind string, ts time.Time, turn int,
	lastNonZero, lastNZTurn int, lastNZTime time.Time, haveNonZero bool,
	lastFireKind string, lastFireTime time.Time, haveFire bool,
) (CompactSessionReport, []*pendingFire, string, time.Time, bool) {

	if haveFire && withinPairWindow(lastFireTime, ts) {
		if kind == lastFireKind {
			rep.DuplicateFires++
			if n := len(rep.Fires); n > 0 {
				rep.Fires[n-1].Anomalies = appendUnique(rep.Fires[n-1].Anomalies, AnomalyDuplicateFireEvent)
			}
		} else {
			// The expected pair's second half: same fire, already counted.
			rep.PairedEvents++
		}
		return rep, pending, kind, ts, true
	}

	f := CompactFire{
		Index:         len(rep.Fires),
		At:            ts,
		Turn:          turn,
		ContextWindow: rep.ContextWindow,
	}
	if haveNonZero {
		f.PreTokens = lastNonZero
		if rep.ContextWindow > 0 {
			f.CeilingRatio = round4(float64(lastNonZero) / float64(rep.ContextWindow))
		}
		if turn-lastNZTurn > CompactAdjacencyTurns {
			f.Confidence = CompactConfidenceLow
			f.Reason = CompactReasonAdjacencyAmbiguous
		}
		_ = lastNZTime
	} else {
		f.Anomalies = appendUnique(f.Anomalies, AnomalyMissingPreWitness)
		f.Confidence = CompactConfidenceNone
		f.Reason = CompactReasonTelemetryMissing
	}
	rep.Fires = append(rep.Fires, f)
	pending = append(pending, &pendingFire{idx: len(rep.Fires) - 1})
	return rep, pending, kind, ts, true
}

func withinPairWindow(prev, cur time.Time) bool {
	if prev.IsZero() || cur.IsZero() {
		return false // no clock: treat as distinct rather than silently merging fires
	}
	d := cur.Sub(prev)
	if d < 0 {
		d = -d
	}
	return d <= CompactPairWindow
}

// resolvePending binds each open fire to the first non-zero sample after it (its post
// witness), then keeps watching for the rebound crossing.
func resolvePending(rep *CompactSessionReport, pending []*pendingFire, resident, turn int, ts time.Time) {
	for _, p := range pending {
		f := &rep.Fires[p.idx]
		if !p.haveePost {
			p.haveePost = true
			f.PostTokens = resident
			if f.PreTokens > 0 {
				f.Shed = f.PreTokens - resident
				f.ResidualRatio = round4(float64(resident) / float64(f.PreTokens))
			}
			if turn-f.Turn > CompactAdjacencyTurns {
				f.Confidence = CompactConfidenceLow
				f.Reason = CompactReasonAdjacencyAmbiguous
			}
			continue
		}
		if p.rebounded || f.PreTokens <= 0 {
			continue
		}
		if float64(resident) >= CompactReboundFraction*float64(f.PreTokens) {
			p.rebounded = true
			f.ReboundTurns = turn - f.Turn
			if !ts.IsZero() && !f.At.IsZero() {
				f.ReboundSeconds = round4(ts.Sub(f.At).Seconds())
			}
		}
	}
}

// finalizeCompactReport classifies every fire and lands the session verdict. It runs
// after the stream so it can see peak resident and every fire's witnesses.
func finalizeCompactReport(rep *CompactSessionReport) {
	for i := range rep.Fires {
		f := &rep.Fires[i]
		if f.ContextWindow == 0 {
			f.ContextWindow = rep.ContextWindow
			if f.PreTokens > 0 && rep.ContextWindow > 0 {
				f.CeilingRatio = round4(float64(f.PreTokens) / float64(rep.ContextWindow))
			}
		}
		if f.PreTokens <= 0 {
			f.Anomalies = appendUnique(f.Anomalies, AnomalyMissingPreWitness)
		}
		if f.PostTokens <= 0 {
			f.Anomalies = appendUnique(f.Anomalies, AnomalyMissingPostWitness)
			f.Confidence = CompactConfidenceNone
			f.Reason = CompactReasonTelemetryMissing
		}
		if f.CeilingRatio >= CompactCeilingLateFraction {
			f.Anomalies = appendUnique(f.Anomalies, AnomalyLateFire)
		}
		if f.PreTokens > 0 && f.PostTokens > 0 {
			if f.PostTokens >= f.PreTokens {
				// Fired but did not reduce resident context. Reported with the
				// confidence of its binding: a far-away witness is an adjacency
				// artifact candidate, not a proven failure.
				f.Anomalies = appendUnique(f.Anomalies, AnomalyIneffectiveFire)
			} else if f.ResidualRatio > CompactOversizedResidualRatio {
				f.Anomalies = appendUnique(f.Anomalies, AnomalyOversizedResidual)
			}
			if f.ReboundTurns > 0 && f.ReboundTurns <= CompactFastReboundTurns {
				f.Anomalies = appendUnique(f.Anomalies, AnomalyFastRebound)
			}
		}
		if f.Confidence == "" {
			f.Confidence = CompactConfidenceHigh
			f.Reason = CompactReasonOK
		}
		sort.Strings(f.Anomalies)
		rep.Anomalies = appendUnique(rep.Anomalies, f.Anomalies...)
	}

	atCeiling := rep.ContextWindow > 0 &&
		float64(rep.PeakResidentTokens) >= CompactCeilingApproachFraction*float64(rep.ContextWindow)

	// Wedge detection: count measured fires whose POST-fire resident stayed above the late-fire
	// ceiling. When at least CompactWedgeMinFires did, the session fired repeatedly and never came
	// off the ceiling — the oversized-single-item wedge. Requires a known window (else the ceiling
	// ratio is undefined) and post witnesses (an unmeasured fire cannot prove residency).
	wedgedFires := 0
	if rep.ContextWindow > 0 {
		ceiling := int(CompactCeilingLateFraction * float64(rep.ContextWindow))
		for i := range rep.Fires {
			if rep.Fires[i].PostTokens >= ceiling {
				wedgedFires++
			}
		}
	}
	wedged := wedgedFires >= CompactWedgeMinFires
	if wedged {
		rep.Anomalies = appendUnique(rep.Anomalies, AnomalyWedgedAtCeiling)
	}

	switch {
	case rep.TokenSamples == 0:
		rep.Verdict = VerdictTelemetryMissing
	case len(rep.Fires) == 0 && atCeiling:
		rep.Anomalies = appendUnique(rep.Anomalies, AnomalyNoFireAboveCeiling)
		rep.Verdict = VerdictNoFireAtCeiling
	case len(rep.Fires) == 0:
		// Bounded without ever needing to fire. Not an anomaly, however big the file.
		rep.Verdict = VerdictNoFireBounded
	case wedged:
		// Fired repeatedly and still pinned at the ceiling — the worst FIRED outcome.
		rep.Verdict = VerdictWedgedAtCeiling
	case len(rep.Anomalies) > 0:
		rep.Verdict = VerdictFiredWithAnomaly
	default:
		rep.Verdict = VerdictFiredAndHeld
	}
	sort.Strings(rep.Anomalies)
}

// AggregateCompactReports rolls per-session reports up to the fleet answer. Medians
// (not means) are the headline: fire sizes are heavy-tailed, so a mean is dragged by a
// handful of enormous sessions.
func AggregateCompactReports(reports []CompactSessionReport) CompactAggregate {
	agg := CompactAggregate{
		AnomalyCounts: map[string]int{},
		VerdictCounts: map[string]int{},
	}
	var pre, post, shed []int
	var residual []float64
	for _, r := range reports {
		agg.Sessions++
		agg.Bytes += r.Bytes
		agg.Fires += len(r.Fires)
		if len(r.Fires) > 0 {
			agg.CompactedSessions++
		}
		agg.VerdictCounts[r.Verdict]++
		for _, a := range r.Anomalies {
			agg.AnomalyCounts[a]++
		}
		for _, f := range r.Fires {
			if f.PreTokens > 0 && f.PostTokens > 0 {
				agg.MeasuredFires++
				pre = append(pre, f.PreTokens)
				post = append(post, f.PostTokens)
				shed = append(shed, f.Shed)
				residual = append(residual, f.ResidualRatio)
			}
		}
	}
	agg.MedianPreTokens = medianInt(pre)
	agg.MedianPostTokens = medianInt(post)
	agg.MedianShedTokens = medianInt(shed)
	agg.MedianResidualRatio = round4(medianFloat(residual))
	agg.Regrowth = rollupCompactRegrowth(reports)
	return agg
}

func medianInt(v []int) int {
	if len(v) == 0 {
		return 0
	}
	s := append([]int(nil), v...)
	sort.Ints(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func medianFloat(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func round4(f float64) float64 {
	return float64(int64(f*10000+0.5)) / 10000
}

func appendUnique(dst []string, vals ...string) []string {
	for _, v := range vals {
		found := false
		for _, d := range dst {
			if d == v {
				found = true
				break
			}
		}
		if !found {
			dst = append(dst, v)
		}
	}
	return dst
}

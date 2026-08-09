package sessionaudit

// deferrefault.go — post-run cold-tool-defer re-request audit (#3625, epic #3569
// cache-verify): did the deferred tool schemas fault BACK IN?
//
// THE GAP THIS CLOSES. The 10x floor lever (#3232, gateway/messages_tooldefer.go) marks
// every allowed-but-COLD tool `defer_loading:true` and injects one `tool_search_tool`, so
// the provider loads only the hot core into context and faults a cold schema in on demand
// when the model searches for it. That trade is only a win if the faults stay RARE. If the
// model keeps searching for tools it already pulled in, each fault re-materializes a schema
// that was already resident: context churns, the cache_control prefix shifts, and the
// session cache stops holding — the defer is DEFEATED while the shed counter still reports
// a happy cold count. Nothing after a run measured that, which is the gap here.
//
// WHAT "REFAULT" MEANS, PRECISELY. A fault event is one tool-search call. A
// MATERIALIZATION is one deferred schema that call pulled back into context. A REFAULT is a
// materialization of a schema ALREADY resident earlier in the same session — the redundant
// re-load. `defer_refault_rate` is refaults / materializations: the share of all schema
// materializations that bought nothing because the schema was already there. That ratio, not
// the raw fault count, is what defeats the defer — a session may fault twenty DISTINCT cold
// schemas in and still cache perfectly, while a session that faults the same three schemas
// back in over and over churns the prefix on every turn.
//
// Like the Behavior/Confusion/posture lenses this stays pure and deterministic: stdlib-only,
// no clock of its own (the caller supplies `generated`), stable ordering — same fault
// sequence in, same audit row out. Live alarming and changing the hot-tool set selection are
// explicitly out of scope (#3625): this is a post-run reading, not a control loop.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// deferRefaultSchema versions the audit-row wire shape so a reader can detect a future
// field addition without guessing.
const deferRefaultSchema = "fak.session_audit.defer_refault.v1"

// DefaultDeferRefaultThreshold is the refault rate above which the defer is judged
// DEFEATED. At 0.5 more than half of every schema materialization in the session was a
// redundant re-load of a schema already resident — the defer is doing net-negative work,
// paying search round-trips AND churning the cache prefix to deliver context the model
// already had. Below that a session is still buying real sheddage from the cold tail.
const DefaultDeferRefaultThreshold = 0.5

// MinDeferRefaultSample is the smallest number of materializations that can carry a
// DEFER_DEFEATED verdict. A session with one search and one repeat is a 1.0 rate on a
// sample of two, which says nothing about whether the defer holds; the finding names the
// hold rather than flagging noise.
const MinDeferRefaultSample = 4

// deferSearchToolNames are the tool spellings that count as a fault EVENT. The gateway
// injects Anthropic's standard `tool_search_tool` (gateway.toolSearchToolName, mirrored
// here by value to keep this package stdlib-only and import-free); `ToolSearch` is the
// Claude Code harness spelling of the same seam, already known to this package's
// ReadOnlyTools table. Both are the model asking for a deferred schema.
var deferSearchToolNames = map[string]bool{
	"tool_search_tool": true,
	"ToolSearch":       true,
}

// DeferRefaultVerdict is the closed-vocabulary outcome of reading a session's deferred-tool
// fault sequence.
type DeferRefaultVerdict string

const (
	// DeferOK: the deferred tail held. Either no schema ever faulted back in, or the
	// re-request rate stayed at or below the threshold, so the faults were mostly first-time
	// materializations the defer is designed to serve.
	DeferOK DeferRefaultVerdict = "DEFER_OK"
	// DeferDefeated: the re-request rate exceeded the threshold on a large enough sample —
	// the model kept faulting schemas it already had back into context, so the defer churned
	// the prefix instead of shedding it.
	DeferDefeated DeferRefaultVerdict = "DEFER_DEFEATED"
)

// DeferFault is ONE tool-search fault event: the deferred tool schemas that one search call
// materialized back into context. Order is significant across a []DeferFault — a
// materialization only counts as a REFAULT relative to the faults before it — so the slice
// must be in transcript order. Tools may be empty for a search whose materialized names could
// not be attributed (see DeferFaultsFromTranscript); such a fault is counted as an event and
// held as Unattributed rather than guessed at.
type DeferFault struct {
	// CallID is the transcript block id of the search call, carried so a flagged session can
	// be walked back to the exact turn. Optional; the pure lens never reads it.
	CallID string   `json:"call_id,omitempty"`
	Tools  []string `json:"tools,omitempty"`
}

// DeferRefaultAudit is one dated per-session reading of the cold-tool-defer fault sequence:
// how many searches fired, how many schemas they materialized, how many of those were
// redundant re-loads, the resulting rate, and the DEFER_OK / DEFER_DEFEATED verdict.
// Generated stamps WHEN the pass ran so the row is a dated audit artifact.
type DeferRefaultAudit struct {
	Schema    string              `json:"schema"`
	Generated string              `json:"generated"`
	Session   string              `json:"session,omitempty"`
	Verdict   DeferRefaultVerdict `json:"verdict"`
	// Searches is the number of tool-search fault EVENTS (search calls).
	Searches int `json:"searches"`
	// Materializations is the number of deferred schemas those searches pulled back into
	// context, counting repeats — the denominator of the rate.
	Materializations int `json:"materializations"`
	// DistinctTools is how many DIFFERENT deferred tools faulted back in at all: the literal
	// "how many deferred tools were faulted back in" count, kept apart from the rate because a
	// wide-but-flat fault profile is healthy and a narrow-but-repeating one is not.
	DistinctTools int `json:"distinct_tools"`
	// Refaults is the number of materializations that re-loaded an already-resident schema —
	// the numerator of the rate.
	Refaults int `json:"refaults"`
	// Unattributed is the number of search events whose materialized tool names could not be
	// recovered from the transcript. They count as searches but contribute no materialization,
	// so they can only ever DILUTE the rate, never inflate it — the conservative direction for
	// a finding that accuses the defer of failing.
	Unattributed int `json:"unattributed,omitempty"`
	// RefaultRate is Refaults/Materializations, 0 when nothing materialized.
	RefaultRate float64 `json:"defer_refault_rate"`
	Threshold   float64 `json:"threshold"`
	// RefaultedTools are the schemas that faulted in more than once, sorted — the churning
	// set an operator would pin into the hot tool set first.
	RefaultedTools []string `json:"refaulted_tools,omitempty"`
	Finding        string   `json:"finding"`
}

// AuditDeferRefault reads a session's tool-search fault sequence and returns a dated
// DEFER_OK / DEFER_DEFEATED audit row. faults must be in transcript order. generated stamps
// the row (the caller supplies it, keeping this pure). threshold is the refault rate above
// which the defer is judged defeated; pass DefaultDeferRefaultThreshold for the standard
// reading, and any value <= 0 falls back to it so a zero-valued caller cannot accidentally
// flag every session.
func AuditDeferRefault(session string, faults []DeferFault, threshold float64, generated time.Time) DeferRefaultAudit {
	if threshold <= 0 {
		threshold = DefaultDeferRefaultThreshold
	}
	audit := DeferRefaultAudit{
		Schema:    deferRefaultSchema,
		Generated: generated.UTC().Format(time.RFC3339),
		Session:   session,
		Threshold: threshold,
	}
	// resident counts how many times each schema has been materialized SO FAR, which is what
	// makes the second and later materializations refaults. Insertion into this map is also
	// what defines DistinctTools.
	resident := map[string]int{}
	churned := map[string]struct{}{}
	for _, f := range faults {
		audit.Searches++
		named := 0
		for _, raw := range f.Tools {
			name := strings.TrimSpace(raw)
			if name == "" {
				continue
			}
			named++
			audit.Materializations++
			if resident[name] > 0 {
				audit.Refaults++
				churned[name] = struct{}{}
			}
			resident[name]++
		}
		if named == 0 {
			audit.Unattributed++
		}
	}
	audit.DistinctTools = len(resident)
	if audit.Materializations > 0 {
		audit.RefaultRate = float64(audit.Refaults) / float64(audit.Materializations)
	}
	audit.RefaultedTools = sortedKeys(churned)

	switch {
	case audit.Searches == 0:
		audit.Verdict = DeferOK
		audit.Finding = "DEFER_OK — no tool-search fault fired: every deferred schema stayed cold for the whole session, which is the defer working exactly as designed"
	case audit.Materializations == 0:
		audit.Verdict = DeferOK
		audit.Finding = fmt.Sprintf(
			"DEFER_OK — %d tool-search fault(s) fired but no materialized schema could be attributed, so no refault rate is claimed; the verdict is a HOLD on missing evidence, not a clean bill",
			audit.Searches)
	case audit.RefaultRate > threshold && audit.Materializations >= MinDeferRefaultSample:
		audit.Verdict = DeferDefeated
		audit.Finding = fmt.Sprintf(
			"DEFER_DEFEATED — %d of %d schema materializations re-loaded a tool already resident this session (defer_refault_rate=%.2f > %.2f): the deferred tail is churning back into context, so the cache prefix cannot hold and the defer is paying search round-trips for context the model already had. Churning: %s",
			audit.Refaults, audit.Materializations, audit.RefaultRate, threshold,
			strings.Join(audit.RefaultedTools, ", "))
	case audit.RefaultRate > threshold:
		audit.Verdict = DeferOK
		audit.Finding = fmt.Sprintf(
			"DEFER_OK — the refault rate is %.2f (> %.2f) but only %d schema materialization(s) were seen, under the %d-materialization floor a verdict needs; too small a sample to accuse the defer, so this is a HOLD",
			audit.RefaultRate, threshold, audit.Materializations, MinDeferRefaultSample)
	default:
		audit.Verdict = DeferOK
		audit.Finding = fmt.Sprintf(
			"DEFER_OK — %d fault(s) materialized %d schema(s) across %d distinct deferred tool(s) with %d redundant re-load(s) (defer_refault_rate=%.2f <= %.2f): the faults were mostly first-time materializations the defer exists to serve",
			audit.Searches, audit.Materializations, audit.DistinctTools, audit.Refaults,
			audit.RefaultRate, threshold)
	}
	return audit
}

// sortedKeys returns the sorted keys of a set, or nil when empty — the deterministic
// ordering the audit row's tool list depends on.
func sortedKeys(set map[string]struct{}) []string {
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// AuditDeferRefaultTranscript is the post-run pass over ONE completed transcript: it
// extracts the tool-search fault sequence and folds it into a dated per-session audit row.
// A read error is returned rather than reported as a clean DEFER_OK, so a missing transcript
// can never masquerade as a session whose defer held.
func AuditDeferRefaultTranscript(path string, threshold float64, generated time.Time) (DeferRefaultAudit, error) {
	faults, err := DeferFaultsFromTranscript(path)
	if err != nil {
		return DeferRefaultAudit{}, err
	}
	session := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	return AuditDeferRefault(session, faults, threshold, generated), nil
}

// DeferFaultsFromTranscript walks a completed JSONL transcript and returns its tool-search
// fault sequence in transcript order: one DeferFault per search call, carrying the deferred
// tool names that call's RESULT materialized.
//
// The pairing is by tool_use id, not by position: a search call appears as a `tool_use` block
// on an assistant record and its materialized schemas come back on a LATER `tool_result`
// block naming that id, with any number of unrelated records in between. A search whose
// result never arrives (an interrupted session) stays in the sequence as an unattributed
// fault — the model DID ask, and dropping the event would silently improve the rate.
//
// ASSUMPTION, stated because it is the one thing here that can go stale: the materialized
// tool names are read out of the result body by deferMaterializedTools, which understands the
// `<function>{...}</function>` envelope the tool-search result uses plus a direct JSON
// array of tool descriptors. If the provider changes that envelope, faults decode as
// unattributed — the rate is then withheld (a HOLD finding) rather than silently reported as
// healthy, which is the safe failure direction, but the audit stops being informative until
// the decoder is taught the new shape.
func DeferFaultsFromTranscript(path string) ([]DeferFault, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		faults []DeferFault
		// index maps a pending search call id to its slot in faults, so a result arriving many
		// records later still lands on the right event without disturbing the order.
		index = map[string]int{}
	)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 64*1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r transcriptRecord
		if err := json.Unmarshal(line, &r); err != nil {
			continue
		}
		if len(r.Message.Content) == 0 {
			continue
		}
		var blocks []contentBlock
		if json.Unmarshal(r.Message.Content, &blocks) != nil {
			continue
		}
		for _, b := range blocks {
			switch b.Type {
			case "tool_use":
				if !deferSearchToolNames[b.Name] {
					continue
				}
				// A repeated block id is a split/duplicated assistant record replaying a call
				// already counted, not a second search — see noteToolUses for the same hazard.
				if b.ID != "" {
					if _, dup := index[b.ID]; dup {
						continue
					}
					index[b.ID] = len(faults)
				}
				faults = append(faults, DeferFault{CallID: b.ID})
			case "tool_result":
				slot, ok := index[b.ToolUseID]
				if !ok {
					continue
				}
				// An errored search materialized nothing; leaving Tools empty keeps it an
				// unattributed event instead of crediting it with a fault-in that never landed.
				if b.IsError {
					continue
				}
				faults[slot].Tools = append(faults[slot].Tools, deferMaterializedTools(b.Content)...)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return faults, nil
}

// deferMaterializedTools recovers the deferred tool names one tool-search result pulled back
// into context. It is deliberately tolerant, in the spirit of
// ManagedCacheClaimFromDebugVars: every shape it does not recognize yields NO names, which
// the caller records as an unattributed fault rather than a guess.
//
// Recognized shapes, in order:
//   - a content array of blocks, each either a text block carrying `<function>{…}</function>`
//     envelopes or a descriptor block with its own "name";
//   - a bare JSON string carrying those same envelopes;
//   - a direct JSON array of tool descriptors ([{"name":…}, …]).
func deferMaterializedTools(raw json.RawMessage) []string {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	// Shape 1/3: an array of blocks or descriptors.
	var arr []struct {
		Type string          `json:"type"`
		Name string          `json:"name"`
		Text json.RawMessage `json:"text"`
	}
	if json.Unmarshal(raw, &arr) == nil {
		var out []string
		for _, el := range arr {
			if el.Name != "" {
				out = append(out, el.Name)
			}
			out = append(out, deferToolNamesInText(rawText(el.Text))...)
		}
		return out
	}
	// Shape 2: a bare JSON string.
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return deferToolNamesInText(s)
	}
	return nil
}

// rawText unwraps a JSON string field that may legitimately arrive as a string, as an array,
// or absent. Only the string form carries prose; anything else yields "".
func rawText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return ""
}

// deferToolNamesInText pulls tool names out of the `<function>{…}</function>` envelopes a
// tool-search result body uses. Each envelope's payload is decoded as JSON and its "name"
// taken; a payload that does not decode is skipped rather than pattern-matched, so a prose
// mention of the word "name" can never be mistaken for a materialized schema.
func deferToolNamesInText(text string) []string {
	const openTag, closeTag = "<function>", "</function>"
	var out []string
	for {
		i := strings.Index(text, openTag)
		if i < 0 {
			return out
		}
		text = text[i+len(openTag):]
		j := strings.Index(text, closeTag)
		if j < 0 {
			return out
		}
		payload := text[:j]
		text = text[j+len(closeTag):]
		var desc struct {
			Name string `json:"name"`
		}
		if json.Unmarshal([]byte(payload), &desc) == nil && desc.Name != "" {
			out = append(out, desc.Name)
		}
	}
}

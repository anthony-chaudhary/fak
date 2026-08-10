package trajectory

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// THE SCRUBBED-CHAT ADAPTER — the one supported way a conversation/chat export enters
// the trajectory plane.
//
// A raw agent chat log is the highest-risk corpus in the building: it is mostly prompts,
// user utterances, pasted files, and whatever secret was in scope that day. So the
// adapter does not accept one. It accepts a SCRUBBED export — a caller-produced
// projection that has already dropped message bodies — and it enforces that boundary
// STRUCTURALLY rather than by asking the producer nicely:
//
//   - The decode structs below have no content field at all. A `text` / `content` /
//     `message` key in the input is discarded by encoding/json before this package can
//     see it, so there is no code path that could copy it into a Turn.
//   - The only free-form strings admitted are a conversation id, a tool name and a
//     refusal reason, and `reason` must be label-shaped (A-Z, 0-9, _) — a sentence,
//     a path, or a pasted secret is refused as malformed, not stored.
//   - Everything else is a closed enum, a digest, or a number.
//
// Anything that is not exactly ScrubbedChatFormat is refused as unsupported rather than
// parsed on a best-effort basis: guessing at an unknown chat schema is how raw content
// gets ingested by accident.
//
// WIRE FORMAT (fak.scrubbed-chat/1), a single JSON document:
//
//	{
//	  "format": "fak.scrubbed-chat/1",
//	  "conversations": [
//	    {"id": "conv-1", "entries": [
//	      {"role": "user"},
//	      {"role": "assistant", "tool": "Grep", "status": "ok",
//	       "args_digest": "sha256:ab…", "result_digest": "sha256:cd…",
//	       "tokens": 120, "bytes": 4096, "cache_hit": false,
//	       "ts_unix_nano": 1786326019106107600},
//	      {"role": "assistant", "tool": "Bash", "status": "denied", "reason": "POLICY_BLOCK"}
//	    ]}
//	  ]
//	}
//
// An entry with no `tool` carries no agent action; it is counted as a skipped message
// and never becomes a Turn (the same rule the Recorder applies to non-decision events).

// ScrubbedChatFormat is the only chat/conversation export id this package accepts.
const ScrubbedChatFormat = "fak.scrubbed-chat/1"

// FormatErrorKind is the closed set of adapter refusal classes.
type FormatErrorKind string

const (
	// FormatMalformed: the document claims the supported format but does not satisfy it.
	FormatMalformed FormatErrorKind = "MALFORMED_EXPORT"
	// FormatUnsupported: the document is some other (or unnamed) export schema.
	FormatUnsupported FormatErrorKind = "UNSUPPORTED_FORMAT"
)

// FormatError is the adapter's typed refusal. Detail names the offending FIELD and the
// position; it never echoes a rejected value, because the value is exactly the thing
// that might be content.
type FormatError struct {
	Kind         FormatErrorKind
	Detail       string
	Conversation string // conversation id when known
	Entry        int    // 1-based entry index when known
}

func (e *FormatError) Error() string {
	var b strings.Builder
	b.WriteString(string(e.Kind))
	if e.Detail != "" {
		fmt.Fprintf(&b, ": %s", e.Detail)
	}
	if e.Conversation != "" {
		fmt.Fprintf(&b, " (conversation %q", e.Conversation)
		if e.Entry > 0 {
			fmt.Fprintf(&b, ", entry %d", e.Entry)
		}
		b.WriteString(")")
	}
	return b.String()
}

// Is lets callers match on the CLASS with errors.Is against the sentinels below,
// while a concrete error still carries the field-level detail via errors.As.
func (e *FormatError) Is(target error) bool {
	t, ok := target.(*FormatError)
	return ok && t.Kind == e.Kind
}

// Sentinels for errors.Is. Match the class; use errors.As for the detail.
var (
	ErrMalformedExport   error = &FormatError{Kind: FormatMalformed}
	ErrUnsupportedFormat error = &FormatError{Kind: FormatUnsupported}
)

// chatExport / chatConversation / chatEntry are the decode surface. Their field sets
// ARE the privacy contract: there is no member that can hold a prompt, a message body,
// or a tool-argument body, so an unscrubbed export loses its content at json.Unmarshal.
type chatExport struct {
	Format        string             `json:"format"`
	Conversations []chatConversation `json:"conversations"`
}

type chatConversation struct {
	ID      string      `json:"id"`
	Entries []chatEntry `json:"entries"`
}

type chatEntry struct {
	Role         string `json:"role"`
	Tool         string `json:"tool"`
	Status       string `json:"status"`
	Reason       string `json:"reason"`
	ArgsDigest   string `json:"args_digest"`
	ResultDigest string `json:"result_digest"`
	Tokens       int    `json:"tokens"`
	Bytes        int64  `json:"bytes"`
	CacheHit     bool   `json:"cache_hit"`
	TSUnixNano   int64  `json:"ts_unix_nano"`
}

// chatStatus maps the closed status vocabulary to the same verdict labels verdictName
// emits, so a mined chat export and a mined kernel export share one vocabulary.
var chatStatus = map[string]string{
	"":            "ALLOW",
	"ok":          "ALLOW",
	"allowed":     "ALLOW",
	"denied":      "DENY",
	"quarantined": "QUARANTINE",
	"witness":     "WITNESS",
}

// ImportScrubbedChat reads a fak.scrubbed-chat/1 document into a Recorder whose traces
// are the conversations, ready for [Recorder.Mine]. Returns the number of turns
// ingested. Every refusal is a *FormatError: match the class with errors.Is against
// [ErrMalformedExport] / [ErrUnsupportedFormat], or errors.As for the field detail.
//
// Unlike [ImportFrom] — which tolerates a torn line because it reads a corpus this
// process itself wrote — this adapter fails CLOSED on the first defect. A partially
// understood third-party export is exactly the case where "keep going" turns into
// "ingest something you did not model".
func ImportScrubbedChat(rd io.Reader) (*Recorder, int, error) {
	raw, err := io.ReadAll(rd)
	if err != nil {
		return nil, 0, &FormatError{Kind: FormatMalformed, Detail: "read export: " + err.Error()}
	}
	var doc chatExport
	if err := json.Unmarshal(raw, &doc); err != nil {
		// The decoder's message can quote input bytes; report the position class only.
		return nil, 0, &FormatError{Kind: FormatMalformed, Detail: "export is not a JSON object"}
	}
	if doc.Format != ScrubbedChatFormat {
		detail := "field \"format\" names an export schema this build does not support"
		if doc.Format == "" {
			detail = "field \"format\" is missing; only " + ScrubbedChatFormat + " is supported"
		}
		return nil, 0, &FormatError{Kind: FormatUnsupported, Detail: detail}
	}
	if len(doc.Conversations) == 0 {
		return nil, 0, &FormatError{Kind: FormatMalformed, Detail: "field \"conversations\" is empty"}
	}

	r := New().MaxTraces(0).MaxPerTrace(0)
	total := 0
	for _, conv := range doc.Conversations {
		if strings.TrimSpace(conv.ID) == "" {
			return nil, 0, &FormatError{Kind: FormatMalformed, Detail: "conversation is missing \"id\""}
		}
		if _, dup := r.byID[conv.ID]; dup {
			return nil, 0, &FormatError{Kind: FormatMalformed, Detail: "duplicate conversation \"id\"", Conversation: conv.ID}
		}
		turns := make([]Turn, 0, len(conv.Entries))
		for i, e := range conv.Entries {
			if strings.TrimSpace(e.Tool) == "" {
				continue // a message entry: counted as skipped, never a Turn
			}
			verdict, ok := chatStatus[strings.ToLower(strings.TrimSpace(e.Status))]
			if !ok {
				return nil, 0, &FormatError{Kind: FormatMalformed,
					Detail:       "field \"status\" is outside the closed vocabulary (ok|denied|quarantined|witness)",
					Conversation: conv.ID, Entry: i + 1}
			}
			if !labelShaped(e.Reason) {
				return nil, 0, &FormatError{Kind: FormatMalformed,
					Detail:       "field \"reason\" must be an upper-case label (A-Z, 0-9, _), not free text",
					Conversation: conv.ID, Entry: i + 1}
			}
			turns = append(turns, Turn{
				TraceID:       conv.ID,
				Seq:           len(turns) + 1,
				TSUnixNano:    e.TSUnixNano,
				Tool:          e.Tool,
				Verdict:       verdict,
				Reason:        e.Reason,
				ArgsDigest:    e.ArgsDigest,
				ResultDigest:  e.ResultDigest,
				TokenEstimate: e.Tokens,
				Bytes:         e.Bytes,
				CacheHit:      e.CacheHit,
			})
		}
		// An empty conversation still registers as a trace so the miner can ABSTAIN on
		// it out loud rather than silently dropping a conversation the caller supplied.
		r.order = append(r.order, conv.ID)
		r.byID[conv.ID] = turns
		total += len(turns)
	}
	return r, total, nil
}

// labelShaped reports whether s is empty or a closed-vocabulary-looking label. It is the
// guard that keeps a "reason" field from becoming a free-text content channel.
func labelShaped(s string) bool {
	for _, c := range s {
		switch {
		case c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_':
		default:
			return false
		}
	}
	return true
}

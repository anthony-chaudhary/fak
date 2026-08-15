package agent

// anthropic_elide_stale.go — read-lifecycle STALE elision, a size-INDEPENDENT sibling of the
// oversized-tool_result elision in anthropic_elide.go.
//
// Oversized elision shrinks a body because it is BIG. This transform shrinks one because it is
// SUPERSEDED: a Read tool_result for a file that a LATER in-session Edit/Write/MultiEdit/
// NotebookEdit turn has already changed is stale — the model is looking at bytes that no longer
// reflect the file on disk, and (unlike a scrolled-past command output) re-reading the pre-edit
// snapshot is actively misleading. A coding agent (the flagship `fak guard -- claude` case) reads
// a file, edits it, and never needs the pre-edit read again — yet that whole read sits in the
// window every subsequent turn, un-cached middle weight the model can be actively misled by.
//
// The predicate is STALENESS, not byte size: a two-line stale read is elided; a huge fresh read is
// not. "Later" is document order over the message list — a Read tool_result at message i is stale
// iff the same (normalized) file_path is targeted by an edit tool_use at some message j > i. The
// pairing from a tool_result back to its originating Read is by tool_use_id (tool_result.tool_use_id
// == the Read tool_use.id); the file path is the tool_use input.file_path (or notebook_path).
//
// The cache guarantee, and the way it is enforced, is IDENTICAL to anthropic_elide.go and shares
// its machinery verbatim: only a tool_result that lives STRICTLY AFTER the protected prefix (the
// FIRST cache_control breakpoint), is OUTSIDE the recent working-set window (the last
// elideRecentKeepMsgs messages), and whose message carries NO cache_control reachable by the
// shrinker may be rewritten. The rewrite splices on the ORIGINAL bytes (applySpliceEdits memcpies
// every non-edited byte, the whole head prefix included) and is then re-proven by verifySplicedBody
// (re-decodes, head prefix bytes byte-identical, tail survives, no empty-block shape). On ANY
// ambiguity the function returns its input UNCHANGED.
//
// Unlike oversized elision, the elided body is not thrown away: the FULL original tool_result text
// is returned on the outcome as a StaleRestore so the gateway can stash it behind its content
// address (originatingTaskDigestID) and fak_context_restore can page it back in verbatim. The
// in-band marker carries that id=<hex> handle so the model knows both WHY the read was dropped and
// HOW to recover it.
//
// Like elision this is a REQUEST-side transform only — it touches the bytes sent upstream, never the
// decoded req.Messages the kernel adjudicates — and it is lossy (the pre-edit snapshot is replaced
// by a restorable marker), so it is off by default and gated on the same working-set guard.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Stale-elision bail-reason vocabulary — the closed set of identity-return causes, mirrored on
// StaleElideOutcome so a caller can label a metric and an operator can see WHY the pass did nothing
// (silence must not read as success). StaleReasonNone means the body was rewritten.
const (
	StaleReasonNone       = ""                // FIRED: a rewrite happened (Elided/ShedBytes/Restores meaningful)
	StaleReasonOff        = "off"             // empty body — nothing to scan
	StaleReasonNonJSON    = "non_json"        // body is not a JSON object
	StaleReasonNoMsgsKey  = "no_messages_key" // no "messages" key
	StaleReasonTooFewMsgs = "too_few_msgs"    // < 2 messages — nothing to scan (benign, high-volume)
	// StaleReasonDecodeFailed is the STRUCTURAL messages[] failure, split out of too_few_msgs so a
	// present-but-undecodable `messages` value is not counted in the benign short-request bucket.
	// Mirrors CompactReasonDecodeFailed; same wire token, so the three subsystems agree.
	StaleReasonDecodeFailed    = "decode_failed"
	StaleReasonNoBreakpoint    = "no_breakpoint"    // no cache_control anchor — cannot know the cache boundary
	StaleReasonNoStaleReads    = "no_stale_reads"   // no Read superseded by a later edit in the eligible band
	StaleReasonSpliceFailed    = "splice_failed"    // the edits overlapped or fell out of range
	StaleReasonRedecodeFail    = "redecode_failed"  // the spliced body failed to re-decode
	StaleReasonPrefixMismatch  = "prefix_mismatch"  // the splice changed the protected prefix bytes
	StaleReasonMalformedResult = "malformed_result" // the spliced body lands an Anthropic-400 empty-block shape
)

// StaleRestore is one stashable original: the sha256-hex content-address embedded in the marker (the
// SAME scheme compaction uses, computable here with originatingTaskDigestID), the verbatim original
// tool_result text a fak_context_restore(id) pages back in, and a bounded orientation excerpt.
type StaleRestore struct {
	ID      string
	Bytes   []byte
	Excerpt string
}

// StaleElideOutcome is the observable verdict of one stale-elision attempt. Reason==StaleReasonNone
// means FIRED — Elided (stale reads rewritten), ShedBytes (raw bytes removed), and Restores (one per
// rewritten read, for the gateway to stash) are then meaningful. Any other Reason means the body was
// returned unchanged (identity) and the counts are 0 / Restores nil.
type StaleElideOutcome struct {
	Reason     string
	Elided     int
	ShedBytes  int
	ShedTokens int
	Restores   []StaleRestore
}

// ElideStaleReads is the byte-only wrapper: it returns the rewritten body and discards the restore
// payloads. Callers that want fak_context_restore to be able to recover the originals must use
// ElideStaleReadsWithOutcome and stash outcome.Restores; the live gateway path does exactly that.
func ElideStaleReads(raw []byte) []byte {
	out, _ := ElideStaleReadsWithOutcome(raw)
	return out
}

// ElideStaleReadsWithOutcome replaces every Read tool_result superseded by a later same-file edit
// (and lying in the eligible band) with a compact, restorable marker, byte-splicing on the original
// bytes so the cached head prefix is preserved verbatim. It returns raw UNCHANGED whenever it cannot
// prove the rewrite is both cache-safe and well-formed. outcome.Restores carries the full original
// text of each rewritten read, content-addressed by the id embedded in its marker.
func ElideStaleReadsWithOutcome(raw []byte) ([]byte, StaleElideOutcome) {
	if len(raw) == 0 {
		return raw, StaleElideOutcome{Reason: StaleReasonOff}
	}
	// Decode + protected-prefix anchor — the SAME rule as oversized elision (first cache_control
	// breakpoint, deep detector, same system-only fallback), so both passes run the one shared
	// preamble; only the Reason words are this pass's own.
	elems, spans, pfxEnd, reason := anchorElidableMessages(raw, elideAnchorReasons{
		nonJSON:      StaleReasonNonJSON,
		noMsgsKey:    StaleReasonNoMsgsKey,
		decodeFailed: StaleReasonDecodeFailed,
		tooFewMsgs:   StaleReasonTooFewMsgs,
		noBreakpoint: StaleReasonNoBreakpoint,
	})
	if reason != "" {
		return raw, StaleElideOutcome{Reason: reason}
	}

	// Classify the read/edit lifecycle across the WHOLE message list: staleness is a cross-turn
	// property (an edit anywhere after a read supersedes it), so the index scans every message even
	// though only the eligible band is ever rewritten below.
	idx := classifyReadLifecycle(elems)
	if len(idx.readPathByToolUse) == 0 || len(idx.maxEditIdxByPath) == 0 {
		return raw, StaleElideOutcome{Reason: StaleReasonNoStaleReads}
	}

	// The eligible band: strictly after the protected prefix, before the recent working-set window,
	// never a message with cache_control reachable by the shrinker. Rewriting here keeps the head
	// prefix byte-identical (proven below); later breakpoints cascade-burst, as documented.
	var edits []spliceEdit
	var restores []StaleRestore
	shed := 0
	eachElidableMessage(elems, spans, pfxEnd, func(start, i int, elem json.RawMessage) {
		es, rs, sh := idx.collectStaleReadEdits(start, i, elem)
		edits = append(edits, es...)
		restores = append(restores, rs...)
		shed += sh
	})
	if len(edits) == 0 {
		return raw, StaleElideOutcome{Reason: StaleReasonNoStaleReads}
	}

	out, ok := applySpliceEdits(raw, edits)
	if !ok {
		return raw, StaleElideOutcome{Reason: StaleReasonSpliceFailed}
	}
	// Prove it with the shared post-splice check (re-decode + protected-prefix byte-equality + tail
	// survival + Anthropic semantic well-formedness). The marker is always a non-empty JSON string,
	// so it cannot introduce the empty-block shape; any non-OK verdict is a splice bug → identity.
	switch verifySplicedBody(raw, out, spans, pfxEnd) {
	case spliceVerdictRedecodeFail:
		return raw, StaleElideOutcome{Reason: StaleReasonRedecodeFail}
	case spliceVerdictPrefixMismatch:
		return raw, StaleElideOutcome{Reason: StaleReasonPrefixMismatch}
	case spliceVerdictMalformedResult:
		return raw, StaleElideOutcome{Reason: StaleReasonMalformedResult}
	}
	shedTokens := staleElideTokenDelta(elems, out)
	return out, StaleElideOutcome{
		Reason: StaleReasonNone, Elided: len(edits), ShedBytes: shed,
		ShedTokens: shedTokens, Restores: restores,
	}
}

// staleElideTokenDelta uses the same image-aware house estimator as compaction. Provider tokenizers
// are not available on this local rewrite seam, so this receipt is explicitly token-equivalent; it
// is nevertheless computed from the exact before/after wire elements rather than from display text.
func staleElideTokenDelta(before []json.RawMessage, rawAfter []byte) int {
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(rawAfter, &body) != nil || len(body.Messages) != len(before) {
		return 0
	}
	beforeTokens, afterTokens := 0, 0
	for i := range before {
		beforeTokens += estimateElementTokens(before[i])
		afterTokens += estimateElementTokens(body.Messages[i])
	}
	if beforeTokens <= afterTokens {
		return 0
	}
	return beforeTokens - afterTokens
}

// staleEditToolNames is the closed set of tool names whose tool_use supersedes an earlier Read of the
// same file: the file-mutating Claude Code tools. A Bash write is deliberately excluded — its target
// is not a structured file_path we can match — keeping detection conservative (a missed stale read is
// merely un-elided; a false stale would only ever be a bounded, restorable loss).
var staleEditToolNames = map[string]bool{
	"Edit":         true,
	"Write":        true,
	"MultiEdit":    true,
	"NotebookEdit": true,
}

// readLifecycleIndex is the cross-turn read/edit map used to classify a Read tool_result as STALE:
// which tool_use_id was a Read of which (display) path, and the highest message index at which each
// (normalized) path was edited. A read at message i is stale iff maxEditIdxByPath[norm(path)] > i.
type readLifecycleIndex struct {
	readPathByToolUse map[string]string // Read tool_use id -> display file path (as the model sent it)
	maxEditIdxByPath  map[string]int    // normalized file path -> highest message index that edited it
}

// classifyReadLifecycle walks every messages[] element once (structure only — no byte offsets) and
// builds the read/edit index. tool_use blocks live in assistant turns; a Read records id->path, an
// edit tool records the message index under the normalized path. Malformed / non-array content is
// skipped, never fatal.
func classifyReadLifecycle(elems []json.RawMessage) readLifecycleIndex {
	idx := readLifecycleIndex{
		readPathByToolUse: map[string]string{},
		maxEditIdxByPath:  map[string]int{},
	}
	for i, el := range elems {
		var m struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(el, &m) != nil || len(m.Content) == 0 || m.Content[0] != '[' {
			continue
		}
		var blocks []json.RawMessage
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		for _, blk := range blocks {
			var b struct {
				Type  string `json:"type"`
				ID    string `json:"id"`
				Name  string `json:"name"`
				Input struct {
					FilePath     string `json:"file_path"`
					NotebookPath string `json:"notebook_path"`
				} `json:"input"`
			}
			if json.Unmarshal(blk, &b) != nil || b.Type != "tool_use" {
				continue
			}
			path := firstNonBlank(b.Input.FilePath, b.Input.NotebookPath)
			if strings.TrimSpace(path) == "" {
				continue
			}
			switch {
			case b.Name == "Read" && b.ID != "":
				idx.readPathByToolUse[b.ID] = path
			case staleEditToolNames[b.Name]:
				norm := normReadPath(path)
				if i > idx.maxEditIdxByPath[norm] {
					idx.maxEditIdxByPath[norm] = i
				}
			}
		}
	}
	return idx
}

// collectStaleReadEdits scans one user-turn element for tool_result blocks that (a) pair back to a
// Read tool_use via tool_use_id and (b) name a file edited in a LATER message than this one, and
// returns a splice edit replacing each such block's whole content VALUE with a restorable marker.
// msgBase is the element's absolute start byte; msgIdx is its message index (the "later" comparison
// point). Value spans are located by KEY via objectValueSpan, never bytes.Index, so a byte-identical
// sibling field cannot be mis-located. A cache_control-bearing block is skipped (belt-and-suspenders
// over the per-message skip). The marker is only applied when it is strictly shorter than the
// original, so a rewrite never grows the body.
func (idx readLifecycleIndex) collectStaleReadEdits(msgBase, msgIdx int, el json.RawMessage) (edits []spliceEdit, restores []StaleRestore, shed int) {
	var m struct {
		Role string `json:"role"`
	}
	if json.Unmarshal(el, &m) != nil || m.Role != "user" {
		return nil, nil, 0 // tool_result blocks live in user turns
	}
	forEachToolResultBlock(msgBase, el, func(blk json.RawMessage, blkBase int) {
		if rawHasCacheControl(blk) || toolResultContentHasCacheControl(blk) {
			return
		}
		var b struct {
			ToolUseID string `json:"tool_use_id"`
		}
		if json.Unmarshal(blk, &b) != nil || b.ToolUseID == "" {
			return
		}
		displayPath, isRead := idx.readPathByToolUse[b.ToolUseID]
		if !isRead {
			return // this tool_result did not come from a Read
		}
		editIdx, edited := idx.maxEditIdxByPath[normReadPath(displayPath)]
		if !edited || editIdx <= msgIdx {
			return // fresh: the file was not edited AFTER this read
		}
		cStart, cEnd, ok := objectValueSpan(blk, "content")
		if !ok {
			return
		}
		cVal := blk[cStart:cEnd]
		origText, ok := decodeToolResultContentText(cVal)
		if !ok || origText == "" {
			return
		}
		id := originatingTaskDigestID([]byte(origText))
		newVal, err := json.Marshal(staleReadMarker(displayPath, id))
		if err != nil || len(newVal) >= len(cVal) {
			return // never grow the body; a read already smaller than the marker is not worth eliding
		}
		edits = append(edits, spliceEdit{start: blkBase + cStart, end: blkBase + cEnd, repl: newVal})
		restores = append(restores, StaleRestore{ID: id, Bytes: []byte(origText), Excerpt: staleReadExcerpt(displayPath, origText)})
		shed += len(cVal) - len(newVal)
	})
	return edits, restores, shed
}

// decodeToolResultContentText renders a tool_result `content` value — a bare JSON string OR an array
// of blocks — to its plain text, the bytes a re-fetching model wants back and the bytes the restore
// handle is content-addressed on. A bare string decodes to itself; an array yields the concatenation
// of its text blocks (non-text blocks contribute nothing). ok is false for a shape we do not model,
// so the caller leaves it untouched.
func decodeToolResultContentText(cVal []byte) (string, bool) {
	c := skipSpace(cVal)
	if len(c) == 0 {
		return "", false
	}
	if c[0] == '"' {
		var s string
		if json.Unmarshal(c, &s) != nil {
			return "", false
		}
		return s, true
	}
	if c[0] != '[' {
		return "", false
	}
	var blocks []json.RawMessage
	if json.Unmarshal(c, &blocks) != nil {
		return "", false
	}
	var sb strings.Builder
	for _, blk := range blocks {
		var b struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if json.Unmarshal(blk, &b) == nil && b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return sb.String(), true
}

// staleReadMarker is the in-band notice that stands in for an elided stale read: it names the file,
// says WHY the body was dropped (a later same-file edit superseded it), and carries the id=<hex>
// content-address (the same compactRestoreIDField grammar the compaction stub uses) a model presents
// to fak_context_restore to page the original read back in.
func staleReadMarker(path, id string) string {
	return fmt.Sprintf("…[fak: this Read of %s was superseded by a later in-session edit and its body was elided to stay within the context budget; the file's current state reflects that edit, not this snapshot. Recover the original read via fak_context_restore %s%s]…", path, compactRestoreIDField, id)
}

// staleReadExcerpt is the bounded, single-line orientation string stashed alongside the original
// bytes (echoed back by fak_context_restore as "what it is"). It is the file path plus a short,
// rune-safe head of the content — never the whole body, which rides the restore Bytes.
func staleReadExcerpt(path, text string) string {
	head := strings.ReplaceAll(strings.ReplaceAll(text, "\r", " "), "\n", " ")
	const cap = 160
	if r := []rune(head); len(r) > cap {
		head = string(r[:cap]) + "…"
	}
	return "stale Read of " + path + ": " + head
}

// normReadPath normalizes a file path for cross-tool matching: trimmed, backslashes folded to forward
// slashes, and lowercased. This matches the reference classifier's normcase and tolerates the
// case/slash variation the same file can pick up across Read and Edit tool inputs on Windows. It is
// deliberately case-INSENSITIVE: on a case-sensitive filesystem two files differing only in case
// would be conflated, but the only consequence is a bounded, restorable over-elision, never a
// correctness loss.
func normReadPath(p string) string {
	return strings.ToLower(strings.ReplaceAll(strings.TrimSpace(p), "\\", "/"))
}

// firstNonBlank returns a if it is non-blank, else b (used to read file_path with a notebook_path
// fallback for NotebookEdit).
func firstNonBlank(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}

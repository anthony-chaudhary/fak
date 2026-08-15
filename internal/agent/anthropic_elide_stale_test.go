package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// staleWire builds a /v1/messages body from a compact spec DSL so each read-lifecycle test states
// exactly the transcript it needs. Every body opens with a cached-head breakpoint (the protected
// prefix) so the transform has a cache anchor. The DSL block builders:
//
//	head(text)            → user turn carrying the FIRST cache_control breakpoint (pfxEnd=0)
//	readUse(id, path)     → assistant turn with a Read tool_use (id links its later tool_result)
//	editUse(tool, path)   → assistant turn with a file-mutating tool_use (Edit/Write/…)
//	result(id, text, cc)  → user turn with a tool_result for tool_use `id`; cc adds cache_control
//	filler(role, text)    → a plain text turn (recent-window padding / role alternation)
type staleBlock map[string]any

func staleWire(t *testing.T, msgs []staleBlock) []byte {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"system":     []map[string]any{{"type": "text", "text": "policy", "cache_control": map[string]any{"type": "ephemeral"}}},
		"messages":   msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

func staleHead(text string) staleBlock {
	return staleBlock{"role": "user", "content": []map[string]any{{"type": "text", "text": text, "cache_control": map[string]any{"type": "ephemeral"}}}}
}

func staleReadUse(id, path string) staleBlock {
	return staleBlock{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": id, "name": "Read", "input": map[string]any{"file_path": path}}}}
}

func staleEditUse(tool, path string) staleBlock {
	key := "file_path"
	if tool == "NotebookEdit" {
		key = "notebook_path"
	}
	return staleBlock{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": "u_" + path, "name": tool, "input": map[string]any{key: path}}}}
}

func staleResult(id, text string, cached bool) staleBlock {
	blk := map[string]any{"type": "tool_result", "tool_use_id": id, "content": []map[string]any{{"type": "text", "text": text}}}
	if cached {
		blk["cache_control"] = map[string]any{"type": "ephemeral"}
	}
	return staleBlock{"role": "user", "content": []map[string]any{blk}}
}

func staleFiller(role, text string) staleBlock {
	return staleBlock{"role": role, "content": []map[string]any{{"type": "text", "text": text}}}
}

// stalePrefixEnd is the absolute byte offset just past the first cache_control breakpoint — the
// bytes the rewrite must preserve verbatim (uses the SAME anchor the transform does).
func stalePrefixEnd(t *testing.T, raw []byte) int {
	t.Helper()
	var o map[string]json.RawMessage
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElements(raw, o["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	if pfx := firstBreakpointForElide(elems); pfx >= 0 {
		return spans[pfx].end
	}
	return spans[0].start
}

// TestReadLifecycleElidesStaleKeepsFreshAndPrefix is the core witness: a Read whose file is Edited
// in a LATER turn is replaced by a restorable marker, while a Read of a never-edited file (and the
// cached prefix) survive byte-for-byte, the original is returned for the restore stash under a
// content-address that matches the marker, and the body still decodes.
func TestReadLifecycleElidesStaleKeepsFreshAndPrefix(t *testing.T) {
	// Newline-free runs so a raw bytes.Contains needle matches the JSON body (JSON escapes \n to \\n).
	bigX := strings.Repeat("PREEDIT-X-", 400) // stale read content
	bigY := strings.Repeat("FRESH-Y-", 400)   // fresh read content
	msgs := []staleBlock{
		staleHead("cached head context"),                // 0 breakpoint → pfxEnd=0
		staleReadUse("rX", "/repo/x.go"),                // 1
		staleResult("rX", bigX, false),                  // 2 ELIGIBLE + STALE (x.go edited @3) → ELIDE
		staleEditUse("Edit", "/repo/x.go"),              // 3 supersedes rX
		staleResult("u_/repo/x.go", "edited ok", false), // 4 edit result (not a read) → ignored
		staleReadUse("rY", "/repo/y.go"),                // 5
		staleResult("rY", bigY, false),                  // 6 ELIGIBLE + FRESH (y.go never edited) → SURVIVE
		staleFiller("assistant", "a7"),                  // 7
		staleFiller("user", "u8"),                       // 8 ┐ recent window (last 4: 8..11)
		staleFiller("assistant", "a9"),                  // 9 │
		staleFiller("user", "u10"),                      // 10│
		staleFiller("assistant", "a11"),                 // 11┘
	}
	raw := staleWire(t, msgs)
	orig := append([]byte(nil), raw...)
	prefixEnd := stalePrefixEnd(t, orig)

	out, outcome := ElideStaleReadsWithOutcome(raw)

	if outcome.Reason != StaleReasonNone {
		t.Fatalf("expected FIRED (StaleReasonNone), got %q", outcome.Reason)
	}
	if outcome.Elided != 1 {
		t.Fatalf("expected exactly 1 stale read elided, got %d", outcome.Elided)
	}
	if outcome.ShedBytes <= 0 {
		t.Fatalf("expected positive ShedBytes, got %d", outcome.ShedBytes)
	}
	if outcome.ShedTokens <= 0 {
		t.Fatalf("expected positive ShedTokens, got %d", outcome.ShedTokens)
	}
	// (a) The stale read body is gone and the restore marker (naming the file) is present.
	if bytes.Contains(out, []byte(bigX)) {
		t.Error("stale read of x.go was NOT elided (pre-edit body still present)")
	}
	if !bytes.Contains(out, []byte("superseded by a later in-session edit")) || !bytes.Contains(out, []byte("/repo/x.go")) {
		t.Error("stale restore marker (or file path) missing from output")
	}
	// (b) The fresh read of a never-edited file survives verbatim.
	if !bytes.Contains(out, []byte(bigY)) {
		t.Error("fresh read of y.go was wrongly elided")
	}
	// (c) The protected prefix bytes are byte-identical (cache preserved).
	if prefixEnd > len(out) || !bytes.Equal(orig[:prefixEnd], out[:prefixEnd]) {
		t.Error("protected prefix bytes changed — cache hit would be lost")
	}
	// (d) The rewritten body still decodes as a valid Anthropic request.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("rewritten body failed to re-decode: %v", err)
	}
	// (e) The restore payload is the FULL original content, content-addressed by the SAME id the
	// marker embeds — so fak_context_restore(id) pages back exactly what was elided.
	if len(outcome.Restores) != 1 {
		t.Fatalf("expected 1 restore payload, got %d", len(outcome.Restores))
	}
	r := outcome.Restores[0]
	if string(r.Bytes) != bigX {
		t.Errorf("restore Bytes are not the original read content (len %d vs %d)", len(r.Bytes), len(bigX))
	}
	if want := originatingTaskDigestID([]byte(bigX)); r.ID != want {
		t.Errorf("restore ID = %q, want content-address %q", r.ID, want)
	}
	if !bytes.Contains(out, []byte("id="+r.ID)) {
		t.Error("marker does not carry the id=<hash> restore handle")
	}
	// The input slice is not mutated in place when a rewrite happens.
	if !bytes.Equal(raw, orig) {
		t.Error("input raw was mutated in place")
	}
}

// TestReadLifecycleExcludesProtectedStaleReads proves the cache-safety exclusions: a stale read is
// left INTACT when it is cache_control-bearing (cache prefix) or inside the recent working-set
// window, even though a later edit superseded it. No edit is produced → identity return.
func TestReadLifecycleExcludesProtectedStaleReads(t *testing.T) {
	// Newline-free runs (see TestReadLifecycleElidesStaleKeepsFreshAndPrefix) so bytes.Contains matches.
	bigW := strings.Repeat("CACHED-STALE-W-", 300)
	bigZ := strings.Repeat("RECENT-STALE-Z-", 300)
	msgs := []staleBlock{
		staleHead("cached head"),                 // 0 breakpoint
		staleReadUse("rW", "/repo/w.go"),         // 1
		staleResult("rW", bigW, true),            // 2 STALE (w.go edited @3) BUT cache_control → SURVIVE
		staleEditUse("Write", "/repo/w.go"),      // 3
		staleResult("u_/repo/w.go", "ok", false), // 4
		staleReadUse("rZ", "/repo/z.go"),         // 5
		staleResult("rZ", bigZ, false),           // 6 STALE (z.go edited @7) BUT recent (last 4: 6..9) → SURVIVE
		staleEditUse("Edit", "/repo/z.go"),       // 7
		staleFiller("user", "u8"),                // 8
		staleFiller("assistant", "a9"),           // 9
	}
	raw := staleWire(t, msgs)
	orig := append([]byte(nil), raw...)

	out, outcome := ElideStaleReadsWithOutcome(raw)

	if outcome.Reason != StaleReasonNoStaleReads {
		t.Fatalf("expected StaleReasonNoStaleReads (all stale reads protected), got %q", outcome.Reason)
	}
	if !bytes.Equal(out, orig) {
		t.Fatal("protected stale reads must leave the body byte-identical")
	}
	if !bytes.Contains(out, []byte(bigW)) || !bytes.Contains(out, []byte(bigZ)) {
		t.Error("a protected (cache_control / recent) stale read was wrongly elided")
	}
}

// TestReadLifecyclePathNormalization proves the read/edit pairing tolerates the case/slash variation
// the same file can pick up across tool inputs: a Read of "C:\\repo\\X.go" is superseded by an Edit
// of "C:/repo/x.go".
func TestReadLifecyclePathNormalization(t *testing.T) {
	bigX := strings.Repeat("windows path body line.\n", 300)
	msgs := []staleBlock{
		staleHead("cached head"),             // 0 breakpoint
		staleReadUse("rX", `C:\repo\X.go`),   // 1 backslashes + uppercase
		staleResult("rX", bigX, false),       // 2 ELIGIBLE + STALE (edit @3, normalized match)
		staleEditUse("Edit", "C:/repo/x.go"), // 3 forward slash + lowercase → same file
		staleFiller("assistant", "a4"),       // 4
		staleFiller("user", "u5"),            // 5 ┐ recent (last 4: 4..7)
		staleFiller("assistant", "a6"),       // 6 │
		staleFiller("user", "u7"),            // 7 ┘
	}
	raw := staleWire(t, msgs)
	out, outcome := ElideStaleReadsWithOutcome(raw)

	if outcome.Reason != StaleReasonNone || outcome.Elided != 1 {
		t.Fatalf("expected the cross-slash/case read to be detected stale: reason=%q elided=%d", outcome.Reason, outcome.Elided)
	}
	if bytes.Contains(out, []byte(bigX)) {
		t.Error("stale read across path-normalization was not elided")
	}
}

// TestReadLifecycleNotebookEdit proves a NotebookEdit (which names notebook_path, not file_path)
// supersedes an earlier Read of the same notebook.
func TestReadLifecycleNotebookEdit(t *testing.T) {
	big := strings.Repeat("notebook cell dump line.\n", 300)
	msgs := []staleBlock{
		staleHead("cached head"),                             // 0 breakpoint
		staleReadUse("rN", "/repo/analysis.ipynb"),           // 1
		staleResult("rN", big, false),                        // 2 ELIGIBLE + STALE (notebook edited @3)
		staleEditUse("NotebookEdit", "/repo/analysis.ipynb"), // 3
		staleFiller("assistant", "a4"),                       // 4
		staleFiller("user", "u5"),                            // 5
		staleFiller("assistant", "a6"),                       // 6
		staleFiller("user", "u7"),                            // 7
	}
	raw := staleWire(t, msgs)
	_, outcome := ElideStaleReadsWithOutcome(raw)
	if outcome.Reason != StaleReasonNone || outcome.Elided != 1 {
		t.Fatalf("NotebookEdit did not supersede its Read: reason=%q elided=%d", outcome.Reason, outcome.Elided)
	}
}

// TestReadLifecycleFreshReadIdentity proves a read of a file edited BEFORE the read (or never) is
// NOT stale — the read already reflects the post-edit state.
func TestReadLifecycleFreshReadIdentity(t *testing.T) {
	big := strings.Repeat("post-edit body line.\n", 300)
	msgs := []staleBlock{
		staleHead("cached head"),                 // 0 breakpoint
		staleEditUse("Edit", "/repo/x.go"),       // 1 edit BEFORE the read
		staleResult("u_/repo/x.go", "ok", false), // 2
		staleReadUse("rX", "/repo/x.go"),         // 3 read AFTER the edit → FRESH
		staleResult("rX", big, false),            // 4 must NOT be elided
		staleFiller("assistant", "a5"),           // 5
		staleFiller("user", "u6"),                // 6
		staleFiller("assistant", "a7"),           // 7
	}
	raw := staleWire(t, msgs)
	orig := append([]byte(nil), raw...)
	out, outcome := ElideStaleReadsWithOutcome(raw)
	if outcome.Reason != StaleReasonNoStaleReads {
		t.Fatalf("a read AFTER its file's edit is fresh, expected StaleReasonNoStaleReads, got %q", outcome.Reason)
	}
	if !bytes.Equal(out, orig) {
		t.Error("a fresh read must leave the body unchanged")
	}
}

// TestReadLifecycleBailReasons pins the identity-return vocabulary for the degenerate inputs.
func TestReadLifecycleBailReasons(t *testing.T) {
	cases := []struct {
		name string
		raw  []byte
		want string
	}{
		{"empty", nil, StaleReasonOff},
		{"non-json", []byte("not json"), StaleReasonNonJSON},
		{"no-messages", []byte(`{"model":"m"}`), StaleReasonNoMsgsKey},
		{"too-few", []byte(`{"messages":[{"role":"user","content":"hi"}]}`), StaleReasonTooFewMsgs},
		{
			// A stale read exists but there is NO cache_control anchor anywhere → cannot know the
			// cache boundary → refuse to touch the body.
			"no-breakpoint",
			func() []byte {
				b, _ := json.Marshal(map[string]any{
					"model": "m", "max_tokens": 1,
					"messages": []staleBlock{
						staleReadUse("rX", "/repo/x.go"),
						staleResult("rX", "body", false),
						staleEditUse("Edit", "/repo/x.go"),
						staleFiller("user", "u3"),
					},
				})
				return b
			}(),
			StaleReasonNoBreakpoint,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			orig := append([]byte(nil), tc.raw...)
			out, outcome := ElideStaleReadsWithOutcome(tc.raw)
			if outcome.Reason != tc.want {
				t.Fatalf("reason = %q, want %q", outcome.Reason, tc.want)
			}
			if !bytes.Equal(out, orig) {
				t.Error("a bail must return the body unchanged")
			}
		})
	}
}

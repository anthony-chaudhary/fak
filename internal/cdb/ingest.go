// Package cdb is the context debugger: it attaches to a FINISHED agent session as
// if to a core dump and answers questions by demand-paging only the working set the
// question touches — never by replaying the whole address space.
//
// The reframe (../../EXPLAINER-trust-floor-two-lenses-2026-06-17.md, the goal note):
// because the context-MMU is a WRITE-TIME gate, the heavy bytes of a session were
// already paged out the moment they were produced — each tool result is a
// content-addressed Ref into a shared blob store, and oversize-but-benign results
// were replaced in place with a <2KB pointer. So a "350k-token session" is really
//
//	manifest  (the page table: roles + digests + descriptors + quarantine state) ← small
//	+ CAS     (the swap device: dedup'd, content-addressed cold bytes)            ← cold
//	+ a frozen world-version (no more writes will ever happen)
//
// That is a core image. cdb is the debugger you attach to it. It is a PURE CONSUMER
// of the shipped recall core-image substrate (which is itself a pure consumer of the
// ctxmmu write-time gate + the blob CAS); it registers NOTHING with the frozen ABI
// and adds nothing to it.
//
// ingest.go is the front door: it turns a REAL, finished Claude Code session
// transcript (the JSONL under <claude-home>/projects/<ns>/<uuid>.jsonl) into a core
// image, driving every tool result back through the SAME shipped gate at record time.
// So a session you actually ran becomes a debuggable core dump — heavy results page
// out to the CAS, and an injection/secret result is sealed exactly as it would have
// been live.
package cdb

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// IngestStats accounts for what the ingest saw, so a caller can report how a flat
// transcript decomposed into a page table (the core-dump view) without re-reading it.
type IngestStats struct {
	SessionID   string `json:"session_id"`
	Records     int    `json:"records"`   // JSONL lines parsed
	ToolUses    int    `json:"tool_uses"` // tool_use blocks seen (the call side)
	Pages       int    `json:"pages"`     // tool_result blocks recorded as pages
	Sealed      int    `json:"sealed"`    // pages the write-time gate quarantined
	RawBytes    int64  `json:"raw_bytes"` // total result bytes recorded (pre-dedup)
	SourcePath  string `json:"source_path"`
	SourceBytes int64  `json:"source_bytes"` // size of the transcript on disk
}

// jrecord is one JSONL line. Only the fields cdb needs are typed; everything else
// (usage, meta, mode, attachment, ...) is ignored — forward-compatible by construction.
type jrecord struct {
	Type    string `json:"type"`
	Message *struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"message"`
}

// jblock is one content block inside a message. tool_use carries (id, name); a
// tool_result carries (tool_use_id, content) and is linked to its tool_use by id.
type jblock struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`          // tool_use
	Name      string          `json:"name"`        // tool_use
	ToolUseID string          `json:"tool_use_id"` // tool_result
	Content   json.RawMessage `json:"content"`     // tool_result (string OR block list)
	IsError   bool            `json:"is_error"`    // tool_result
	Text      string          `json:"text"`        // text block (unused as a page)
}

// IngestSession parses a Claude Code session JSONL at path into a recall.Recorder: one
// PAGE per tool_result, with the page's role resolved to the tool that produced it
// (via the tool_use_id link) and the page body run through the SHIPPED ctxmmu gate.
// The returned recorder can be Persist()ed to a durable core image and re-Attach()ed.
//
// Faithful mapping: pages == tool RESULTS, because the context-MMU is a gate on
// results — the call args are the adjudicator's side, not the MMU's. A result that is
// oversize pages out to the CAS; one that trips the injection/secret/pollution shapes
// is sealed, exactly as it would have been in the live run.
func IngestSession(ctx context.Context, path, sessionID string) (*recall.Recorder, IngestStats, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, IngestStats{}, err
	}
	defer f.Close()
	st := IngestStats{SessionID: sessionID, SourcePath: path}
	if fi, e := f.Stat(); e == nil {
		st.SourceBytes = fi.Size()
	}
	rec, s, err := ingest(ctx, f, sessionID)
	if err != nil {
		return nil, IngestStats{}, err
	}
	s.SourcePath, s.SourceBytes = st.SourcePath, st.SourceBytes
	return rec, s, nil
}

// ingest is the reader-driven core, split out so tests can drive it from an
// in-memory transcript without a temp file.
func ingest(ctx context.Context, r io.Reader, sessionID string) (*recall.Recorder, IngestStats, error) {
	rec := recall.NewRecorder(sessionID)
	st := IngestStats{SessionID: sessionID}
	toolName := map[string]string{} // tool_use id -> tool name

	sc := bufio.NewScanner(r)
	// transcript lines can be large (a 20KB+ tool result on one line); raise the cap.
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr jrecord
		if err := json.Unmarshal(line, &jr); err != nil {
			continue // a malformed line is skipped, not fatal — best-effort over real data
		}
		st.Records++
		if jr.Message == nil || len(jr.Message.Content) == 0 {
			continue
		}
		// message.content is either a JSON string (a user prompt / hook text — never a
		// page) or an array of blocks. Only the array form carries tool traffic.
		var blocks []jblock
		if err := json.Unmarshal(jr.Message.Content, &blocks); err != nil {
			continue
		}
		for _, b := range blocks {
			switch b.Type {
			case "tool_use":
				st.ToolUses++
				if b.ID != "" {
					toolName[b.ID] = b.Name
				}
			case "tool_result":
				role := toolName[b.ToolUseID]
				if role == "" {
					role = "tool"
				}
				body := flattenContent(b.Content)
				if b.IsError {
					body = append([]byte("[tool error] "), body...)
				}
				if len(body) == 0 {
					continue
				}
				v := rec.Record(ctx, role, body)
				st.Pages++
				st.RawBytes += int64(len(body))
				if v.Kind == abi.VerdictQuarantine {
					st.Sealed++
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, IngestStats{}, fmt.Errorf("cdb: scan %s: %w", sessionID, err)
	}
	return rec, st, nil
}

// flattenContent renders a tool_result's content into the raw result bytes the gate
// should screen — exactly what entered the live context, BYTE-FAITHFULLY so a page-in
// is rung-0 identical to the transcript.
//
//   - a JSON string (the common small result) -> the string bytes;
//   - an all-text block list                  -> the texts, newline-joined (clean);
//   - anything else (an image block carrying a base64 PNG, a mixed list, an unknown
//     shape) -> the RAW content bytes verbatim. This is load-bearing: a 160KB image
//     result is the canonical oversize-but-benign page that must page out to the CAS,
//     not be silently dropped to a stub because our typed struct ignores its `source`.
func flattenContent(raw json.RawMessage) []byte {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []byte(s)
	}
	var blocks []jblock
	if json.Unmarshal(raw, &blocks) == nil {
		allText := len(blocks) > 0
		for _, b := range blocks {
			if b.Type != "text" {
				allText = false
				break
			}
		}
		if allText {
			var sb strings.Builder
			for i, b := range blocks {
				if i > 0 {
					sb.WriteByte('\n')
				}
				sb.WriteString(b.Text)
			}
			return []byte(sb.String())
		}
	}
	// mixed / non-text / unknown: preserve the transcript bytes verbatim.
	return append([]byte(nil), raw...)
}

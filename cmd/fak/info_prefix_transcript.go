package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// prefixTranscriptTurnResult is one line of `fak info --prefix-transcript` output: the
// turn number and the live cachemeta.PrefixStabilityScore computed for it.
type prefixTranscriptTurnResult struct {
	Turn  int                            `json:"turn"`
	Score cachemeta.PrefixStabilityScore `json:"score"`
}

// prefixTranscriptReport is the full `fak info --prefix-transcript` artifact: every
// turn's score plus the FINAL turn's score again as Summary (the state a live session
// watching this transcript would report right now).
type prefixTranscriptReport struct {
	Turns   []prefixTranscriptTurnResult    `json:"turns"`
	Summary *cachemeta.PrefixStabilityScore `json:"summary"`
}

// runInfoPrefixTranscript is issue #1602's compute-and-display entry point: it reads a
// recorded Claude Code / GLM transcript (the same JSONL shape cmd/prefixlint reads),
// runs a fresh cachemeta.PrefixStabilityTracker turn-by-turn over the PROTECTED span of
// each turn (system + tool-schema + any sealed span — the front cacheable run, capped at
// the first message/tool-result segment), and prints the three-state verdict
// (prefix-stable / prefix-mutated / prefix-unknown) for every turn plus a final summary.
// It needs no running gateway: the whole computation is local and offline, exactly like
// `fak info --json` needs no agent, only here the input is a transcript file instead of
// a live /debug/vars poll.
func runInfoPrefixTranscript(stdout, stderr io.Writer, path string, asJSON bool) int {
	turns, err := loadPrefixTranscriptTurns(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak info: --prefix-transcript: %v\n", err)
		return 1
	}
	if len(turns) == 0 {
		fmt.Fprintf(stderr, "fak info: --prefix-transcript: no assistant turns found in %s\n", path)
		return 1
	}
	tr := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	report := prefixTranscriptReport{Turns: make([]prefixTranscriptTurnResult, 0, len(turns))}
	for i, turn := range turns {
		score := tr.Observe(protectedSpanOf(turn))
		report.Turns = append(report.Turns, prefixTranscriptTurnResult{Turn: i + 1, Score: score})
		report.Summary = &report.Turns[len(report.Turns)-1].Score
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak info")
	}
	fmt.Fprintf(stdout, "prefix-stability (%d turns, %s)\n", len(report.Turns), path)
	for _, row := range report.Turns {
		fmt.Fprintf(stdout, "  turn %-4d %-14s %s\n", row.Turn, row.Score.State, row.Score.Reason)
	}
	if report.Summary != nil {
		fmt.Fprintf(stdout, "summary: %s — %s\n", report.Summary.State, report.Summary.Reason)
	}
	return 0
}

// protectedSpanOf returns the leading run of a turn that is meant to stay
// stable/cacheable — every segment up to (but not including) the first ordinary
// message/tool-result segment, INCLUDING a sealed span so a quarantined span still
// caps the baseline (mirroring frontCacheableRun's contract in prefix_stability.go,
// but keeping a sealed segment IN the compared span rather than stopping before it, so
// PrefixStabilityTracker can observe and report the seal itself rather than silently
// truncating it away).
func protectedSpanOf(turn []cachemeta.PromptSegment) []cachemeta.PromptSegment {
	end := 0
	for _, s := range turn {
		switch s.Kind {
		case cachemeta.SegStable, cachemeta.SegToolSchema, cachemeta.SegVolatile:
			end++
			continue
		case cachemeta.SegSealed:
			end++
		}
		break
	}
	return turn[:end]
}

// loadPrefixTranscriptTurns parses a Claude Code / GLM transcript JSONL into the
// per-assistant-request cumulative turns cachemeta.TurnsFromConversation expects — the
// same coarse role-classified parsing cmd/prefixlint's runJSONL uses, kept local so
// `fak info` has no dependency on the prefixlint binary.
func loadPrefixTranscriptTurns(path string) ([][]cachemeta.PromptSegment, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	type jblock struct {
		Type    string          `json:"type"`
		Text    string          `json:"text"`
		Content json.RawMessage `json:"content"`
	}
	type jrecord struct {
		Type    string `json:"type"`
		Message *struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"message"`
	}

	var parts []cachemeta.ConvPart
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var jr jrecord
		if json.Unmarshal([]byte(line), &jr) != nil || jr.Message == nil {
			continue
		}
		role := jr.Message.Role
		var s string
		if json.Unmarshal(jr.Message.Content, &s) == nil {
			parts = append(parts, cachemeta.ConvPart{Role: role, Content: []byte(s)})
			continue
		}
		var blocks []jblock
		if json.Unmarshal(jr.Message.Content, &blocks) != nil {
			continue
		}
		for _, bl := range blocks {
			switch bl.Type {
			case "text":
				parts = append(parts, cachemeta.ConvPart{Role: role, Content: []byte(bl.Text)})
			case "tool_result":
				parts = append(parts, cachemeta.ConvPart{Role: "tool_result", Content: []byte(bl.Content)})
			case "tool_use":
				parts = append(parts, cachemeta.ConvPart{Role: "tool_schema", Content: []byte(bl.Content)})
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for i := range parts {
		if parts[i].Role == "" {
			parts[i].Role = "user"
		}
	}
	return cachemeta.TurnsFromConversation(parts), nil
}

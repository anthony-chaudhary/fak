// Command prefixlint is the §A3 prefix-stability devtool (GLM52-HOSTED-CACHE-
// COHERENCE): it reads a recorded conversation and reports the provider-cache
// consequence — how many prompt tokens are cacheable across the session, how many are
// re-billed, the specific turn where the prefix broke, and the recoverable uplift from
// fixing volatile-ahead-of-stable ordering. It is the operator front-end over the
// witnessed cachemeta prefix-stability core.
//
//	prefixlint -selfcheck         # zero-dependency proof on a synthetic session
//	prefixlint -jsonl <session>   # report over a Claude Code / GLM transcript JSONL
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run a zero-dependency synthetic proof and assert the bug is detected")
	jsonl := flag.String("jsonl", "", "path to a Claude Code / GLM session transcript (JSONL)")
	flag.Parse()

	switch {
	case *selfcheck:
		runSelfcheck()
	case *jsonl != "":
		runJSONL(*jsonl)
	default:
		fmt.Fprintln(os.Stderr, "usage: prefixlint -selfcheck | -jsonl <session.jsonl>")
		os.Exit(2)
	}
}

// runSelfcheck builds a synthetic session whose every turn front-loads a volatile
// request id (the classic cache-killer that strands the stable prefix), prints the
// report, and ASSERTS the linter detected the recoverable uplift — a self-contained
// proof the devtool path is correct with no external file.
func runSelfcheck() {
	turns := syntheticSession()
	rep := cachemeta.AnalyzeStability(turns)
	fmt.Print(renderReport(turns, rep))
	if rep.RecoverableTokens <= 0 {
		fail("selfcheck: expected a recoverable volatile-ahead uplift, got %d", rep.RecoverableTokens)
	}
	if rep.BrokeAtTurn < 0 {
		fail("selfcheck: expected the prefix to break (the req id changes every turn)")
	}
	fmt.Println("\nselfcheck OK: volatile-ahead ordering bug detected and the uplift scored.")
}

// syntheticSession mirrors the cachemeta witness: each turn is [volatile req-id,
// system, tool schema, user msg], so the whole stable prefix is stranded behind the
// changing id and is recoverable by moving the id to the tail.
func syntheticSession() [][]cachemeta.PromptSegment {
	mk := func(reqid, msg string) []cachemeta.ConvPart {
		return []cachemeta.ConvPart{
			{Role: "meta", Volatile: true, Tokens: 6, Content: []byte(reqid)},
			{Role: "system", Tokens: 100, Content: []byte("You are a coding agent.")},
			{Role: "tool_schema", Tokens: 200, Content: []byte(`{"tools":["read","write"]}`)},
			{Role: "user", Tokens: 10, Content: []byte(msg)},
		}
	}
	var turns [][]cachemeta.PromptSegment
	for _, t := range []struct{ id, msg string }{{"req=1", "a"}, {"req=2", "b"}, {"req=3", "c"}} {
		turns = append(turns, cachemeta.SegmentsFromParts(mk(t.id, t.msg)))
	}
	return turns
}

// renderReport is the human-facing §A3 output.
func renderReport(turns [][]cachemeta.PromptSegment, rep cachemeta.StabilityReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "prefix-stability report (%d turns)\n", rep.Turns)
	fmt.Fprintf(&b, "  cacheable tokens across session : %d\n", rep.CacheableTokens)
	fmt.Fprintf(&b, "  re-billed (lost) tokens         : %d\n", rep.LostTokens)
	if rep.BrokeAtTurn >= 0 {
		fmt.Fprintf(&b, "  prefix first broke at turn      : %d\n", rep.BrokeAtTurn)
	} else {
		fmt.Fprintf(&b, "  prefix first broke at turn      : (never — clean across the session)\n")
	}
	fmt.Fprintf(&b, "  recoverable by reorder (uplift) : %d tokens\n", rep.RecoverableTokens)
	// Per-turn layout recommendation where a fix helps.
	for i, t := range turns {
		if rec := cachemeta.RecommendLayout(t); rec.Changed {
			fmt.Fprintf(&b, "  turn %d: move %d volatile segment(s) to the tail -> +%d cacheable tokens\n",
				i, rec.MovedVolatile, rec.PredictedUplift)
		}
	}
	return b.String()
}

// --- transcript path ---

type jrecord struct {
	Type    string `json:"type"`
	Message *struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"message"`
}

type jblock struct {
	Type    string          `json:"type"`
	Text    string          `json:"text"`
	Content json.RawMessage `json:"content"`
}

// runJSONL parses a Claude Code transcript into a coarse conversation (one ConvPart
// per message / tool result) and reports. This v1 classifies by role; wiring the
// ctxmmu seal decision (cdb.IngestSession's quarantine) into SegSealed is the tracked
// follow-up — until then a sealed result is treated as an ordinary tool_result.
func runJSONL(path string) {
	f, err := os.Open(path)
	if err != nil {
		fail("open %s: %v", path, err)
	}
	defer f.Close()
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
		role := jr.Message.Role // "user" | "assistant"
		// content is either a JSON string (plain text) or an array of blocks.
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
		fail("scan %s: %v", path, err)
	}
	// Treat each assistant message as a model request; report over those turns.
	for i := range parts {
		if parts[i].Role == "" {
			parts[i].Role = "user"
		}
		if parts[i].Role == "assistant" {
			parts[i].Role = "assistant" // marks a turn boundary in TurnsFromConversation
		}
	}
	turns := cachemeta.TurnsFromConversation(parts)
	if len(turns) == 0 {
		fail("no assistant turns found in %s (need an array-content transcript)", path)
	}
	fmt.Print(renderReport(turns, cachemeta.AnalyzeStability(turns)))
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", a...)
	os.Exit(1)
}

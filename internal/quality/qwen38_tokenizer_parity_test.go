package quality

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// qwen38TokRunner implements Runner for Qwen3.8 tokenization differential tests.
type qwen38TokRunner struct {
	tok    *tokenizer.Tokenizer
	defect string
}

func (r qwen38TokRunner) Name() string {
	if r.defect != "" {
		return "qwen38-tok-" + r.defect
	}
	return "qwen38-tok"
}

func (r qwen38TokRunner) Run(c QualityCase) (Trace, error) {
	ids, err := r.tok.Encode(c.Prompt)
	if err != nil {
		return Trace{}, err
	}
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = strconv.Itoa(id)
	}

	switch r.defect {
	case "":
		// Clean run
	case "missing-im-start":
		// Drops first token -> fails at index 0.
		if len(tokens) > 0 {
			tokens = tokens[1:]
		}
	case "double-im-start":
		// Duplicates <|im_start|> at start -> fails at index 1.
		if len(tokens) > 0 {
			tokens = append([]string{tokens[0]}, tokens...)
		}
	case "corrupt-eos":
		// Swaps <|im_end|> token ID -> fails at exact EOS position.
		eosStr := strconv.Itoa(248046)
		for i, s := range tokens {
			if s == eosStr {
				tokens[i] = "999999"
				break
			}
		}
	case "think-seed-drift":
		// Alters <think> seed -> fails at exact think token position.
		thinkStr := strconv.Itoa(248068)
		for i, s := range tokens {
			if s == thinkStr {
				tokens[i] = "999999"
				break
			}
		}
	default:
		return Trace{}, fmt.Errorf("qwen38TokRunner: unknown defect %q", r.defect)
	}

	t := Trace{
		Tokens: tokens,
		Text:   strings.Join(tokens, " "),
		Runner: r.Name(),
	}
	return t, nil
}

func loadQwen38UDHeaderGZ(t *testing.T) []byte {
	t.Helper()
	candidates := []string{
		filepath.Join("..", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"),
		filepath.Join("internal", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"),
	}

	var compressed []byte
	var err error
	for _, path := range candidates {
		compressed, err = os.ReadFile(path)
		if err == nil {
			break
		}
	}
	if err != nil {
		t.Fatalf("could not locate qwen38_ud_q2kxl_header.gguf.gz in candidates %v: %v", candidates, err)
	}

	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatalf("gzip.NewReader: %v", err)
	}
	defer zr.Close()

	raw, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	if err != nil {
		t.Fatalf("decompressing gzip header: %v", err)
	}

	const wantBytes = 10996640
	const wantHash = "1fe82fda85430cca654a156e9ec2915baf460752197013563b426db2581dcc0f"

	if len(raw) != wantBytes {
		t.Fatalf("decompressed byte length = %d, want %d", len(raw), wantBytes)
	}
	gotHash := fmt.Sprintf("%x", sha256.Sum256(raw))
	if gotHash != wantHash {
		t.Fatalf("decompressed sha256 = %s, want %s", gotHash, wantHash)
	}

	return raw
}

func loadQwen38TokenizerAndGT(t *testing.T) (*tokenizer.Tokenizer, *ggufload.GGMLTokenizer) {
	t.Helper()
	raw := loadQwen38UDHeaderGZ(t)

	gg, err := ggufload.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("ggufload.Read: %v", err)
	}

	gt, ok := gg.GGMLTokenizer()
	if !ok {
		t.Fatalf("gg.GGMLTokenizer() returned ok=false")
	}
	if gt.Pre != "qwen35" {
		t.Fatalf("gt.Pre = %q, want %q", gt.Pre, "qwen35")
	}
	if len(gt.Tokens) != 248320 {
		t.Fatalf("len(gt.Tokens) = %d, want 248320", len(gt.Tokens))
	}
	if len(gt.Merges) != 247587 {
		t.Fatalf("len(gt.Merges) = %d, want 247587", len(gt.Merges))
	}
	if len(gt.TokenTypes) != 248320 {
		t.Fatalf("len(gt.TokenTypes) = %d, want 248320", len(gt.TokenTypes))
	}

	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("tokenizer.FromGGML: %v", err)
	}
	if tok.Vocab() != 248320 {
		t.Fatalf("tok.Vocab() = %d, want 248320", tok.Vocab())
	}

	return tok, gt
}

func TestQwen38TokenizerParity(t *testing.T) {
	tok, gt := loadQwen38TokenizerAndGT(t)

	t.Run("SpecialAndControlTokenParity", func(t *testing.T) {
		wantIDs := map[string]int{
			"<|endoftext|>": 248044,
			"<|im_start|>":  248045,
			"<|im_end|>":    248046,
			"<think>":       248068,
			"</think>":      248069,
		}

		for name, wantID := range wantIDs {
			if gt.Tokens[wantID] != name {
				t.Fatalf("gt.Tokens[%d] = %q, want %q", wantID, gt.Tokens[wantID], name)
			}

			got, err := tok.Encode(name)
			if err != nil {
				t.Fatalf("tok.Encode(%q): %v", name, err)
			}
			if len(got) != 1 || got[0] != wantID {
				t.Fatalf("tok.Encode(%q) = %v, want [%d]", name, got, wantID)
			}

			decoded, err := tok.Decode([]int{wantID})
			if err != nil {
				t.Fatalf("tok.Decode([%d]): %v", wantID, err)
			}
			if decoded != name {
				t.Fatalf("tok.Decode([%d]) = %q, want %q", wantID, decoded, name)
			}
		}
	})

	const chatMLSeq = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

	var chatMLIDs []int

	t.Run("ChatMLMultiTurnTokenization", func(t *testing.T) {
		var err error
		chatMLIDs, err = tok.Encode(chatMLSeq)
		if err != nil {
			t.Fatalf("tok.Encode(chatMLSeq): %v", err)
		}

		var countIMStart, countIMEnd, countThink, countThinkEnd int
		for _, id := range chatMLIDs {
			switch id {
			case 248045:
				countIMStart++
			case 248046:
				countIMEnd++
			case 248068:
				countThink++
			case 248069:
				countThinkEnd++
			}
		}

		if countIMStart != 3 {
			t.Fatalf("expected 3 <|im_start|> (248045), got %d", countIMStart)
		}
		if countIMEnd != 2 {
			t.Fatalf("expected 2 <|im_end|> (248046), got %d", countIMEnd)
		}
		if countThink != 1 {
			t.Fatalf("expected 1 <think> (248068), got %d", countThink)
		}
		if countThinkEnd != 1 {
			t.Fatalf("expected 1 </think> (248069), got %d", countThinkEnd)
		}
		if len(chatMLIDs) == 0 || chatMLIDs[0] != 248045 {
			t.Fatalf("ChatML sequence must start with <|im_start|> (248045), got %v", chatMLIDs)
		}

		decoded, err := tok.Decode(chatMLIDs)
		if err != nil {
			t.Fatalf("tok.Decode(chatMLIDs): %v", err)
		}
		if decoded != chatMLSeq {
			t.Fatalf("ChatML decode roundtrip mismatch:\ngot:  %q\nwant: %q", decoded, chatMLSeq)
		}
	})

	t.Run("DifferentialOracleIntegration", func(t *testing.T) {
		if len(chatMLIDs) == 0 {
			var err error
			chatMLIDs, err = tok.Encode(chatMLSeq)
			if err != nil {
				t.Fatalf("tok.Encode(chatMLSeq): %v", err)
			}
		}

		var firstIMEndIdx, thinkIdx int = -1, -1
		for i, id := range chatMLIDs {
			if id == 248046 && firstIMEndIdx < 0 {
				firstIMEndIdx = i
			}
			if id == 248068 && thinkIdx < 0 {
				thinkIdx = i
			}
		}
		if firstIMEndIdx < 0 {
			t.Fatal("chatMLIDs missing <|im_end|> (248046)")
		}
		if thinkIdx < 0 {
			t.Fatal("chatMLIDs missing <think> (248068)")
		}

		idStrings := make([]string, len(chatMLIDs))
		for i, id := range chatMLIDs {
			idStrings[i] = strconv.Itoa(id)
		}

		c := QualityCase{
			Schema:  CaseSchema,
			ID:      "qwen38-ud-q2kxl-tokenizer-parity",
			Version: 1,
			Prompt:  chatMLSeq,
			Params:  SamplingParams{Temperature: 0, MaxTokens: len(idStrings)},
			Reference: Trace{
				Tokens: idStrings,
				Text:   strings.Join(idStrings, " "),
			},
			Oracles: []string{"tokenizer-parity"},
		}

		// Clean run
		res, err := RunCase(c, ReferenceRunner{}, qwen38TokRunner{tok: tok, defect: ""}, oraclesFor(t, c))
		if err != nil {
			t.Fatalf("RunCase(clean): %v", err)
		}
		if !res.Pass {
			t.Fatalf("clean run should pass; got %s", Explain(res))
		}
		if res.FailureBundle != nil {
			t.Fatalf("clean run must not carry failure bundle: %+v", res.FailureBundle)
		}

		// Injected defect runs
		defects := []struct {
			name      string
			wantIndex int
		}{
			{name: "missing-im-start", wantIndex: 0},
			{name: "double-im-start", wantIndex: 1},
			{name: "corrupt-eos", wantIndex: firstIMEndIdx},
			{name: "think-seed-drift", wantIndex: thinkIdx},
		}

		for _, tc := range defects {
			runner := qwen38TokRunner{tok: tok, defect: tc.name}
			res, err := RunCase(c, ReferenceRunner{}, runner, oraclesFor(t, c))
			if err != nil {
				t.Fatalf("RunCase(%s): %v", tc.name, err)
			}
			if res.Pass {
				t.Fatalf("defect %q must not pass", tc.name)
			}
			fb := res.FailureBundle
			if fb == nil {
				t.Fatalf("defect %q must carry failure bundle", tc.name)
			}
			if fb.FailingOracle != "tokenizer-parity" {
				t.Fatalf("defect %q failing oracle = %q, want %q", tc.name, fb.FailingOracle, "tokenizer-parity")
			}
			if fb.FirstDivergence == nil {
				t.Fatalf("defect %q must have FirstDivergence", tc.name)
			}
			if fb.FirstDivergence.Index != tc.wantIndex {
				t.Fatalf("defect %q divergence index = %d, want %d", tc.name, fb.FirstDivergence.Index, tc.wantIndex)
			}
		}
	})

	t.Run("ToolCallXMLTokenizationParity", func(t *testing.T) {
		const toolCallXML = "<tool_call>\n<function=Read>\n<parameter=path>\nmain.go\n</parameter>\n</function>\n</tool_call>"
		toolIDs, err := tok.Encode(toolCallXML)
		if err != nil {
			t.Fatalf("tok.Encode(toolCallXML): %v", err)
		}
		if len(toolIDs) == 0 {
			t.Fatalf("tok.Encode(toolCallXML) returned empty slice")
		}

		decodedXML, err := tok.Decode(toolIDs)
		if err != nil {
			t.Fatalf("tok.Decode(toolIDs): %v", err)
		}
		if decodedXML != toolCallXML {
			t.Fatalf("tool call XML roundtrip mismatch:\ngot:  %q\nwant: %q", decodedXML, toolCallXML)
		}
	})
}

// TestQwen38UDQ2KXLTokenizerParity satisfies the exact issue #11948 witness selector.
func TestQwen38UDQ2KXLTokenizerParity(t *testing.T) {
	TestQwen38TokenizerParity(t)
}

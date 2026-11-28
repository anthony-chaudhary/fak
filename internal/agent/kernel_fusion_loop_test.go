package agent

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolgrammar"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

func buildFusionTestVocab() *toolgrammar.PrefixVocab {
	var toks []toolgrammar.VocabToken
	var id int32
	for b := 0; b < 256; b++ {
		toks = append(toks, toolgrammar.VocabToken{ID: id, Text: string([]byte{byte(b)})})
		id++
	}
	multi := []string{
		`{"`, `"}`, `"}}`, `":"`, `","`, `":{"`, `":`, `,"`,
		`name`, `arguments`, `query`, `top`, `_k`, `min`, `_score`, `include`, `_meta`,
		`search`, `_kb`, `write`, `_file`, `path`, `content`, `true`, `false`,
		`out`, `.txt`, `hello`, `kv`,
	}
	for _, m := range multi {
		toks = append(toks, toolgrammar.VocabToken{ID: id, Text: m})
		id++
	}
	return toolgrammar.NewPrefixVocab(toks)
}

// fak-test:runtime fast est=150ms
func TestKernelFusionEndToEndTurn(t *testing.T) {
	ctx := context.Background()
	vocab := buildFusionTestVocab()
	v := vdso.New(16)

	searchSchema := `{
		"name": "search_kb",
		"parameters": {
			"type": "object",
			"properties": {
				"query": {"type": "string"},
				"top_k": {"type": "integer"},
				"min_score": {"type": "number"},
				"include_meta": {"type": "boolean"}
			},
			"required": ["query", "top_k", "min_score", "include_meta"]
		}
	}`
	writeSchema := `{
		"name": "write_file",
		"parameters": {
			"type": "object",
			"properties": {
				"path": {"type": "string"},
				"content": {"type": "string"}
			},
			"required": ["path", "content"]
		}
	}`

	searchCG, err := toolgrammar.Compile([]byte(searchSchema), toolgrammar.GrammarOptions{
		Dialect:        toolgrammar.DialectJSON,
		IncludeToolTag: true,
	})
	if err != nil {
		t.Fatalf("compile searchSchema: %v", err)
	}
	writeCG, err := toolgrammar.Compile([]byte(writeSchema), toolgrammar.GrammarOptions{
		Dialect:        toolgrammar.DialectJSON,
		IncludeToolTag: true,
	})
	if err != nil {
		t.Fatalf("compile writeSchema: %v", err)
	}

	var readExecCalls int64
	var writeExecCalls int64
	ctrl, err := NewKernelFusionController(vocab, v, func(_ context.Context, call *abi.ToolCall) (*abi.Result, error) {
		switch call.Tool {
		case "search_kb":
			atomic.AddInt64(&readExecCalls, 1)
		case "write_file":
			atomic.AddInt64(&writeExecCalls, 1)
		}
		return &abi.Result{
			Call:    call,
			Status:  abi.StatusOK,
			Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"hits":["doc1"]}`)},
		}, nil
	})
	if err != nil {
		t.Fatalf("NewKernelFusionController: %v", err)
	}

	specs := []FusionToolCallSpec{
		{
			Index:   0,
			ID:      "call_ro_0",
			Tool:    "search_kb",
			Grammar: searchCG,
			Payload: `{"name":"search_kb","arguments":{"query":"kv","top_k":8,"min_score":0.9,"include_meta":true}}`,
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
			},
		},
		{
			Index:   1,
			ID:      "call_rw_1",
			Tool:    "write_file",
			Grammar: writeCG,
			Payload: `{"name":"write_file","arguments":{"path":"out.txt","content":"hello"}}`,
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
			},
		},
	}

	receipt, err := ctrl.RunTurn(ctx, specs)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	t.Logf("receipt=%+v", receipt)

	if receipt.StepReductionRatio < 2.0 {
		t.Fatalf("StepReductionRatio = %.2fx, want >= 2.0x (seq=%d, fused=%d)",
			receipt.StepReductionRatio, receipt.SequentialSteps, receipt.FusedSteps)
	}
	if receipt.PrefetchedHits != 1 {
		t.Fatalf("PrefetchedHits = %d, want 1", receipt.PrefetchedHits)
	}
	if receipt.RefusedWrites != 1 {
		t.Fatalf("RefusedWrites = %d, want 1", receipt.RefusedWrites)
	}
	if got := atomic.LoadInt64(&readExecCalls); got != 1 {
		t.Fatalf("readExecCalls = %d, want 1", got)
	}
	if got := atomic.LoadInt64(&writeExecCalls); got != 0 {
		t.Fatalf("writeExecCalls = %d, want 0 (write/destructive tool must never execute)", got)
	}
}

package headroom

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func TestLinguaCompressorUsesFirstClassAdapter(t *testing.T) {
	var got linguaRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/compress" {
			t.Errorf("path=%s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Error(err)
		}
		_ = json.NewEncoder(w).Encode(linguaResponse{Text: "critical fact", Model: "microsoft/llmlingua-2", OriginalTokens: 100, CompressedTokens: 10})
	}))
	defer srv.Close()
	t.Setenv("FAK_LINGUA_URL", srv.URL)
	out, err := (LinguaCompressor{}).Compress(context.Background(), Input{Bytes: []byte("long context containing critical fact")})
	if err != nil {
		t.Fatal(err)
	}
	if got.Text == "" || string(out.Bytes) != "critical fact" || !out.Compressed || out.Codec != "llmlingua-2" {
		t.Fatalf("request=%+v output=%+v", got, out)
	}
}

func TestLinguaRegisteredButNotDefault(t *testing.T) {
	if _, ok := Lookup(LinguaName); !ok {
		t.Fatal("lingua adapter is not registered")
	}
	t.Setenv("FAK_COMPRESSOR", "")
	if Selected().Name() == LinguaName {
		t.Fatal("lossy lingua adapter must never be default")
	}
}

func TestLinguaGatePreservesOriginalInCAS(t *testing.T) {
	orig := []byte(strings.Repeat("The customer account is 12345. The critical fact is BLUE ORCHID. Please preserve it. ", 12))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(linguaResponse{
			Text:             "customer account 12345 critical BLUE ORCHID preserve",
			Model:            "microsoft/llmlingua-2",
			OriginalTokens:   252,
			CompressedTokens: 11,
		})
	}))
	defer srv.Close()
	t.Setenv("FAK_LINGUA_URL", srv.URL)
	withSelected(t, LinguaName)

	result := &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: orig, Len: int64(len(orig))}}
	verdict := NewGate().Admit(context.Background(), &abi.ToolCall{Tool: "read_file"}, result)
	if verdict.Kind != abi.VerdictTransform {
		t.Fatalf("lingua gate verdict=%v", verdict.Kind)
	}
	if verdict.Meta["compressor"] != LinguaName || verdict.Meta["origin"] == "" {
		t.Fatalf("lingua gate metadata=%v", verdict.Meta)
	}
	got, err := abi.ActiveResolver().Resolve(context.Background(), abi.Ref{
		Kind: abi.RefBlob, Digest: verdict.Meta["origin"], Len: int64(len(orig)),
	})
	if err != nil {
		t.Fatalf("resolve preserved original: %v", err)
	}
	if string(got) != string(orig) {
		t.Fatal("preserved original did not round-trip")
	}
}

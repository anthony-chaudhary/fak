package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type gatewayTokenStreamDecoder interface {
	Push(int) (string, error)
	Finalize() (string, error)
}

func TestGatewayCanUseIncrementalTokenizer(t *testing.T) {
	tok, err := tokenizer.ParseJSON([]byte(`{
	  "model": {"type": "BPE", "vocab": {"H": 0, "i": 1}},
	  "decoder": {"type": "ByteLevel"}
	}`))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	var dec gatewayTokenStreamDecoder = tok.NewIncrementalDecoder()
	first, err := dec.Push(0)
	if err != nil {
		t.Fatalf("Push H: %v", err)
	}
	second, err := dec.Push(1)
	if err != nil {
		t.Fatalf("Push i: %v", err)
	}
	tail, err := dec.Finalize()
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	if got := first + second + tail; got != "Hi" {
		t.Fatalf("incremental gateway decode = %q, want Hi", got)
	}
}

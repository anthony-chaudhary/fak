package modelengine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type modelEngineTokenStreamDecoder interface {
	Push(int) (string, error)
	Finalize() (string, error)
}

func TestModelEngineCanUseIncrementalTokenizer(t *testing.T) {
	tok, err := tokenizer.ParseJSON([]byte(`{
	  "model": {"type": "BPE", "vocab": {"o": 0, "k": 1}},
	  "decoder": {"type": "ByteLevel"}
	}`))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	var dec modelEngineTokenStreamDecoder = tokenizer.NewIncrementalDecoder(tok)
	first, err := dec.Push(0)
	if err != nil {
		t.Fatalf("Push o: %v", err)
	}
	second, err := dec.Push(1)
	if err != nil {
		t.Fatalf("Push k: %v", err)
	}
	tail, err := dec.Finalize()
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	if got := first + second + tail; got != "ok" {
		t.Fatalf("incremental modelengine decode = %q, want ok", got)
	}
}

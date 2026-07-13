package agent

import (
	"bytes"
	"testing"
)

func TestDecodePreservesTypedBlocks(t *testing.T) {
	raw := []byte(`{"model":"x","max_tokens":1,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"u","name":"f","input":{"x":1}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"u","content":[{"type":"text","text":"ok"}]}]}]}`)
	got, err := DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.ContentBlocks) != 2 {
		t.Fatalf("blocks=%d", len(got.ContentBlocks))
	}
	want := [][]byte{[]byte(`[{"type":"tool_use","id":"u","name":"f","input":{"x":1}}]`), []byte(`[{"type":"tool_result","tool_use_id":"u","content":[{"type":"text","text":"ok"}]}]`)}
	for i := range want {
		if !bytes.Equal(got.ContentBlocks[i], want[i]) {
			t.Fatalf("block %d: %s", i, got.ContentBlocks[i])
		}
	}
}

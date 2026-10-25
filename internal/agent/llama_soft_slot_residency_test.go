package agent

import (
	"fmt"
	"slices"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/agent/llama_soft_slot_residency.go
// fak-test:runtime fast est=1ms lane=default
func TestLlamaPreparedPrefixTransportProgress(t *testing.T) {
	body := func(stream bool, progress, temperature, content string) []byte {
		return []byte(fmt.Sprintf(`{"id_slot":1,"cache_prompt":true,"model":"m","messages":[{"role":"user","content":%q}],"temperature":%s,"stream":%t%s}`, content, temperature, stream, progress))
	}
	baseEnvelope, baseChain, ok := llamaPreparedPrefix(body(false, "", "0.2", "hello"), 1)
	if !ok || len(baseChain) != 1 {
		t.Fatal("valid baseline prefix was rejected")
	}
	for _, stream := range []bool{false, true} {
		for _, progress := range []string{"", `,"return_progress":true`, `,"return_progress":false`} {
			envelope, chain, ok := llamaPreparedPrefix(body(stream, progress, "0.2", "hello"), 1)
			if !ok || envelope != baseEnvelope || !slices.Equal(chain, baseChain) {
				t.Errorf("transport-only option changed residency identity: stream=%t progress=%q", stream, progress)
			}
		}
	}
	changedEnvelope, _, ok := llamaPreparedPrefix(body(true, `,"return_progress":true`, "0.3", "hello"), 1)
	if !ok || changedEnvelope == baseEnvelope {
		t.Error("inference option change did not invalidate the envelope")
	}
	_, changedChain, ok := llamaPreparedPrefix(body(true, `,"return_progress":true`, "0.2", "different"), 1)
	if !ok || slices.Equal(changedChain, baseChain) {
		t.Error("message change did not invalidate the prefix")
	}
}

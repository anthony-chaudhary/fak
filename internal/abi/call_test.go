package abi

import "testing"

func TestInlineResultCopiesPayload(t *testing.T) {
	payload := []byte("ok")
	call, result := InlineResult("search", payload)
	payload[0] = 'x'
	if call.Tool != "search" || string(result.Payload.Inline) != "ok" || result.Call != call || result.Status != StatusOK {
		t.Fatalf("InlineResult = %#v, %#v", call, result)
	}
}

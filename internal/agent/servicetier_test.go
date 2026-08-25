package agent

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestOpenAIServiceTierRequestResponse(t *testing.T) {
	a := openAIAdapter{}
	base := adapterRequest{Model: "gpt-test", Messages: []Message{{Role: RoleUser, Content: "hi"}}, Temperature: 0, MaxTokens: 10}
	standard, err := a.MarshalRequest(base)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(standard, []byte("service_tier")) {
		t.Fatalf("standard bytes changed: %s", standard)
	}
	base.ServiceTier = modelroute.ServiceModeFast
	fast, err := a.MarshalRequest(base)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(fast, []byte(`"service_tier":"priority"`)) {
		t.Fatalf("fast request=%s", fast)
	}
	c, err := a.ParseResponse([]byte(`{"model":"gpt-test","service_tier":"default","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	if err != nil || c.ServiceTier != modelroute.ServiceModeStandard {
		t.Fatalf("completion=%+v err=%v", c, err)
	}
}

func TestAnthropicServiceTierRequestResponse(t *testing.T) {
	a := anthropicAdapter{}
	base := adapterRequest{Model: "claude-test", Messages: []Message{{Role: RoleUser, Content: "hi"}}, Temperature: 0, MaxTokens: 10}
	standard, err := a.MarshalRequest(base)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(standard, []byte("service_tier")) {
		t.Fatalf("standard bytes changed: %s", standard)
	}
	base.ServiceTier = modelroute.ServiceModeFast
	fast, err := a.MarshalRequest(base)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(fast, []byte(`"service_tier":"auto"`)) {
		t.Fatalf("fast request=%s", fast)
	}
	c, err := a.ParseResponse([]byte(`{"model":"claude-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,"service_tier":"standard"}}`))
	if err != nil || c.ServiceTier != modelroute.ServiceModeStandard {
		t.Fatalf("completion=%+v err=%v", c, err)
	}
}

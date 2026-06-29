package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// TestGatewayCanceledRequestContextCancelsInKernelDecode is the #32 gateway
// witness: the inbound request context reaches the native decode scheduler, so a
// disconnected/canceled client request becomes a real scheduler cancellation
// instead of run-to-completion-then-discard.
func TestGatewayCanceledRequestContextCancelsInKernelDecode(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, toolAdj{})

	sched := modelengine.NewNativeScheduler(model.NewSynthetic(modelengine.SyntheticConfig()))
	t.Cleanup(sched.Close)
	abi.RegisterEngine("native-sched-cancel", sched)

	srv, err := New(Config{EngineID: "native-sched-cancel", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New(scheduler gateway): %v", err)
	}
	t.Cleanup(srv.Close)

	_, ctrlEnv, err := srv.syscall(context.Background(), "allow_search", `{"q":"control"}`, false, "", "")
	if err != nil {
		t.Fatalf("control syscall: %v", err)
	}
	if got := decodeGatewayGeneratedTokens(t, ctrlEnv); len(got) == 0 {
		t.Fatal("control request produced no in-kernel tokens")
	}

	cctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, discEnv, err := srv.syscall(cctx, "allow_lookup", `{"id":"disconnect"}`, false, "", "")
	if err != nil {
		t.Fatalf("disconnect syscall returned transport error: %v", err)
	}
	if discEnv == nil {
		t.Fatal("disconnect syscall returned no envelope")
	}
	if discEnv.Status != "ERROR" {
		t.Fatalf("disconnect envelope status = %q, want ERROR; content=%q", discEnv.Status, discEnv.Content)
	}
	if e := discEnv.Meta["error"]; !strings.Contains(e, context.Canceled.Error()) {
		t.Fatalf("disconnect envelope error = %q, want it to carry %q", e, context.Canceled.Error())
	}
	if toks := tryDecodeGatewayTokens(discEnv); len(toks) != 0 {
		t.Fatalf("cancelled request produced %d completed decode tokens", len(toks))
	}

	_, againEnv, err := srv.syscall(context.Background(), "allow_list", `{"region":"EU"}`, false, "", "")
	if err != nil {
		t.Fatalf("post-cancel syscall: %v", err)
	}
	if got := decodeGatewayGeneratedTokens(t, againEnv); len(got) == 0 {
		t.Fatal("post-cancel request produced no in-kernel tokens")
	}
}

func tryDecodeGatewayTokens(env *ResultEnvelope) []int {
	if env == nil || env.Status != statusName(abi.StatusOK) {
		return nil
	}
	return decodeGatewayGeneratedTokensConc(env)
}

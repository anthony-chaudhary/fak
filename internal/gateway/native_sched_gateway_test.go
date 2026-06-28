package gateway

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// TestGatewayDrivesContinuousBatchingScheduler is the issue-#36 acceptance-#4 witness:
// the continuous-batching iteration scheduler — modelengine.NativeScheduler, which
// admits many requests and advances ALL live lanes with ONE model.BatchSession
// StepBatch call per iteration, freeing a lane the instant it finishes — is REACHABLE
// FROM THE LIVE GATEWAY REQUEST PATH, not only from the fleetserve bench harness.
//
// The request path under test is the production one: srv.syscall (what POST
// /v1/fak/syscall and the fak_syscall MCP tool both call) -> kernel.Syscall -> Reap ->
// EngineDriver.Complete. For the scheduler-backed engine that Complete drives Admit ->
// the shared StepBatch loop -> per-lane token stream -> reclaim, i.e. the admit/step/
// stream/reclaim seam the issue names ("Engine.Complete currently bypasses BatchSession
// entirely — that is the integration gap to close"). Driving three DISTINCT calls
// CONCURRENTLY admits up to three lanes into the one scheduler loop, so overlapping
// gateway requests are co-batched and then individually freed.
//
// The witness is OUTPUT EQUIVALENCE, not a throughput claim (the issue asserts none).
// The scheduler and a per-request in-kernel engine are driven over the SAME shared
// *model.Model, so a difference in generated tokens could only come from the scheduler.
// For every surviving sequence the gateway-served tokens are bit-identical to the
// per-request engine reached through the same gateway path — the StepBatch f32 guarantee
// (batch.go: bit-for-bit identical to serial Step) carried end-to-end through the
// gateway. That closes acceptance #3 (bit-identical surviving-sequence output) and #4
// (gateway-path reachability) in one gateway-lane test.
func TestGatewayDrivesContinuousBatchingScheduler(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, toolAdj{}) // allow* dispatch to the engine, deny* refuse

	// One shared synthetic checkpoint, driven two ways through the SAME EngineDriver
	// seam: the continuous-batching scheduler (batched) and the per-request in-kernel
	// engine (one Session per request). Sharing the *model.Model makes their decodes
	// weight-identical, so any divergence is the scheduler, never the weights.
	m := model.NewSynthetic(modelengine.SyntheticConfig())
	sched := modelengine.NewNativeScheduler(m)
	t.Cleanup(sched.Close) // stop the run loop; idempotent, lanes already drained
	ref := modelengine.New()
	ref.Preload(m) // same weights, no lazy synthetic build

	abi.RegisterEngine("native-sched", sched)
	abi.RegisterEngine("inkernel-ref", ref)

	srvSched, err := New(Config{EngineID: "native-sched", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New(scheduler gateway): %v", err)
	}
	t.Cleanup(srvSched.Close)
	srvRef, err := New(Config{EngineID: "inkernel-ref", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New(reference gateway): %v", err)
	}
	t.Cleanup(srvRef.Close)

	// Distinct allowed calls: distinct args keep the kernel vDSO from deduping them,
	// and distinct work lets >1 lane be live in the scheduler loop at once.
	calls := []struct{ tool, args string }{
		{"allow_search", `{"q":"flights SFO"}`},
		{"allow_lookup", `{"id":42}`},
		{"allow_list", `{"region":"EU"}`},
	}

	// Reference: the per-request in-kernel engine through the gateway (serial decode).
	refTokens := make([][]int, len(calls))
	for i, c := range calls {
		_, env, err := srvRef.syscall(context.Background(), c.tool, c.args, false, "", "")
		if err != nil {
			t.Fatalf("reference syscall %q: %v", c.tool, err)
		}
		refTokens[i] = decodeGatewayGeneratedTokens(t, env)
	}

	// Scheduler: the SAME calls driven CONCURRENTLY through the scheduler-backed
	// gateway. Each goroutine's kernel.Reap calls NativeScheduler.Complete, whose Admit
	// appends a lane to the one shared StepBatch loop; the loop fans each step's token
	// into that lane's stream and frees the lane on completion.
	schedTokens := make([][]int, len(calls))
	errs := make([]error, len(calls))
	var wg sync.WaitGroup
	for i, c := range calls {
		wg.Add(1)
		go func(i int, tool, args string) {
			defer wg.Done()
			_, env, e := srvSched.syscall(context.Background(), tool, args, false, "", "")
			if e != nil {
				errs[i] = e
				return
			}
			schedTokens[i] = decodeGatewayGeneratedTokensConc(env)
		}(i, c.tool, c.args)
	}
	wg.Wait()

	for i, c := range calls {
		if errs[i] != nil {
			t.Fatalf("scheduler syscall %q: %v", c.tool, errs[i])
		}
		if len(schedTokens[i]) == 0 {
			t.Fatalf("scheduler call %q returned no generated tokens", c.tool)
		}
		// Bit-identical surviving-sequence output: the continuous-batching scheduler,
		// reached through the live gateway, decodes exactly what the per-request engine
		// does over the same weights.
		if !reflect.DeepEqual(schedTokens[i], refTokens[i]) {
			t.Fatalf("call %q: scheduler tokens %v != per-request reference %v",
				c.tool, schedTokens[i], refTokens[i])
		}
	}
}

// decodeGatewayGeneratedTokens parses the in-kernel decode payload a gateway syscall
// returns and asserts it is a real in-kernel decode (engine="inkernel", non-empty
// generated tokens) rather than a mock/echo. Fails the test on any mismatch.
func decodeGatewayGeneratedTokens(t *testing.T, env *ResultEnvelope) []int {
	t.Helper()
	if env == nil {
		t.Fatal("nil result envelope from gateway syscall")
	}
	if env.Status != statusName(abi.StatusOK) {
		t.Fatalf("result status = %q, want OK (payload %q)", env.Status, env.Content)
	}
	var p struct {
		Engine string `json:"engine"`
		Tokens []int  `json:"generated_tokens"`
	}
	if err := json.Unmarshal([]byte(env.Content), &p); err != nil {
		t.Fatalf("decode in-kernel payload %q: %v", env.Content, err)
	}
	if p.Engine != modelengine.EngineID {
		t.Fatalf("payload engine = %q, want %q (a real in-kernel decode, not a mock)",
			p.Engine, modelengine.EngineID)
	}
	if len(p.Tokens) == 0 {
		t.Fatalf("in-kernel payload carried no generated tokens: %q", env.Content)
	}
	return p.Tokens
}

// decodeGatewayGeneratedTokensConc is the *testing.T-free decode used inside the
// concurrent goroutines (calling t.Fatalf off the test goroutine is illegal). It
// returns nil on any malformed/empty payload; the caller asserts non-empty + equality
// back on the test goroutine.
func decodeGatewayGeneratedTokensConc(env *ResultEnvelope) []int {
	if env == nil || env.Status != statusName(abi.StatusOK) {
		return nil
	}
	var p struct {
		Engine string `json:"engine"`
		Tokens []int  `json:"generated_tokens"`
	}
	if err := json.Unmarshal([]byte(env.Content), &p); err != nil || p.Engine != modelengine.EngineID {
		return nil
	}
	return p.Tokens
}

package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
)

type inputClaimWitness struct {
	mu       sync.Mutex
	claims   []AdmittedInputClaim
	bindings []InputClaimBinding
	releases []string
}

func (w *inputClaimWitness) lifecycle() InputClaimLifecycle {
	return InputClaimLifecycle{
		Claim: func(claim AdmittedInputClaim) (InputClaimBinding, error) {
			w.mu.Lock()
			defer w.mu.Unlock()
			binding := InputClaimBinding{
				ID: "claim-" + string(rune('0'+len(w.claims)+1)), SHA256: "sha256:witness", Count: len(claim.Inputs),
			}
			w.claims = append(w.claims, claim)
			w.bindings = append(w.bindings, binding)
			return binding, nil
		},
		Release: func(binding InputClaimBinding, reason string) error {
			w.mu.Lock()
			defer w.mu.Unlock()
			w.releases = append(w.releases, binding.ID+":"+reason)
			return nil
		},
	}
}

func TestInputClaimSeparatesArrivalsWhilePromptAssemblyBlocked(t *testing.T) {
	const trace = "input-claim-blocked-assembly"
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
	enqueueInputClaimSteer(t, key, "A")
	t.Cleanup(func() { drainInputClaimSteers(key) })

	witness := &inputClaimWitness{}
	assemblyEntered := make(chan struct{})
	continueAssembly := make(chan struct{})
	var assemblyCalls atomic.Int32
	assembler := func(_ context.Context, messages []Message) ([]Message, error) {
		if assemblyCalls.Add(1) == 1 {
			close(assemblyEntered)
			<-continueAssembly
		}
		return messages, nil
	}
	var boundaries []ModelRequestBoundary
	observe := func(boundary ModelRequestBoundary) error {
		boundaries = append(boundaries, boundary)
		return nil
	}

	first := make(chan error, 1)
	go func() {
		_, err := RunArm(context.Background(), &recordingPlanner{}, "first", false, 1, nil,
			WithSessionTable(nil, trace), WithInputClaimLifecycle(witness.lifecycle()),
			WithPromptAssembler(assembler), WithModelRequestObserver(observe))
		first <- err
	}()
	<-assemblyEntered
	enqueueInputClaimSteer(t, key, "B")
	close(continueAssembly)
	if err := <-first; err != nil {
		t.Fatalf("request 1: %v", err)
	}
	if _, err := RunArm(context.Background(), &recordingPlanner{}, "second", false, 1, nil,
		WithSessionTable(nil, trace), WithInputClaimLifecycle(witness.lifecycle()),
		WithPromptAssembler(assembler), WithModelRequestObserver(observe)); err != nil {
		t.Fatalf("request 2: %v", err)
	}

	if len(witness.claims) != 2 || len(boundaries) != 2 {
		t.Fatalf("claims=%d boundaries=%d, want 2/2", len(witness.claims), len(boundaries))
	}
	for i, want := range []string{"A", "B"} {
		claim := witness.claims[i]
		if len(claim.Inputs) != 1 || claim.Inputs[0].Content != want {
			t.Fatalf("claim %d inputs = %+v, want exactly %q", i+1, claim.Inputs, want)
		}
		if boundaries[i].InputClaim == nil || boundaries[i].InputClaim.ID != witness.bindings[i].ID {
			t.Fatalf("boundary %d claim = %+v, binding = %+v", i+1, boundaries[i].InputClaim, witness.bindings[i])
		}
		if !claimedInputsSurviveAssembly(claim.Inputs, boundaries[i].Messages) {
			t.Fatalf("boundary %d omitted its claimed input: claim=%+v messages=%+v", i+1, claim.Inputs, boundaries[i].Messages)
		}
	}
	if len(witness.releases) != 0 {
		t.Fatalf("successful requests released claims: %v", witness.releases)
	}
}

func TestInputClaimReleasedWhenPromptAssemblyDropsClaimedInput(t *testing.T) {
	const trace = "input-claim-assembly-drop"
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
	enqueueInputClaimSteer(t, key, "A")
	t.Cleanup(func() { drainInputClaimSteers(key) })

	witness := &inputClaimWitness{}
	planner := &recordingPlanner{}
	_, err := RunArm(context.Background(), planner, "drop", false, 1, nil,
		WithSessionTable(nil, trace), WithInputClaimLifecycle(witness.lifecycle()),
		WithPromptAssembler(func(_ context.Context, messages []Message) ([]Message, error) {
			return messages[:len(messages)-1], nil
		}), WithModelRequestObserver(func(ModelRequestBoundary) error {
			t.Fatal("request observer ran after assembly dropped claimed input")
			return nil
		}))
	if err == nil || !strings.Contains(err.Error(), "dropped claimed input") {
		t.Fatalf("RunArm error = %v, want dropped claimed input", err)
	}
	if len(witness.releases) != 1 || witness.releases[0] != "claim-1:PROMPT_ASSEMBLY_DROPPED_CLAIMED_INPUT" {
		t.Fatalf("claim releases = %v", witness.releases)
	}
	if len(planner.seen) != 0 {
		t.Fatalf("planner saw %d messages after claimed input was dropped", len(planner.seen))
	}
}

func TestInputClaimReleasedOnceWhenPromptAssemblyFails(t *testing.T) {
	const trace = "input-claim-assembly-failure"
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
	enqueueInputClaimSteer(t, key, "A")
	t.Cleanup(func() { drainInputClaimSteers(key) })

	witness := &inputClaimWitness{}
	planner := &recordingPlanner{}
	_, err := RunArm(context.Background(), planner, "failure", false, 1, nil,
		WithSessionTable(nil, trace), WithInputClaimLifecycle(witness.lifecycle()),
		WithPromptAssembler(func(context.Context, []Message) ([]Message, error) {
			return nil, errors.New("assembly fixture failed")
		}), WithModelRequestObserver(func(ModelRequestBoundary) error {
			t.Fatal("request observer ran after assembly failure")
			return nil
		}))
	if err == nil || !strings.Contains(err.Error(), "prompt assembly") {
		t.Fatalf("RunArm error = %v, want prompt assembly failure", err)
	}
	if len(witness.claims) != 1 || len(witness.releases) != 1 || witness.releases[0] != "claim-1:PROMPT_ASSEMBLY_FAILED" {
		t.Fatalf("claim lifecycle = claims %+v releases %v", witness.claims, witness.releases)
	}
	if len(planner.seen) != 0 {
		t.Fatalf("planner saw %d messages after assembly failure", len(planner.seen))
	}
}

func enqueueInputClaimSteer(t *testing.T, key a2achan.ChannelKey, payload string) {
	t.Helper()
	if v := a2achan.Default.Send(context.Background(), "operator", key, a2achan.Shared([]byte(payload)), a2achan.CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("enqueue %q: %v", payload, v.Kind)
	}
}

func drainInputClaimSteers(key a2achan.ChannelKey) {
	for {
		if _, _, ok := a2achan.Default.TryRecv(context.Background(), key, a2achan.CapA2ARecv); !ok {
			return
		}
	}
}

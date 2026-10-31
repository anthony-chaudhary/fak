package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41LegacyDeferredProviderOwner struct {
	compute.Backend
	*v41DeferredLedger
}

type v41ExplicitDeferredProviderOwner struct {
	*v41LegacyDeferredProviderOwner
	selected compute.V41DeferredStatusBackend
	admit    bool
}

func (b *v41ExplicitDeferredProviderOwner) V41DeferredStatusForModel() (compute.V41DeferredStatusBackend, bool) {
	return b.selected, b.admit
}

// fak-test:runtime fast est=1ms lane=default
// Actual model publication/readback consumers select an operator separately
// from tensor ownership. Host ledger arithmetic is wiring proof only.
func TestV41DeferredProviderModelReadback(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, mode := range []string{"provided", "refused", "nil-admitted", "legacy"} {
		t.Run(mode, func(t *testing.T) {
			legacy := &v41LegacyDeferredProviderOwner{Backend: compute.Default()}
			var owner compute.Backend = legacy
			provider := &v41ExplicitDeferredProviderOwner{v41LegacyDeferredProviderOwner: legacy, admit: true}
			if mode != "legacy" {
				owner = provider
			}
			legacy.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return owner })
			selected := newV41DeferredLedger(func() compute.Backend { return owner })
			provider.selected = selected
			if mode == "refused" {
				provider.admit = false
			}
			if mode == "nil-admitted" {
				provider.selected = nil
			}
			deferred, ok := v41DeferredBF16Publisher(owner, 2)
			if mode == "refused" || mode == "nil-admitted" {
				if ok || deferred != nil {
					t.Fatal("provider refusal fell back to direct deferred interface")
				}
				if legacy.deferredPublishes != 0 || selected.deferredPublishes != 0 {
					t.Fatal("refusal issued work")
				}
				return
			}
			if !ok {
				t.Fatal("supported model consumer did not bind")
			}
			if mode == "provided" && deferred != selected {
				t.Fatal("provider selection was replaced by legacy interface")
			}
			input := compute.NewF32(owner, []int{2}, []float32{1, -2})
			out, status, err := deferred.V41BF16PublishDeferred(input, compute.V41BF16PublishRoutedDown)
			if err != nil {
				t.Fatal(err)
			}
			if input.Backend() != owner || out.Backend() != owner || status.Record.Backend() != owner {
				t.Fatal("separate deferred operator changed original backend identity")
			}
			s := &Session{Backend: owner}
			row, verdict, err := s.v41ReadResidentBF16(out, status)
			if err != nil || verdict != nil || !reflect.DeepEqual(row, []float32{1, -2}) {
				t.Fatalf("actual model readback failed: %v %v %v", row, verdict, err)
			}
			owner.Free(out)
			if mode == "provided" {
				if selected.deferredPublishes != 1 || selected.deferredFences != 1 || legacy.deferredPublishes != 0 || legacy.deferredFences != 0 {
					t.Fatal("model consumer bypassed selected operator")
				}
			} else if legacy.deferredPublishes != 1 || legacy.deferredFences != 1 {
				t.Fatal("legacy direct-interface behavior changed")
			}
		})
	}
}

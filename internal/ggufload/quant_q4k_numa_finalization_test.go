package ggufload

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Observe finalization ordering with NUMA disabled; no worker or replica release is measured.
// fak-test:runtime fast est=200ms lane=default
func TestQ4KLateRefusalPrecedesNUMAFinalization(t *testing.T) {
	t.Setenv("FAK_NUMA_REPLICAS", "off")
	t.Setenv("FAK_NUMA_INTERLEAVE", "off")
	var acceptedLog bytes.Buffer
	acceptedProfile := NewLoadProfiler()
	acceptedProfile.Progress = &acceptedLog
	accepted, err := LoadModelQ4KStreamedDenseContext(context.Background(), writeV41UnboundEngramFixture(t, true), acceptedProfile)
	if accepted != nil {
		t.Cleanup(func() {
			if err := accepted.CloseWeights(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil || accepted == nil || !accepted.V41EngramAttached() {
		t.Fatalf("accepted control: model=%v err=%v", accepted != nil, err)
	}
	if strings.Count(acceptedLog.String(), "numa: ") != 1 || strings.Count(acceptedLog.String(), "numa_replicas: ") != 1 {
		t.Fatalf("accepted control omitted or repeated NUMA finalization labels: %q", acceptedLog.String())
	}
	if !strings.Contains(acceptedLog.String(), "numa_replicas=disabled(reason=disabled_by_request)") {
		t.Fatal("accepted control did not honor disabled NUMA replicas")
	}
	if err := accepted.CloseWeights(); err != nil {
		t.Fatal(err)
	}

	var refusedLog bytes.Buffer
	refusedProfile := NewLoadProfiler()
	refusedProfile.Progress = &refusedLog
	refusedSource, err := OpenWeights(writeV41DeclaredEngramNoTableFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = refusedSource.Close() })
	refused, err := refusedSource.QuantModelQ4KProfileOptionsContext(context.Background(), refusedProfile, WithStreamedDenseQ4K(true))
	if refused != nil {
		_ = refused.CloseWeights()
		t.Fatal("late Engram refusal returned a partial model")
	}
	if !errors.Is(err, model.ErrV41NativeUnsupported) || !strings.Contains(strings.ToLower(err.Error()), "engram") {
		t.Fatalf("late refusal lost its typed Engram error: %v", err)
	}
	if err := refusedSource.Close(); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(refusedLog.String(), "numa: ") || strings.Contains(refusedLog.String(), "numa_replicas: ") {
		t.Error("late-refused model reached NUMA finalization")
	}
}

package model

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type qwen35InitFailureBackend struct {
	*recordingQwen35Backend
	attempts int
	failAt   int
	mode     string
	primary  error
	uploaded []compute.Buffer
}

func (b *qwen35InitFailureBackend) UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if class != compute.MemoryKVCache || !strings.HasPrefix(site, "qwen35-gdn-") {
		return b.recordingQwen35Backend.UploadClass(t, as, class, site)
	}
	b.attempts++
	if b.attempts == b.failAt {
		siblingFault(b.mode, b.primary)
	}
	out := b.recordingQwen35Backend.UploadClass(t, as, class, site)
	b.uploaded = append(b.uploaded, out.Buf())
	return out
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35HALInitFailureReleasesUnpublishedState(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	uploads := 2 * linearQwen35Layers(cfg)
	for _, mode := range []string{"panic", "goexit"} {
		for failAt := 1; failAt <= uploads; failAt++ {
			t.Run(fmt.Sprintf("%s/upload-%d", mode, failAt), func(t *testing.T) {
				testQwen35HALInitFailure(t, cfg, mode, failAt, false)
			})
		}
		t.Run(mode+"/secondary-cleanup-panic", func(t *testing.T) {
			testQwen35HALInitFailure(t, cfg, mode, uploads, true)
		})
	}
}

func testQwen35HALInitFailure(t *testing.T, cfg Config, mode string, failAt int, cleanupPanic bool) {
	t.Helper()
	m := NewSynthetic(cfg)
	t.Cleanup(func() { _ = m.CloseWeights() })
	be := &qwen35InitFailureBackend{
		recordingQwen35Backend: newRecordingQwen35Backend(m),
		failAt:                 failAt, mode: mode, primary: errors.New("injected recurrent state upload failure"),
	}
	t.Cleanup(be.reference.Close)
	m.weightCloser.mu.Lock()
	holdsBefore := m.weightCloser.sessions
	m.weightCloser.mu.Unlock()
	if cleanupPanic {
		be.freeHook = func() { panic("secondary state cleanup failure") }
	}
	var session *Session
	out := siblingRun(func() {
		var err error
		session, err = m.NewBackendSessionChecked(be)
		if err != nil {
			panic(err)
		}
	})
	if out.returned || session != nil || (mode == "panic" && (!out.resumed || out.value != be.primary)) ||
		(mode == "goexit" && (out.resumed || out.value != nil)) {
		t.Fatalf("constructor outcome=%+v session=%p, want original %s", out, session, mode)
	}
	if be.attempts != failAt || len(be.uploaded) != failAt-1 {
		t.Fatalf("upload attempts=%d returned=%d, want %d/%d", be.attempts, len(be.uploaded), failAt, failAt-1)
	}
	for _, buffer := range be.uploaded {
		if got := be.freeCalls[buffer]; got != 1 {
			t.Errorf("unpublished state %p freed %d times, want exactly once", buffer, got)
		}
	}
	if len(be.freeCalls) != len(be.uploaded) {
		t.Errorf("cleanup touched %d buffers, want only %d returned state buffers", len(be.freeCalls), len(be.uploaded))
	}
	m.weightCloser.mu.Lock()
	holdsAfter := m.weightCloser.sessions
	m.weightCloser.mu.Unlock()
	if holdsAfter != holdsBefore {
		t.Errorf("constructor retained model holds: before=%d after=%d", holdsBefore, holdsAfter)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35HALInitPublishesOwnedState(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	t.Cleanup(func() { _ = m.CloseWeights() })
	be := &qwen35InitFailureBackend{recordingQwen35Backend: newRecordingQwen35Backend(m)}
	t.Cleanup(be.reference.Close)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	want := 2 * len(be.linearLayers)
	if s.qwen35HAL == nil || len(be.uploaded) != want || len(be.freeCalls) != 0 {
		t.Fatalf("publication state=%p uploads=%d frees=%d, want %d owned uploads and no frees", s.qwen35HAL, len(be.uploaded), len(be.freeCalls), want)
	}
	for _, layer := range be.linearLayers {
		state := s.qwen35HAL.layers[layer]
		if state.owner == nil || state.owner.refs != 1 || state.conv.Buf() == nil || state.recurrent.Buf() == nil ||
			state.owner.conv.Buf() != state.conv.Buf() || state.owner.recurrent.Buf() != state.recurrent.Buf() {
			t.Fatalf("layer %d did not publish one complete private owner", layer)
		}
	}
	s.Close()
	s.Close()
	for _, buffer := range be.uploaded {
		if got := be.freeCalls[buffer]; got != 1 {
			t.Errorf("published state %p freed %d times after repeated Close, want exactly once", buffer, got)
		}
	}
}

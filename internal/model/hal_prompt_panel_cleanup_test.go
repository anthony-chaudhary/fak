package model

import (
	"errors"
	"runtime"
	"testing"
)

type promptPanelCleanupFailureBackend struct {
	*promptPanelRecordingBackend
	chunkFailure func()
	cleanupError error
}

func (b *promptPanelCleanupFailureBackend) PromptPanelChunk(rows int) {
	b.promptPanelRecordingBackend.PromptPanelChunk(rows)
	if b.chunkFailure != nil {
		b.chunkFailure()
	}
}

func (b *promptPanelCleanupFailureBackend) PromptPanelEnd() {
	b.ends++
	panic(b.cleanupError)
}

// One cleanup attempt must preserve abnormal body unwind without hiding a
// normal-path cleanup fault. No CUDA runtime or device is used by this witness.
// fak-test:runtime fast est=20ms lane=default
func TestHALPromptPanelCleanupPreservesUnwind(t *testing.T) {
	bodyError, cleanupError := errors.New("body fault"), errors.New("cleanup fault")
	for _, kind := range []string{"body-panic", "body-goexit", "cleanup-panic"} {
		t.Run(kind, func(t *testing.T) {
			be := &promptPanelCleanupFailureBackend{
				promptPanelRecordingBackend: newPromptPanelRecordingBackend(),
				cleanupError:                cleanupError,
			}
			ids := []int{3}
			want := any(bodyError)
			switch kind {
			case "body-panic":
				be.chunkFailure = func() { panic(bodyError) }
			case "body-goexit":
				be.chunkFailure = runtime.Goexit
				want = nil
			case "cleanup-panic":
				ids, want = nil, cleanupError
			}
			s := promptPanelBiasModel().NewBackendSession(be)
			defer s.Close()
			type result struct {
				panicValue any
				returned   bool
			}
			done := make(chan result, 1)
			go func() {
				var got result
				defer func() {
					got.panicValue = recover()
					done <- got
				}()
				s.prefillHALPanel(be, ids, false)
				got.returned = true
			}()
			got := <-done
			if got.panicValue != want || got.returned || be.ends != 1 {
				t.Fatalf("panic=%v returned=%v cleanup attempts=%d; want panic=%v, no return, one cleanup", got.panicValue, got.returned, be.ends, want)
			}
		})
	}
}

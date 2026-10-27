package modelengine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak-test:runtime fast est=100ms
func TestNativeSchedulerFullPrefixWithoutLogitsUsesBoundedPrefill(t *testing.T) {
	const budget = nativeQwenPrefillMinChunkTokens
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(3 * budget)
	s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{
		"decode":         nativeSchedulerQwenPrompt(budget),
		"missing-logits": prompt,
	}))
	if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
		t.Fatal(err)
	}
	nativeSchedulerBeginManualDrain(t, s)
	defer nativeSchedulerEndManualDrain(s)
	decode := nativeSchedulerAdmitLane(t, s, "decode")
	defer decode.Cancel()
	nativeSchedulerDriveIteration(t, s)
	if got := nativeSchedulerDrainAvailable(decode); len(got) != 1 {
		t.Fatalf("initial decode tokens = %v, want one", got)
	}

	control := m.NewSession()
	control.Quant = true
	control.Q4K = true
	wantLogits := copyF32(control.Prefill(prompt))
	tree := radixkv.New(0)
	boundary, matched := tree.Lookup(prompt)
	tree.Done(tree.InsertWithLogits(boundary, prompt[matched:], control.Cache, nil))
	control.Close()
	s.SetRadixKV(tree)

	req := nativeSchedulerAdmitLane(t, s, "missing-logits")
	defer req.Cancel()
	s.mu.Lock()
	s.applyPrefixHitsLocked()
	cursor, cacheLen, state := req.promptCursor, req.sess.Cache.Len(), req.state
	s.mu.Unlock()
	if cursor != 0 || cacheLen != 0 || state != schedLanePrefilling || len(req.logits) != 0 {
		t.Fatalf("missing-logit hit installed state: cursor=%d cache=%d state=%d logits=%d", cursor, cacheLen, state, len(req.logits))
	}

	var chunks []nativeSchedulerEvent
	var finalLogits []float32
	s.observeNativeEvent = func(ev nativeSchedulerEvent) {
		if ev.Lane != req {
			return
		}
		if ev.Kind == nativeSchedulerEventPrefill {
			chunks = append(chunks, ev)
		}
		if ev.Kind == nativeSchedulerEventTransition {
			finalLogits = copyF32(req.logits)
		}
	}
	for i := 0; i < 3; i++ {
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(decode); len(got) != 1 {
			t.Fatalf("chunk %d interleaved decode tokens = %v, want one", i, got)
		}
		out := nativeSchedulerDrainAvailable(req)
		if i < 2 && len(out) != 0 {
			t.Fatalf("chunk %d emitted before final logits: %v", i, out)
		}
		if i == 2 && (len(out) != 1 || out[0] != argmax(wantLogits)) {
			t.Fatalf("first continuation = %v, want [%d]", out, argmax(wantLogits))
		}
	}
	if len(chunks) != 3 {
		t.Fatalf("prefill chunks = %d, want three bounded chunks", len(chunks))
	}
	for i, chunk := range chunks {
		if chunk.ChunkStart != i*budget || chunk.ChunkLen != budget || (i > 0 && chunk.Iteration <= chunks[i-1].Iteration) {
			t.Fatalf("chunk %d has start=%d len=%d iteration=%d", i, chunk.ChunkStart, chunk.ChunkLen, chunk.Iteration)
		}
	}
	nativeSchedulerAssertLogitParity(t, finalLogits, wantLogits)
	if req.sess.Cache.Len() != len(prompt)+1 {
		t.Fatalf("continuation cache length = %d, want %d", req.sess.Cache.Len(), len(prompt)+1)
	}
	boundary, matched = tree.Lookup(prompt)
	cachedLogits := copyF32(boundary.Logits())
	tree.Done(boundary)
	if matched != len(prompt) || len(cachedLogits) == 0 {
		t.Fatalf("completed boundary matched=%d logits=%d, want full prompt with logits", matched, len(cachedLogits))
	}
	nativeSchedulerAssertLogitParity(t, cachedLogits, wantLogits)
}

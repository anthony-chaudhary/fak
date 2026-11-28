package engine

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
)

// fak-test:runtime fast est=25ms lane=default
func TestProofLeakL2_ContinuousBatcherCompletedBoundStateMachine(t *testing.T) {
	t.Parallel()

	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 8
	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher: %v", err)
	}
	t.Cleanup(func() { _ = cb.Close() })

	const bound = 16
	cb.mu.Lock()
	cb.maxCompletedSlots = bound
	cb.mu.Unlock()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(2026112802))

	// Ghost model of active/queued sessions and retired sessions eligible for
	// acknowledgement or re-admission.
	live := make(map[string]uint64)    // sessionID -> SubmissionSeq
	retired := make(map[string]uint64) // sessionID -> last SubmissionSeq
	var liveIDs []string
	var retiredIDs []string
	nextNewID := 0

	removeLive := func(id string) {
		delete(live, id)
		for i, cur := range liveIDs {
			if cur == id {
				liveIDs[i] = liveIDs[len(liveIDs)-1]
				liveIDs = liveIDs[:len(liveIDs)-1]
				return
			}
		}
	}
	addRetired := func(id string, seq uint64) {
		if _, exists := retired[id]; !exists {
			retiredIDs = append(retiredIDs, id)
		}
		retired[id] = seq
	}
	removeRetired := func(id string) {
		delete(retired, id)
		for i, cur := range retiredIDs {
			if cur == id {
				retiredIDs[i] = retiredIDs[len(retiredIDs)-1]
				retiredIDs = retiredIDs[:len(retiredIDs)-1]
				return
			}
		}
	}

	const steps = 1200
	for step := 0; step < steps; step++ {
		op := rng.Intn(5)
		switch op {
		case 0: // Admit new session
			id := fmt.Sprintf("sess-%d", nextNewID)
			nextNewID++
			target := 1 + rng.Intn(3)
			req := &SubagentRequest{
				SessionID:    id,
				PromptTokens: []int{1, 2},
				TargetTokens: target,
			}
			gotID, err := cb.Submit(req)
			if err != nil {
				t.Fatalf("step %d: Submit(%s): %v", step, id, err)
			}
			live[gotID] = req.SubmissionSeq
			liveIDs = append(liveIDs, gotID)
			removeRetired(gotID)

		case 1: // Re-admit a previously retired/cancelled session ID
			if len(retiredIDs) > 0 {
				id := retiredIDs[rng.Intn(len(retiredIDs))]
				target := 1 + rng.Intn(3)
				req := &SubagentRequest{
					SessionID:    id,
					PromptTokens: []int{3, 4},
					TargetTokens: target,
				}
				gotID, err := cb.Submit(req)
				if err != nil {
					t.Fatalf("step %d: re-admit Submit(%s): %v", step, id, err)
				}
				live[gotID] = req.SubmissionSeq
				liveIDs = append(liveIDs, gotID)
				removeRetired(gotID)
			}

		case 2: // Cancel an active or queued session
			if len(liveIDs) > 0 {
				id := liveIDs[rng.Intn(len(liveIDs))]
				seq := live[id]
				cb.mu.Lock()
				_, wasAdmitted := cb.sessionMap[id]
				cb.mu.Unlock()
				if err := cb.Cancel(id); err != nil {
					t.Fatalf("step %d: Cancel(%s): %v", step, id, err)
				}
				removeLive(id)
				if wasAdmitted {
					addRetired(id, seq)
				}
			}

		case 3: // Step / complete
			res, err := cb.Step(ctx)
			if err != nil {
				t.Fatalf("step %d: Step: %v", step, err)
			}
			for _, id := range res.RetiredSessionIDs {
				if seq, ok := live[id]; ok {
					removeLive(id)
					addRetired(id, seq)
				}
			}

		case 4: // Acknowledge a retired session (same contract as BatchingEngine.acknowledgeRequest)
			if len(retiredIDs) > 0 {
				id := retiredIDs[rng.Intn(len(retiredIDs))]
				seq := retired[id]
				cb.mu.Lock()
				if slot := cb.completedMap[id]; slot != nil && slot.SubmissionSeq == seq {
					delete(cb.completedMap, id)
				}
				cb.mu.Unlock()
				removeRetired(id)
			}
		}

		// Assert invariants after EVERY transition.
		cb.mu.Lock()
		completedLen := len(cb.completedMap)
		orderLen := len(cb.completedOrder)
		limit := cb.completedLimit()
		sessionCount := len(cb.sessionMap)
		waitingCount := len(cb.waitingQueue)

		if completedLen > limit || completedLen > defaultMaxCompletedSlots {
			cb.mu.Unlock()
			t.Fatalf("step %d (op=%d): len(completedMap) = %d, want <= %d (defaultMax=%d)",
				step, op, completedLen, limit, defaultMaxCompletedSlots)
		}
		if orderLen > 2*limit {
			cb.mu.Unlock()
			t.Fatalf("step %d (op=%d): len(completedOrder) = %d, want <= %d",
				step, op, orderLen, 2*limit)
		}
		if sessionCount+waitingCount != len(live) {
			cb.mu.Unlock()
			t.Fatalf("step %d (op=%d): active(%d)+waiting(%d) = %d, want %d",
				step, op, sessionCount, waitingCount, sessionCount+waitingCount, len(live))
		}
		for id, seq := range live {
			if s, ok := cb.sessionMap[id]; ok {
				if s.SubmissionSeq != seq {
					cb.mu.Unlock()
					t.Fatalf("step %d (op=%d): active slot %s seq=%d, want %d",
						step, op, id, s.SubmissionSeq, seq)
				}
				continue
			}
			foundWaiting := false
			for _, q := range cb.waitingQueue {
				if q.SessionID == id && q.SubmissionSeq == seq {
					foundWaiting = true
					break
				}
			}
			if !foundWaiting {
				cb.mu.Unlock()
				t.Fatalf("step %d (op=%d): active/queued session %s (seq=%d) lost",
					step, op, id, seq)
			}
		}
		cb.mu.Unlock()
	}
}

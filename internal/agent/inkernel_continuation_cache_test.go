package agent

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// TestInKernelContinuationCache pins the serving contract at the point where decode state
// and emitted tokens differ: the terminal sampled token is returned to the client but is not
// forwarded through Session.Step, so only the earlier evaluated continuation is reusable.
func TestInKernelContinuationCache(t *testing.T) {
	for _, batch := range []bool{false, true} {
		name := "serial"
		if batch {
			name = "batched"
		}
		t.Run(name, func(t *testing.T) {
			cfg := tinyCfg()
			p := reusePlanner(true, false, cfg)
			p.batchDecode = batch
			prompt := synthIDs(cfg.VocabSize, 18, 12716)
			generated, matched := decode(p, prompt, 5)
			if matched != 0 || len(generated) != 5 {
				t.Fatalf("cold turn generated=%d matched=%d, want 5/0", len(generated), matched)
			}

			evaluated := append(append([]int(nil), prompt...), generated[:len(generated)-1]...)
			if got := p.cachedPrefixLen(evaluated); got != len(evaluated) {
				t.Fatalf("evaluated continuation cached %d/%d tokens", got, len(evaluated))
			}
			terminal := append(append([]int(nil), evaluated...), generated[len(generated)-1])
			if got := p.cachedPrefixLen(terminal); got != len(evaluated) {
				t.Fatalf("unforwarded terminal token entered cache: matched=%d want=%d", got, len(evaluated))
			}

			warmOut, warmMatched := decode(p, evaluated, 3)
			cold := reusePlanner(false, false, cfg)
			coldOut, coldMatched := decode(cold, evaluated, 3)
			if warmMatched != len(evaluated) || coldMatched != 0 {
				t.Fatalf("warm/cold matched=%d/%d, want %d/0", warmMatched, coldMatched, len(evaluated))
			}
			if !eqInts(warmOut, coldOut) {
				t.Fatalf("continuation changed output: warm=%v cold=%v", warmOut, coldOut)
			}
		})
	}
}

// TestInKernelContinuationCacheTwoTurnWorkReceipt compares the complete two-turn workload,
// including first-turn admission. Token counts are the deterministic software witness; wall
// time is diagnostic only and is not hardware-performance evidence.
func TestInKernelContinuationCacheTwoTurnWorkReceipt(t *testing.T) {
	cfg := tinyCfg()
	prompt := synthIDs(cfg.VocabSize, 48, 12720)
	run := func(reuse bool) (matched, prefilled int, elapsed time.Duration) {
		p := reusePlanner(reuse, false, cfg)
		var first []int
		started := time.Now()
		_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(
			context.Background(), prompt, 8, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				first = append(first, id)
				return false
			})
		if err != nil || len(first) != 8 {
			t.Fatalf("first turn reuse=%v generated=%d err=%v", reuse, len(first), err)
		}
		turn2 := append(append([]int(nil), prompt...), first[:len(first)-1]...)
		_, promptTok, _, matched, _, _, _, _, err := p.generateReusedContextWithBias(
			context.Background(), turn2, 1, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
		if err != nil {
			t.Fatalf("second turn reuse=%v: %v", reuse, err)
		}
		return matched, len(prompt) + promptTok - matched, time.Since(started)
	}
	warmMatched, warmPrefilled, warmElapsed := run(true)
	coldMatched, coldPrefilled, coldElapsed := run(false)
	const existingPromptReuse = 48
	const generatedContinuationReuse = 7
	wantAvoided := existingPromptReuse + generatedContinuationReuse
	if warmMatched != wantAvoided || coldMatched != 0 {
		t.Fatalf("second-turn matched warm/cold=%d/%d, want %d/0", warmMatched, coldMatched, wantAvoided)
	}
	if got := coldPrefilled - warmPrefilled; got != wantAvoided {
		t.Fatalf("two-turn prefill avoided=%d tokens, want %d", got, wantAvoided)
	}
	t.Logf("SOFTWARE RECEIPT two_turn cache_on: matched=%d prefilled=%d elapsed=%s; cache_off: matched=%d prefilled=%d elapsed=%s; total_avoided_prefill_tokens=%d existing_prompt_reuse_tokens=%d new_generated_continuation_reuse_tokens=%d; timing_diagnostic_only=true",
		warmMatched, warmPrefilled, warmElapsed, coldMatched, coldPrefilled, coldElapsed, coldPrefilled-warmPrefilled, existingPromptReuse, generatedContinuationReuse)
}

func TestInKernelContinuationCacheBoundaries(t *testing.T) {
	cfg := tinyCfg()
	prompt := synthIDs(cfg.VocabSize, 14, 12717)
	reference := reusePlanner(false, false, cfg)
	generated, _ := decode(reference, prompt, 5)
	if len(generated) != 5 {
		t.Fatalf("reference generated %d tokens, want 5", len(generated))
	}

	t.Run("emit stop excludes terminal emitted token", func(t *testing.T) {
		p := reusePlanner(true, false, cfg)
		var emitted []int
		gen, _, _, _, _, _, _, stopped, err := p.generateReusedContextWithBias(
			context.Background(), prompt, 5, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				emitted = append(emitted, id)
				return len(emitted) == 3
			})
		if err != nil || !stopped || gen != 3 {
			t.Fatalf("emit-stop turn gen=%d stopped=%v err=%v", gen, stopped, err)
		}
		key := append(append([]int(nil), prompt...), emitted[:2]...)
		if got := p.cachedPrefixLen(key); got != len(key) {
			t.Fatalf("evaluated emit-stop key cached %d/%d", got, len(key))
		}
		withTerminal := append(append([]int(nil), key...), emitted[2])
		if got := p.cachedPrefixLen(withTerminal); got != len(key) {
			t.Fatalf("emit-stop terminal entered cache: %d want %d", got, len(key))
		}
	})

	t.Run("token stop retains all prior forwarded tokens", func(t *testing.T) {
		const stopAt = 3
		p := reusePlanner(true, false, cfg)
		var emitted []int
		stops := map[int]bool{}
		gen, _, _, _, _, _, _, stopped, err := p.generateReusedContextWithBias(
			context.Background(), prompt, 5, 0, 0, 0, nil, 0, 0,
			stops, func(id int) bool {
				emitted = append(emitted, id)
				if len(emitted) == stopAt {
					stops[generated[stopAt]] = true
				}
				return false
			})
		if err != nil || !stopped || gen != stopAt || !eqInts(emitted, generated[:stopAt]) {
			t.Fatalf("token-stop emitted=%v gen=%d stopped=%v err=%v", emitted, gen, stopped, err)
		}
		key := append(append([]int(nil), prompt...), emitted...)
		if got := p.cachedPrefixLen(key); got != len(key) {
			t.Fatalf("forwarded pre-stop key cached %d/%d", got, len(key))
		}
	})
}

func TestInKernelContinuationCacheHybridHALSnapshotAndCachedLogits(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyHybridCfg()
	be := &countingBackend{Backend: compute.Default(), deviceMemory: true}
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "qwen35-simulated-hybrid", false, be, false)
	p.quant = false
	prompt := synthIDs(cfg.VocabSize, 11, 12721)
	var generated []int
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(
		context.Background(), prompt, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			generated = append(generated, id)
			return false
		})
	if err != nil || len(generated) != 4 {
		t.Fatalf("simulated HAL first turn generated=%d err=%v", len(generated), err)
	}
	key := append(append([]int(nil), prompt...), generated[:3]...)
	be.reset()
	var replay []int
	_, _, _, matched, tier, _, _, _, err := p.generateReusedContextWithBias(
		context.Background(), key, 1, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			replay = append(replay, id)
			return false
		})
	if err != nil || matched != len(key) || tier != radixkv.SnapshotTierDeviceL1 {
		t.Fatalf("simulated HAL continuation matched=%d/%d tier=%q err=%v", matched, len(key), tier, err)
	}
	if len(replay) != 1 || replay[0] != generated[3] {
		t.Fatalf("cached current logits replay=%v, want terminal %d", replay, generated[3])
	}
	if mat, batched := be.ops(); mat != 0 || batched != 0 {
		t.Fatalf("exact HAL continuation maxNew=1 performed unused forward: mat=%d batched=%d", mat, batched)
	}
	t.Log("SIMULATED SOFTWARE WITNESS: backend PrefixSnapshot restored attention plus recurrent state; no physical-device claim")
}

func TestInKernelContinuationCacheDoesNotAdmitFailedTurn(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	prompt := synthIDs(cfg.VocabSize, 16, 12718)
	ctx, cancel := context.WithCancel(context.Background())
	var emitted []int
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(
		ctx, prompt, 5, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			emitted = append(emitted, id)
			if len(emitted) == 2 {
				cancel()
			}
			return false
		})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("turn error=%v, want context.Canceled", err)
	}
	if len(emitted) != 2 {
		t.Fatalf("emitted=%d, want cancellation after 2", len(emitted))
	}
	key := append(append([]int(nil), prompt...), emitted[0])
	if got := p.cachedPrefixLen(key); got != len(prompt) {
		t.Fatalf("failed turn admitted continuation: matched=%d want prompt-only %d", got, len(prompt))
	}
}

func TestInKernelContinuationCacheDoesNotAdmitCoalescedFailureAfterAdvance(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	prompt := synthIDs(cfg.VocabSize, 15, 12722)
	wantErr := errors.New("injected coalesced decode failure")
	req := &inKernelCoalesceRequest{
		prepared: make(chan *decodeLane, 1),
		proceed:  make(chan error, 1),
	}
	ctx := context.WithValue(context.Background(), inKernelCoalesceContextKey{}, req)
	go func() {
		lane := <-req.prepared
		next, advance := lane.decodeOne(context.Background())
		if !advance {
			req.proceed <- errors.New("test lane did not advance")
			return
		}
		lane.logits = lane.s.Step(next)
		lane.forwarded = append(lane.forwarded, next)
		req.proceed <- wantErr
	}()
	var emitted []int
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(
		ctx, prompt, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			emitted = append(emitted, id)
			return false
		})
	if !errors.Is(err, wantErr) {
		t.Fatalf("coalesced turn error=%v, want %v", err, wantErr)
	}
	if len(emitted) != 1 {
		t.Fatalf("coalesced injected failure emitted=%d, want one advanced token", len(emitted))
	}
	key := append(append([]int(nil), prompt...), emitted[0])
	if got := p.cachedPrefixLen(key); got != len(prompt) {
		t.Fatalf("failed coalesced turn admitted continuation: matched=%d want prompt-only %d", got, len(prompt))
	}
}

func TestInKernelContinuationCacheRespectsTenantScope(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
	prompt := synthIDs(cfg.VocabSize, 12, 12719)
	ctxA := WithPrefixCacheIdentity(context.Background(), "tenant-a", "")
	ctxB := WithPrefixCacheIdentity(context.Background(), "tenant-b", "")
	var generated []int
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(ctxA, prompt, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
		generated = append(generated, id)
		return false
	})
	if err != nil || len(generated) != 4 {
		t.Fatalf("tenant A generation len=%d err=%v", len(generated), err)
	}
	key := append(append([]int(nil), prompt...), generated[:3]...)
	_, _, _, matchedA, _, _, _, _, errA := p.generateReusedContextWithBias(ctxA, key, 0, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	_, _, _, matchedB, _, _, _, _, errB := p.generateReusedContextWithBias(ctxB, key, 0, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if errA != nil || errB != nil {
		t.Fatalf("tenant probes failed: A=%v B=%v", errA, errB)
	}
	if matchedA != len(key) || matchedB != 0 {
		t.Fatalf("tenant matches A/B=%d/%d, want %d/0", matchedA, matchedB, len(key))
	}
}

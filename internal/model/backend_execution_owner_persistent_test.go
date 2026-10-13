package model

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Preserve the optional KV geometry needed by the CPU attention oracle while
// retaining the owner assertions on allocation, clone, eviction and retirement.
// The V41 owner fixture does not need this dense-attention capability.
type persistentExecutionOwnerBackend struct{ *executionOwnerTestBackend }

type persistentExecutionOwnerKV struct {
	*executionOwnerTestKV
	cfg compute.KVConfig
}

func (b *persistentExecutionOwnerBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	return &persistentExecutionOwnerKV{
		executionOwnerTestKV: b.executionOwnerTestBackend.NewKV(cfg).(*executionOwnerTestKV),
		cfg:                  cfg,
	}
}

func (k *persistentExecutionOwnerKV) KVConfig() compute.KVConfig { return k.cfg }
func (k *persistentExecutionOwnerKV) Clone() compute.KVStore {
	return &persistentExecutionOwnerKV{
		executionOwnerTestKV: k.executionOwnerTestKV.Clone().(*executionOwnerTestKV),
		cfg:                  k.cfg,
	}
}

// fak-test:runtime medium est=1s lane=default
func TestBackendExecutionOwnerPersistentPromptCache(t *testing.T) {
	// The retained tiny Qwen MTP model has vocabulary [0,3). Disable drafting
	// deliberately: this witness isolates the public persistent wrapper's HAL
	// replay, reset/reuse and acquiring prompt-cache publication boundaries.
	// It does not claim a resident MTP draft or hybrid GDN execution witness.
	m := qwen35MTPEnabledSyntheticModel(t)
	var owner sync.Mutex
	b := &persistentExecutionOwnerBackend{newExecutionOwnerTestBackend(&owner)}
	other := &persistentExecutionOwnerBackend{newExecutionOwnerTestBackend(&owner)}
	calls := map[string]int{}
	b.hook = func(site string) { calls[site]++ }
	// Run the entire tiny lifecycle behind one completion channel. A recursive
	// mutex regression fails at the bounded wait rather than blocking test cleanup
	// on the same leaked lock; every resource is CPU-reference fixture storage.
	done := executionOwnerAsync(func() {
		target, err := m.NewBackendSessionChecked(b)
		if err != nil {
			panic(err)
		}
		defer target.Close()
		cache := NewMTPPromptCache()
		defer cache.Clear()
		cfg := DefaultQwen38MTPPersistentConfig()
		cfg.DisableMTP, cfg.PromptCache = true, cache
		sess, err := NewQwen38MTPPersistentSession(target, cfg)
		if err != nil {
			panic(err)
		}
		defer sess.Close()
		if sess.Eligibility().Eligible || sess.drafter != nil {
			panic("fixture unexpectedly admitted resident drafting")
		}
		available := func(stage string) {
			if !owner.TryLock() {
				panic(stage + ": execution owner leaked")
			}
			owner.Unlock()
			if target.backendExecutionScope != nil {
				panic(stage + ": expired scope remains bound")
			}
		}
		prefill := func(stage string, prompt []int) {
			got, err := sess.Prefill(prompt)
			if err != nil {
				panic(fmt.Errorf("%s: %w", stage, err))
			}
			available(stage)
			if target.halKV.Len() != len(prompt) || !reflect.DeepEqual(sess.CommittedTokens(), prompt) {
				panic(fmt.Sprintf("%s: device len=%d committed=%v want=%v", stage, target.halKV.Len(), sess.CommittedTokens(), prompt))
			}
			entry := cache.Get(prompt)
			if entry == nil || entry.Snapshot == nil || entry.Snapshot.Tokens != len(prompt) {
				panic(stage + ": complete prompt snapshot was not published")
			}
			fresh, err := m.NewBackendSessionChecked(other)
			if err != nil {
				panic(err)
			}
			want := fresh.Prefill(prompt)
			fresh.Close()
			if len(got) != m.Cfg.VocabSize || len(want) != m.Cfg.VocabSize {
				panic(stage + ": missing vocabulary logits")
			}
			if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(sess.lastLogits, want) {
				panic(stage + ": logits differ from fresh HAL replay")
			}
		}
		resets, clones := calls["graph-reset"], calls["clone-kv"]
		prefill("cold", []int{0, 1})
		if calls["graph-reset"] != resets+1 || calls["clone-kv"] < clones+2 {
			panic("cold: missing owned replay or snapshot capture/Put clone")
		}
		resets = calls["graph-reset"]
		prefill("replacement", []int{2, 1, 0})
		if calls["graph-reset"] != resets+1 {
			panic("replacement: did not replay from empty HAL state")
		}
		beforeResetLen := target.halKV.Len()
		if err := sess.Reset(); err != nil {
			panic(err)
		}
		available("reset")
		if target.halKV.Len() != beforeResetLen {
			panic("reset: existing HAL KV contract changed")
		}
		if len(sess.CommittedTokens()) != 0 || sess.TurnCount() != 0 || len(sess.lastLogits) != 0 {
			panic("reset: persistent conversational state survived")
		}
		// Reset itself only clears host conversational state in the existing API.
		// The next Prefill must restore the device prefix and extend it, without
		// replaying or attempting to borrow a stale execution scope.
		resets, clones = calls["graph-reset"], calls["clone-kv"]
		prefill("reset-cache-hit", []int{0, 1, 2})
		if calls["graph-reset"] != resets || calls["clone-kv"] < clones+3 {
			panic("reset-cache-hit: expected snapshot clone/restore and cache publication")
		}
		resets, frees := calls["graph-reset"], calls["free-kv"]
		prefill("replace-existing-cache-entry", []int{2, 1, 0})
		if calls["graph-reset"] != resets+1 || calls["free-kv"] < frees+3 {
			panic("cache replacement: missing old device KV/snapshot retirement")
		}
		if err := sess.Reset(); err != nil {
			panic(err)
		}
		cache.Clear()
		available("cache-clear")
		if cache.Len() != 0 {
			panic("cache clear retained entries")
		}
		prefill("reset-cold-replay", []int{1, 2})
		if err := sess.Close(); err != nil {
			panic(err)
		}
		cache.Clear()
		target.Close()
		available("close")
	})
	if r := executionOwnerWait(t, done); r != nil {
		t.Fatalf("persistent owner lifecycle: %v", r)
	}
	if calls["new-kv"] == 0 || calls["free-kv"] == 0 || calls["clone-kv"] == 0 {
		t.Fatalf("persistent test bypassed backend lifecycle: %v", calls)
	}
}

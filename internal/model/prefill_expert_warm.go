package model

import (
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"weak"
)

// PrefillExpertWarmEnv opts an expert-ring/offload session into the packed-prefill
// next-layer expert warm (the Strata layer-major borrow). Unset or any value other
// than 1/true/on leaves the hook disarmed and the prefill walk byte-for-byte unchanged.
const PrefillExpertWarmEnv = "FAK_PREFILL_EXPERT_WARM"

func prefillExpertWarmEnvOn() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(PrefillExpertWarmEnv))) {
	case "1", "true", "on", "yes":
		return true
	}
	return false
}

// prefillWarmRecorders counts next-layer warmer arms over the process life (never
// decremented), so the MoE router seam pays one atomic load, not a map lookup, in a
// process that never armed one.
var prefillWarmRecorders atomic.Int64

// prefillNextLayerWarmer is the production hook body: at layer L's entry it stages
// layer L+1's most recently realized activated set into the session's expert ring.
// The prediction is the session's own last routing for L+1 (the previous chunk,
// prompt or decode step); a cold session has none and the fire is a no-op. Staging
// goes through prefetchActivatedExperts, so it inherits the hint rules: budget
// prefix only, no heat, no permanent-residency fallback, inert without a ring.
//
// It holds the session weakly: the arm lives in a package map keyed by a weak
// pointer, and a strong reference here would keep the session alive forever.
type prefillNextLayerWarmer struct {
	sess  weak.Pointer[Session]
	fires atomic.Int64

	mu    sync.Mutex
	picks map[int][]routePick // layer -> last realized picks

	// observe, when set, sees each fire's layer and the picks it staged; a test seam.
	observe func(layer int, next []routePick)
}

func (w *prefillNextLayerWarmer) fire(layer int) {
	w.fires.Add(1)
	next := w.lastPicks(layer + 1)
	if w.observe != nil {
		w.observe(layer, next)
	}
	s := w.sess.Value()
	if s == nil || len(next) == 0 || s.M == nil || layer+1 >= s.M.Cfg.NumLayers {
		return
	}
	s.prefetchActivatedExperts(layer+1, next)
}

func (w *prefillNextLayerWarmer) lastPicks(layer int) []routePick {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.picks[layer]
}

func (w *prefillNextLayerWarmer) record(layer int, picks []routePick) {
	cp := append([]routePick(nil), picks...)
	w.mu.Lock()
	if w.picks == nil {
		w.picks = map[int][]routePick{}
	}
	w.picks[layer] = cp
	w.mu.Unlock()
}

// armPrefillExpertWarmFromEnv arms the next-layer warm hook on a session that has an
// expert ring, when PrefillExpertWarmEnv is on. It is called from the ring/offload
// session setup; with the env off, or no ring budget, it leaves any existing arm
// untouched and does nothing.
func (s *Session) armPrefillExpertWarmFromEnv() bool {
	if s == nil || s.ExpertRingBytes <= 0 || !prefillExpertWarmEnvOn() {
		return false
	}
	if w := s.prefillWarmer(); w != nil {
		return true
	}
	w := &prefillNextLayerWarmer{sess: weak.Make(s)}
	s.armPrefillNextLayerWarmer(w)
	return true
}

func (s *Session) armPrefillNextLayerWarmer(w *prefillNextLayerWarmer) {
	s.SetPrefillExpertWarm(true, w.fire)
	if v, ok := prefillExpertWarmArms.Load(weak.Make(s)); ok {
		if a, _ := v.(*prefillExpertWarmArm); a != nil {
			a.warmer = w
			prefillWarmRecorders.Add(1)
		}
	}
}

// prefillWarmer returns the session's armed next-layer warmer, or nil.
func (s *Session) prefillWarmer() *prefillNextLayerWarmer {
	if s == nil {
		return nil
	}
	v, ok := prefillExpertWarmArms.Load(weak.Make(s))
	if !ok {
		return nil
	}
	a, _ := v.(*prefillExpertWarmArm)
	if a == nil || !a.enabled {
		return nil
	}
	return a.warmer
}

// recordPrefillWarmPicks remembers layer's realized picks for a session armed with a
// next-layer warmer. Unarmed processes pay one atomic load.
func (s *Session) recordPrefillWarmPicks(layer int, picks []routePick) {
	if s == nil || len(picks) == 0 || prefillWarmRecorders.Load() == 0 {
		return
	}
	if w := s.prefillWarmer(); w != nil {
		w.record(layer, picks)
	}
}

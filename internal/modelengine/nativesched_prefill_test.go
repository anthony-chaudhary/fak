package modelengine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeSchedulerInterleavesBoundedQwenPrefill(t *testing.T) {
	const budget = nativeQwenPrefillMinChunkTokens
	shortPrompt := nativeSchedulerQwenPrompt(budget)
	longPrompt := nativeSchedulerQwenPrompt(2*budget + 3)

	t.Run("ceiling_interleaving_and_final_logits", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		prepare := nativeSchedulerPrefillPrepare(map[string][]int{
			"decode": shortPrompt,
			"long":   longPrompt,
		})
		s := newNativeScheduler(m, prepare)
		if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration: %v", err)
		}
		nativeSchedulerBeginManualDrain(t, s)

		var events []nativeSchedulerEvent
		var finalLogits []float32
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			events = append(events, event)
			if event.Kind == nativeSchedulerEventTransition && event.Lane.tool == "long" {
				finalLogits = copyF32(event.Lane.logits)
			}
		}

		decodeReq := nativeSchedulerAdmitLane(t, s, "decode")
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("initial decode tokens = %v, want one token", got)
		}
		if decodeReq.state != schedLaneDecode {
			t.Fatalf("initial lane state = %d, want DECODE", decodeReq.state)
		}

		events = nil
		longReq := nativeSchedulerAdmitLane(t, s, "long")
		if longReq.state != schedLanePrefilling || longReq.promptCursor != 0 || longReq.promptLen != 0 ||
			len(longReq.logits) != 0 || longReq.sess.Cache.Len() != 0 || s.laneKVBlocksLocked(longReq) != 0 {
			t.Fatalf("enabled admission = state %d cursor %d accounted %d logits %d cache %d blocks %d, want PREFILLING/0/0/0/0/0",
				longReq.state, longReq.promptCursor, longReq.promptLen, len(longReq.logits),
				longReq.sess.Cache.Len(), s.laneKVBlocksLocked(longReq))
		}

		var longOut []int
		for iteration := 0; iteration < 3; iteration++ {
			nativeSchedulerDriveIteration(t, s)
			if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
				t.Fatalf("iteration %d decode tokens = %v, want one interleaved token", iteration+1, got)
			}
			gotLong := nativeSchedulerDrainAvailable(longReq)
			if iteration < 2 && len(gotLong) != 0 {
				t.Fatalf("iteration %d exposed long-lane output before final logits: %v", iteration+1, gotLong)
			}
			longOut = append(longOut, gotLong...)
		}
		if len(longOut) != 1 {
			t.Fatalf("long-lane output after final prefill = %v, want exactly one first decode token", longOut)
		}

		prefills := make([]nativeSchedulerEvent, 0, 3)
		transitionSeen := false
		prefillIndexes := make([]int, 0, 3)
		for i, event := range events {
			switch {
			case event.Kind == nativeSchedulerEventPrefill && event.Lane == longReq:
				prefills = append(prefills, event)
				prefillIndexes = append(prefillIndexes, i)
			case event.Kind == nativeSchedulerEventTransition && event.Lane == longReq:
				transitionSeen = true
			case event.Kind == nativeSchedulerEventDecode && event.Lane == longReq && !transitionSeen:
				t.Fatal("long lane decoded before its final-logit transition")
			}
		}
		if got, want := len(prefills), 3; got != want {
			t.Fatalf("prefill chunks = %d, want %d; events=%+v", got, want, events)
		}
		for i, want := range []struct {
			start int
			n     int
		}{{0, budget}, {budget, budget}, {2 * budget, 3}} {
			got := prefills[i]
			if got.ChunkStart != want.start || got.ChunkLen != want.n {
				t.Fatalf("prefill chunk %d = start %d len %d, want start %d len %d",
					i, got.ChunkStart, got.ChunkLen, want.start, want.n)
			}
			if got.ChunkLen > budget {
				t.Fatalf("prefill chunk %d length = %d, exceeds per-iteration ceiling %d", i, got.ChunkLen, budget)
			}
			if i > 0 && got.Iteration == prefills[i-1].Iteration {
				t.Fatalf("prefill chunks %d and %d ran in scheduler iteration %d", i-1, i, got.Iteration)
			}
		}
		for i := 0; i+1 < len(prefillIndexes); i++ {
			if !nativeSchedulerHasDecodeBetween(events, prefillIndexes[i], prefillIndexes[i+1], decodeReq) {
				t.Fatalf("no active decode emission between prefill chunks %d and %d; events=%+v", i, i+1, events)
			}
		}

		control := m.NewSession()
		control.Quant = true
		control.Q4K = true
		wantLogits := control.Prefill(longPrompt)
		control.Close()
		nativeSchedulerAssertLogitParity(t, finalLogits, wantLogits)
		if got, want := longOut[0], argmax(wantLogits); got != want {
			t.Fatalf("first chunked continuation token = %d, synchronous prefill = %d", got, want)
		}

		decodeReq.Cancel()
		for !longReq.terminal {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(decodeReq)
			nativeSchedulerDrainAvailable(longReq)
		}
		res, err := longReq.Result()
		if err != nil {
			t.Fatalf("chunked prefill Result: %v", err)
		}
		if got, want := res.Meta["input_tokens"], fmt.Sprintf("%d", len(longPrompt)); got != want {
			t.Fatalf("chunked prefill input_tokens = %q, want %q", got, want)
		}
		_, idle, _ := nativeSchedulerDriveIteration(t, s)
		if !idle {
			t.Fatal("cancelled interleaving fixture retained runnable lanes")
		}
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("cancellation_closes_once", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"cancel": longPrompt}))
		if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration: %v", err)
		}
		nativeSchedulerBeginManualDrain(t, s)

		var closeMu sync.Mutex
		closes := 0
		s.closeSession = func(sess *model.Session) {
			closeMu.Lock()
			closes++
			closeMu.Unlock()
			sess.Close()
		}

		req := nativeSchedulerAdmitLane(t, s, "cancel")
		chunks := 0
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventPrefill && event.Lane == req {
				chunks++
				if chunks == 2 {
					req.Cancel()
				}
			}
		}

		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(req); len(got) != 0 {
			t.Fatalf("first middle-prefill iteration emitted tokens: %v", got)
		}
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(req); len(got) != 0 {
			t.Fatalf("cancelled middle-prefill lane emitted tokens: %v", got)
		}
		if _, err := req.Result(); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled prefill Result err = %v, want context.Canceled", err)
		}
		if !req.Reclaimed() {
			t.Fatal("cancelled prefill lane did not reclaim its session")
		}
		closeMu.Lock()
		gotCloses := closes
		closeMu.Unlock()
		if gotCloses != 1 {
			t.Fatalf("cancelled prefill session closes = %d, want exactly 1", gotCloses)
		}

		_, idle, _ := nativeSchedulerDriveIteration(t, s)
		if !idle {
			t.Fatal("cancelled prefill lane remained runnable after compaction")
		}
		req.Cancel()
		s.Close()
		closeMu.Lock()
		gotCloses = closes
		closeMu.Unlock()
		if gotCloses != 1 {
			t.Fatalf("repeat cancellation/Close changed session closes to %d, want 1", gotCloses)
		}
		s.endBlockedDrain()
	})

	t.Run("concurrent_stats_keep_published_prefill_and_decode_accounting", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"stats": shortPrompt}))
		if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration: %v", err)
		}
		s.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			MaxBlocks:   100,
			BlockTokens: budget,
		})
		nativeSchedulerBeginManualDrain(t, s)
		req := nativeSchedulerAdmitLane(t, s, "stats")

		prefillEntered := make(chan struct{})
		prefillRelease := make(chan struct{})
		decodeEntered := make(chan struct{})
		decodeRelease := make(chan struct{})
		s.beforeModelExecute = func(kind nativeSchedulerEventKind, lane *schedLane) {
			if lane != req {
				return
			}
			switch kind {
			case nativeSchedulerEventPrefill:
				close(prefillEntered)
				<-prefillRelease
			case nativeSchedulerEventDecode:
				close(decodeEntered)
				<-decodeRelease
			}
		}

		driveDone := make(chan struct{})
		go func() {
			s.executor.Lock()
			s.runIteration(true)
			s.executor.Unlock()
			close(driveDone)
		}()
		wait := func(ch <-chan struct{}, what string) {
			t.Helper()
			select {
			case <-ch:
			case <-time.After(5 * time.Second):
				t.Fatalf("timed out waiting for %s", what)
			}
		}
		statsWhileBlocked := func(what string) NativePreemptionStats {
			t.Helper()
			stats := make(chan NativePreemptionStats, 1)
			go func() { stats <- s.KVPreemptionStats() }()
			select {
			case got := <-stats:
				return got
			case <-time.After(5 * time.Second):
				t.Fatalf("KVPreemptionStats blocked during %s model execution", what)
				return NativePreemptionStats{}
			}
		}
		assertShell := func(what string) {
			t.Helper()
			s.mu.Lock()
			defer s.mu.Unlock()
			if req.sess == nil || req.sess != req.statsSess || req.sess.Cache != nil || req.sess.M != m || req.inflightSess == nil {
				t.Fatalf("%s stats exposure: sess=%p shell=%p cache=%p model=%p inflight=%p, want non-nil immutable shell/no cache/model/live real session",
					what, req.sess, req.statsSess, req.sess.Cache, req.sess.M, req.inflightSess)
			}
		}

		wait(prefillEntered, "prefill model-execution seam")
		assertShell("prefill")
		if stats := statsWhileBlocked("prefill"); stats.Running != 1 || stats.UsedBlocks != 0 {
			t.Fatalf("stats during first prefill chunk = %+v, want Running=1 UsedBlocks=0 from last publication", stats)
		}
		close(prefillRelease)

		wait(decodeEntered, "decode model-execution seam")
		assertShell("decode")
		if stats := statsWhileBlocked("decode"); stats.Running != 1 || stats.UsedBlocks != 2 {
			t.Fatalf("stats during decode cache append = %+v, want Running=1 UsedBlocks=2 from published prompt+emission", stats)
		}
		close(decodeRelease)
		wait(driveDone, "blocked scheduler iteration")
		s.beforeModelExecute = nil

		if got := nativeSchedulerDrainAvailable(req); len(got) != 1 {
			t.Fatalf("stats witness decode tokens = %v, want one", got)
		}
		if stats := s.KVPreemptionStats(); stats.Running != 1 || stats.UsedBlocks != 2 {
			t.Fatalf("stats after decode publication = %+v, want Running=1 UsedBlocks=2", stats)
		}
		req.Cancel()
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(req)
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("disabled_preserves_synchronous_admission", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"disabled": longPrompt}))
		nativeSchedulerBeginManualDrain(t, s)

		var events []nativeSchedulerEvent
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			events = append(events, event)
		}
		req := nativeSchedulerAdmitLane(t, s, "disabled")
		if req.state != schedLaneDecode || req.promptCursor != len(longPrompt) || len(req.logits) == 0 {
			t.Fatalf("disabled admission = state %d cursor %d logits %d, want synchronous DECODE/%d/nonempty",
				req.state, req.promptCursor, len(req.logits), len(longPrompt))
		}
		if got := req.sess.Cache.Len(); got != len(longPrompt) {
			t.Fatalf("disabled Admit returned with cache length %d, want synchronous %d", got, len(longPrompt))
		}
		if len(events) != 0 {
			t.Fatalf("disabled Admit emitted scheduler events before decode: %+v", events)
		}

		want := nativeSchedulerSynchronousTokens(m, longPrompt)
		got := make([]int, 0, genTokens)
		for len(got) < genTokens {
			nativeSchedulerDriveIteration(t, s)
			got = append(got, nativeSchedulerDrainAvailable(req)...)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("disabled scheduler tokens = %v, historical synchronous oracle = %v", got, want)
		}
		for _, event := range events {
			if event.Kind == nativeSchedulerEventPrefill || event.Kind == nativeSchedulerEventTransition {
				t.Fatalf("disabled scheduler entered bounded prefill: %+v", event)
			}
		}
		if _, err := req.Result(); err != nil {
			t.Fatalf("disabled scheduler Result: %v", err)
		}
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("subthreshold_budget_fails_closed", func(t *testing.T) {
		s := newNativeScheduler(nativeSchedulerPrefillModel(t), nil)
		if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens - 1); err == nil {
			t.Fatal("subthreshold Qwen prefill budget was accepted")
		}
		if s.qwenPrefillTokens != 0 {
			t.Fatalf("subthreshold budget left feature enabled at %d tokens", s.qwenPrefillTokens)
		}
	})

	t.Run("unsupported_qwen_qualification_falls_back_to_synchronous_admission", func(t *testing.T) {
		q8Only := model.NewSynthetic(nativeSchedulerPrefillConfig())
		q8Only.Quantize()
		q8Scheduler := newNativeScheduler(q8Only, nil)
		q8Scheduler.qwenPrefillTokens = budget
		q8Session := q8Only.NewSession()
		q8Session.Quant = true
		q8Session.Q4K = true
		if got := q8Scheduler.qwenPrefillChunkBudget(schedPrepare{q4k: true}, q8Session, len(longPrompt)); got != 0 {
			t.Fatalf("Q8-only model qualified from q4k intent with ceiling %d", got)
		}
		q8Session.Close()

		resident := nativeSchedulerPrefillModel(t)
		residentScheduler := newNativeScheduler(resident, nil)
		residentScheduler.qwenPrefillTokens = budget
		residentSession := resident.NewSession()
		residentSession.Quant = true
		residentSession.Q4K = true
		if got := residentScheduler.qwenPrefillChunkBudget(schedPrepare{q4k: true}, residentSession, len(longPrompt)); got != budget {
			t.Fatalf("resident-Q4_K model ceiling = %d, want %d", got, budget)
		}
		residentSession.Close()

		cases := []struct {
			name   string
			mutate func(*model.Config)
			env    bool
		}{
			{name: "attention_output_gate", mutate: func(cfg *model.Config) { cfg.AttnOutputGate = false }},
			{name: "moe", mutate: func(cfg *model.Config) { cfg.NumExperts, cfg.NumExpertsPerTok = 2, 1 }},
			{name: "dense_mlp", mutate: func(cfg *model.Config) { cfg.DenseMLP = true }},
			{name: "alibi", mutate: func(cfg *model.Config) { cfg.Alibi = true }},
			{name: "norm_gain", mutate: func(cfg *model.Config) { cfg.NormGain1p = false }},
			{name: "layer_norm", mutate: func(cfg *model.Config) { cfg.LayerNorm = true }},
			{name: "topology", mutate: func(cfg *model.Config) { cfg.BlockTopology = model.SandwichNorm }},
			{name: "layer_rope", mutate: func(cfg *model.Config) {
				cfg.RopeThetaPerLayer = []float64{cfg.RopeTheta, cfg.RopeTheta * 2}
			}},
			{name: "diagnostic_token_loop", env: true},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				cfg := nativeSchedulerPrefillConfig()
				if tc.mutate != nil {
					tc.mutate(&cfg)
				}
				if tc.env {
					t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "1")
				}
				if nativeQwenResidentAppendEligible(cfg, len(longPrompt)) {
					t.Fatal("unsupported Qwen configuration passed resident append qualification")
				}
			})
		}

		cfg := nativeSchedulerPrefillConfig()
		cfg.AttnOutputGate = false
		m := nativeSchedulerPrefillModel(t, cfg)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"unsupported": longPrompt}))
		if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration: %v", err)
		}
		nativeSchedulerBeginManualDrain(t, s)
		req := nativeSchedulerAdmitLane(t, s, "unsupported")
		if req.state != schedLaneDecode || req.promptCursor != len(longPrompt) || len(req.logits) == 0 {
			t.Fatalf("unsupported Qwen admission = state %d cursor %d logits %d, want synchronous DECODE/%d/nonempty",
				req.state, req.promptCursor, len(req.logits), len(longPrompt))
		}
		req.Cancel()
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(req)
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("prefill_protects_partial_lane_without_suspending_preemption", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{
			"decode-a": shortPrompt,
			"decode-b": shortPrompt,
			"prefill":  longPrompt,
		}))
		if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration: %v", err)
		}
		nativeSchedulerBeginManualDrain(t, s)

		decodeA := nativeSchedulerAdmitLane(t, s, "decode-a")
		decodeB := nativeSchedulerAdmitLane(t, s, "decode-b")
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(decodeA)
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(decodeA)
		nativeSchedulerDrainAvailable(decodeB)
		if decodeA.state != schedLaneDecode || decodeB.state != schedLaneDecode {
			t.Fatalf("decode fixtures not ready: states %d/%d", decodeA.state, decodeB.state)
		}

		s.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			VictimRule:  NativePreemptVictimMostRecent,
			MaxBlocks:   5,
			BlockTokens: budget,
		})
		var events []nativeSchedulerEvent
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			events = append(events, event)
		}
		prefill := nativeSchedulerAdmitLane(t, s, "prefill")
		nativeSchedulerDriveIteration(t, s)

		if prefill.state != schedLanePrefilling || prefill.promptCursor != budget ||
			prefill.promptLen != budget || prefill.sess == nil || prefill.sess.Cache.Len() != budget {
			t.Fatalf("protected prefill lane = state %d cursor %d accounted %d cache %d session %p, want PREFILLING/%d/%d/%d/live",
				prefill.state, prefill.promptCursor, prefill.promptLen, prefill.sess.Cache.Len(), prefill.sess,
				budget, budget, budget)
		}
		if len(s.preempted) != 0 {
			t.Fatalf("future prompt reservation preempted lanes early: %v", preemptedTools(s.preempted))
		}
		if stats := s.KVPreemptionStats(); stats.UsedBlocks != 5 || stats.Preemptions != 0 {
			t.Fatalf("first live-block accounting stats = %+v, want 5 resident blocks and no preemption", stats)
		}
		if blocks := s.laneKVBlocksLocked(prefill); blocks != 1 {
			t.Fatalf("first resident prefill chunk blocks = %d, want 1", blocks)
		}
		nativeSchedulerDrainAvailable(decodeA)
		nativeSchedulerDrainAvailable(decodeB)

		nativeSchedulerDriveIteration(t, s)
		if len(s.preempted) != 0 {
			t.Fatalf("second chunk was preempted before its live blocks crossed the boundary: %v", preemptedTools(s.preempted))
		}
		if stats := s.KVPreemptionStats(); stats.UsedBlocks != 6 || stats.Preemptions != 0 {
			t.Fatalf("second live-block accounting stats = %+v, want 6 resident blocks pending boundary enforcement", stats)
		}
		if prefill.promptLen != 2*budget || prefill.sess.Cache.Len() != 2*budget {
			t.Fatalf("second live chunk accounted prompt/cache = %d/%d, want %d/%d",
				prefill.promptLen, prefill.sess.Cache.Len(), 2*budget, 2*budget)
		}
		if blocks := s.laneKVBlocksLocked(prefill); blocks != 2 {
			t.Fatalf("second resident prefill chunk blocks = %d, want 2", blocks)
		}
		nativeSchedulerDrainAvailable(decodeA)
		nativeSchedulerDrainAvailable(decodeB)

		nativeSchedulerDriveIteration(t, s)
		if len(s.preempted) != 1 || s.preempted[0] != decodeB {
			t.Fatalf("preempted lanes = %v, want decode-b only", preemptedTools(s.preempted))
		}
		if stats := s.KVPreemptionStats(); stats.Preemptions != 1 || stats.RecomputeCount != 1 {
			t.Fatalf("combined prefill/preemption stats = %+v, want one recompute victim", stats)
		}
		if prefill.state != schedLaneDecode || prefill.promptCursor != len(longPrompt) ||
			prefill.promptLen != len(longPrompt) || prefill.sess.Cache.Len() != len(longPrompt)+1 {
			t.Fatalf("final prefill accounting = state %d cursor/accounted/cache %d/%d/%d, want DECODE/%d/%d/%d",
				prefill.state, prefill.promptCursor, prefill.promptLen, prefill.sess.Cache.Len(),
				len(longPrompt), len(longPrompt), len(longPrompt)+1)
		}
		if !nativeSchedulerEventSeen(events, nativeSchedulerEventPrefill, prefill) ||
			!nativeSchedulerEventSeen(events, nativeSchedulerEventDecode, decodeA) {
			t.Fatalf("combined prefill/preemption iteration events = %+v, want protected prefill plus surviving decode", events)
		}

		decodeA.Cancel()
		decodeB.Cancel()
		prefill.Cancel()
		for range 3 {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(decodeA)
			nativeSchedulerDrainAvailable(decodeB)
			nativeSchedulerDrainAvailable(prefill)
		}
		nativeSchedulerEndManualDrain(s)
	})
}

func nativeSchedulerPrefillModel(t *testing.T, configs ...model.Config) *model.Model {
	t.Helper()
	cfg := nativeSchedulerPrefillConfig()
	if len(configs) > 0 {
		cfg = configs[0]
	}
	b := model.NewQuantBuilder(cfg, true)
	add := func(name string, shape []int, norm bool) {
		t.Helper()
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		data := make([]float32, n)
		for i := range data {
			if norm {
				data[i] = 1
			} else {
				data[i] = float32((i%29)-14) * 0.001
			}
		}
		if err := b.AddF32Tensor(name, shape, data); err != nil {
			t.Fatalf("AddF32Tensor(%q): %v", name, err)
		}
	}

	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	add("model.embed_tokens.weight", []int{V, H}, false)
	for layer := 0; layer < cfg.NumLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		add(prefix+"input_layernorm.weight", []int{H}, true)
		if layer < len(cfg.LayerTypes) && cfg.LayerTypes[layer] == "linear_attention" {
			nK, nV := cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads
			keyDim := nK * cfg.LinearKeyHeadDim
			valDim := nV * cfg.LinearValueHeadDim
			convDim := 2*keyDim + valDim
			add(prefix+"linear_attn.in_proj_qkv.weight", []int{convDim, H}, false)
			add(prefix+"linear_attn.in_proj_z.weight", []int{valDim, H}, false)
			add(prefix+"linear_attn.in_proj_b.weight", []int{nV, H}, false)
			add(prefix+"linear_attn.in_proj_a.weight", []int{nV, H}, false)
			add(prefix+"linear_attn.conv1d.weight", []int{convDim * cfg.LinearConvKernelDim}, false)
			add(prefix+"linear_attn.A_log", []int{nV}, false)
			add(prefix+"linear_attn.dt_bias", []int{nV}, false)
			add(prefix+"linear_attn.norm.weight", []int{cfg.LinearValueHeadDim}, true)
			add(prefix+"linear_attn.out_proj.weight", []int{H, valDim}, false)
		} else {
			qRows := nH * hd
			if cfg.AttnOutputGate {
				qRows *= 2
			}
			add(prefix+"self_attn.q_proj.weight", []int{qRows, H}, false)
			add(prefix+"self_attn.k_proj.weight", []int{nKV * hd, H}, false)
			add(prefix+"self_attn.v_proj.weight", []int{nKV * hd, H}, false)
			add(prefix+"self_attn.o_proj.weight", []int{H, nH * hd}, false)
		}
		add(prefix+"post_attention_layernorm.weight", []int{H}, true)
		add(prefix+"mlp.gate_proj.weight", []int{I, H}, false)
		add(prefix+"mlp.up_proj.weight", []int{I, H}, false)
		add(prefix+"mlp.down_proj.weight", []int{H, I}, false)
	}
	add("model.norm.weight", []int{H}, true)

	// The public loader-neutral builder is the production resident-Q4_K construction
	// seam; Model has no QuantizeQ4K conversion API. Seed the exact Qwen projection
	// subset the real q4_k_m loader keeps resident. Zero-scale blocks are valid Q4_K.
	const (
		q4KBlockWeights = 256
		q4KBlockBytes   = 144
	)
	addQ4K := func(name string, out, in int) {
		t.Helper()
		raw := make([]byte, out*(in/q4KBlockWeights)*q4KBlockBytes)
		if err := b.AddResidentQ4K(name, []int{out, in}, raw); err != nil {
			t.Fatalf("AddResidentQ4K(%q): %v", name, err)
		}
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		if layer >= len(cfg.LayerTypes) || cfg.LayerTypes[layer] != "linear_attention" {
			addQ4K(prefix+"self_attn.v_proj.weight", nKV*hd, H)
			addQ4K(prefix+"self_attn.o_proj.weight", H, nH*hd)
		}
		addQ4K(prefix+"mlp.gate_proj.weight", I, H)
		addQ4K(prefix+"mlp.up_proj.weight", I, H)
		addQ4K(prefix+"mlp.down_proj.weight", H, I)
	}
	m, err := b.Build()
	if err != nil {
		t.Fatalf("Build resident-Q4_K prefill model: %v", err)
	}
	if m.Q4KCount() == 0 {
		t.Fatal("resident-Q4_K prefill fixture contains no Q4_K weights")
	}
	gate := "model.layers.0.mlp.gate_proj.weight"
	if out, in := m.Q4KShape(gate); out != I || in != H {
		t.Fatalf("resident Q4_K %s shape = %dx%d, want %dx%d", gate, out, in, I, H)
	}
	return m
}

func nativeSchedulerPrefillConfig() model.Config {
	cfg := SyntheticConfig()
	cfg.HiddenSize = 256
	cfg.HeadDim = 64
	cfg.IntermediateSize = 256
	cfg.NumLayers = 2
	cfg.LayerTypes = []string{"linear_attention", "full_attention"}
	cfg.FullAttentionInterval = 2
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = cfg.HeadDim
	cfg.LinearValueHeadDim = cfg.HeadDim
	cfg.LinearNumKeyHeads = cfg.NumKVHeads
	cfg.LinearNumValueHeads = cfg.NumHeads
	cfg.AttnOutputGate = true
	cfg.NormGain1p = true
	return cfg
}

func nativeSchedulerPrefillPrepare(prompts map[string][]int) schedPrepareFunc {
	return func(_ context.Context, call *abi.ToolCall, _ *model.Model) schedPrepare {
		return schedPrepare{
			prompt: append([]int(nil), prompts[call.Tool]...),
			q4k:    true,
		}
	}
}

func nativeSchedulerQwenPrompt(n int) []int {
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = 3 + (i*17)%239
	}
	return prompt
}

func nativeSchedulerBeginManualDrain(t *testing.T, s *NativeScheduler) {
	t.Helper()
	if !s.beginBlockedDrain() {
		t.Fatal("failed to reserve manual scheduler drain")
	}
}

func nativeSchedulerEndManualDrain(s *NativeScheduler) {
	s.Close()
	s.endBlockedDrain()
}

func nativeSchedulerAdmitLane(t *testing.T, s *NativeScheduler, tool string) *schedLane {
	t.Helper()
	req, err := s.Admit(context.Background(), inlineCall(tool, `{}`))
	if err != nil {
		t.Fatalf("Admit(%q): %v", tool, err)
	}
	ln, ok := req.(*schedLane)
	if !ok {
		t.Fatalf("Admit(%q) returned %T, want *schedLane", tool, req)
	}
	return ln
}

func nativeSchedulerDriveIteration(t *testing.T, s *NativeScheduler) (didWork, idle, closed bool) {
	t.Helper()
	s.executor.Lock()
	didWork, idle, closed = s.runIteration(true)
	s.executor.Unlock()
	return didWork, idle, closed
}

func nativeSchedulerDrainAvailable(ln *schedLane) []int {
	var out []int
	for {
		select {
		case token, ok := <-ln.Tokens():
			if !ok {
				return out
			}
			out = append(out, token.ID)
		default:
			return out
		}
	}
}

func nativeSchedulerHasDecodeBetween(events []nativeSchedulerEvent, left, right int, lane *schedLane) bool {
	for _, event := range events[left+1 : right] {
		if event.Kind == nativeSchedulerEventDecode && event.Lane == lane {
			return true
		}
	}
	return false
}

func nativeSchedulerEventSeen(events []nativeSchedulerEvent, kind nativeSchedulerEventKind, lane *schedLane) bool {
	for _, event := range events {
		if event.Kind == kind && event.Lane == lane {
			return true
		}
	}
	return false
}

func nativeSchedulerSynchronousTokens(m *model.Model, prompt []int) []int {
	sess := m.NewSession()
	sess.Quant = true
	sess.Q4K = true
	logits := sess.Prefill(prompt)
	out := make([]int, 0, genTokens)
	for range genTokens {
		next := argmax(logits)
		out = append(out, next)
		if len(out) < genTokens {
			logits = sess.Step(next)
		}
	}
	sess.Close()
	return out
}

func nativeSchedulerAssertLogitParity(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("final logits length = %d, want %d nonzero", len(got), len(want))
	}
	const (
		absTolerance = 1e-5
		relTolerance = 1e-5
	)
	for i := range got {
		g := float64(got[i])
		w := float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("final logits[%d] non-finite: chunked=%v synchronous=%v", i, got[i], want[i])
		}
		diff := math.Abs(g - w)
		scale := math.Max(math.Abs(g), math.Abs(w))
		if diff > absTolerance && diff > relTolerance*scale {
			t.Fatalf("final logits[%d] differ: chunked=%g synchronous=%g abs=%g rel=%g, tolerances abs=%g rel=%g",
				i, got[i], want[i], diff, diff/math.Max(scale, math.SmallestNonzeroFloat64), absTolerance, relTolerance)
		}
	}
	if gotArgmax, wantArgmax := argmax(got), argmax(want); gotArgmax != wantArgmax {
		t.Fatalf("chunked final-logit argmax = %d, synchronous = %d", gotArgmax, wantArgmax)
	}
}

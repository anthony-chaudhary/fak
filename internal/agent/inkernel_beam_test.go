package agent

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/beam"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func nativeBeamTestPlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	cfg := tinyConcurrencyConfig()
	cfg.VocabSize = 259 // Exact byte vocabulary and three ChatML special IDs.
	m := model.NewSynthetic(cfg)
	p := NewInKernelPlannerWithConfig(m, loadProbeTok(t), "synthetic-beam", false, nil, false, InKernelPlannerConfig{BatchDecode: true, QwenQ4KPrefillChunkTokens: 512})
	p.quant = false
	p.temp = 0
	p.maxNew = 3
	p.SetHostMemoryBudget(1<<30, func() (int64, bool) { return 0, true })
	return p
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeBeamOrdinaryCompleteOwnsReplayAndRanks(t *testing.T) {
	p := nativeBeamTestPlanner(t)
	messages := []Message{{Role: RoleUser, Content: "hello"}}
	opts := []SampleOpt{WithMaxTokens(3), WithNativeBeamSearch(&NativeBeamSearchParams{Width: 2})}
	sp := applySampleOpts(opts...)
	prepared, err := p.preparePrompt(context.Background(), messages, nil, sp, opts...)
	if err != nil {
		t.Fatal(err)
	}
	cfg := beam.SearchConfig{Width: 2, MaxTokens: 3, VocabSize: p.m.Cfg.VocabSize, LengthPenalty: 1}
	for id, stop := range StopIDs(p.tok, p.m.Cfg) {
		if stop {
			cfg.EOS = append(cfg.EOS, id)
		}
	}
	// Independent ordinary Session adapter: no cache/indirection or planner search
	// callback, and every replay is closed before its logits are returned.
	want, err := beam.Search(context.Background(), prepared.ids, cfg, func(ctx context.Context, ids []int) (logits []float32, err error) {
		session := p.m.NewSession()
		defer finishInKernelCleanup(&err, session.Close)
		session.Quant = false
		logits, err = session.PrefillContext(ctx, ids)
		return append([]float32(nil), logits...), err
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.NativeBeamSearch == nil {
		t.Fatal("ordinary Complete did not select beam search")
	}
	r := got.NativeBeamSearch
	if !p.BatchDecodeEnabled() || !p.qwenQ4KPrefillChunkExplicit || p.coalescesQwenDecode() || p.qwenQ4KPrefillChunkTarget() {
		t.Fatal("normal serve structural batch gate was changed or an actual cohort selected")
	}
	if r.Schema != "fak.native-beam-search/1" || r.SessionOwnership != "serial-fresh-owned-cpu-replay" ||
		r.ForwardCalls != want.ForwardCalls || r.ReplayedTokens != want.ReplayedTokens || r.ReplayTokenBound != want.ReplayTokenBound ||
		got.Usage.PromptTokens != len(prepared.ids) || len(r.Sequences) != len(want.Sequences) {
		t.Fatalf("receipt does not describe real replay: %+v", r)
	}
	for i, row := range r.Sequences {
		w := want.Sequences[i]
		if !reflect.DeepEqual(row.TokenIDs, w.TokenIDs) || row.GeneratedTokens != w.GeneratedTokens ||
			row.CumulativeLogprob != w.CumulativeLogprob || row.Score != w.Score || row.FinishReason != w.FinishReason {
			t.Fatalf("row %d = %+v want %+v", i, row, w)
		}
	}
	if !reflect.DeepEqual(got.Message, r.Sequences[0].Message) || got.FinishReason != r.Sequences[0].FinishReason {
		t.Fatal("ordinary completion is not ranked winner")
	}
	if got.Usage.CompletionTokens != len(r.Sequences[0].TokenIDs) {
		t.Fatal("exploration charged as ordinary usage")
	}
	if st := p.HostMemoryBudgetStats(); st.Reserved != 0 {
		t.Fatalf("reservation leaked: %+v", st)
	}
	saved := append([]int(nil), r.Sequences[0].TokenIDs...)
	again, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(again.NativeBeamSearch.Sequences[0].TokenIDs, saved) || !reflect.DeepEqual(r.Sequences[0].TokenIDs, saved) {
		t.Fatal("request/session state or output alias leaked")
	}
	if len(again.NativeBeamSearch.Sequences[0].TokenIDs) > 0 {
		again.NativeBeamSearch.Sequences[0].TokenIDs[0] ^= 1
		if !reflect.DeepEqual(r.Sequences[0].TokenIDs, saved) {
			t.Fatal("results alias across requests")
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeBeamDefaultNilAndCapturedOptions(t *testing.T) {
	penalty := 0.0
	params := &NativeBeamSearchParams{Width: 2, LengthPenalty: &penalty}
	opt := WithNativeBeamSearch(params)
	params.Width = 8
	penalty = 2
	a, b := applySampleOpts(opt), applySampleOpts(opt)
	if a.NativeBeamSearch.Width != 2 || *a.NativeBeamSearch.LengthPenalty != 0 {
		t.Fatal("option did not capture caller values")
	}
	a.NativeBeamSearch.Width = 1
	*a.NativeBeamSearch.LengthPenalty = 1
	if b.NativeBeamSearch.Width != 2 || *b.NativeBeamSearch.LengthPenalty != 0 {
		t.Fatal("option applications alias")
	}
	p := nativeBeamTestPlanner(t)
	messages := []Message{{Role: RoleUser, Content: "hi"}}
	plain, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(2))
	if err != nil {
		t.Fatal(err)
	}
	unchanged, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(2), WithNativeBeamSearch(nil))
	if err != nil {
		t.Fatal(err)
	}
	if plain.NativeBeamSearch != nil || unchanged.NativeBeamSearch != nil ||
		!reflect.DeepEqual(plain.Message, unchanged.Message) || plain.FinishReason != unchanged.FinishReason ||
		plain.Usage.CompletionTokens != unchanged.Usage.CompletionTokens {
		t.Fatal("nil selection changed ordinary decode")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeBeamUnsupportedAndCanceledBeforePublication(t *testing.T) {
	p := nativeBeamTestPlanner(t)
	messages := []Message{{Role: RoleUser, Content: "hi"}}
	selection := WithNativeBeamSearch(&NativeBeamSearchParams{Width: 2})
	for _, extra := range []SampleOpt{
		WithNativeBeamSearch(&NativeBeamSearchParams{Width: 9}),
		WithMaxTokens(129),
		func(sp *SampleParams) { v := math.NaN(); sp.NativeBeamSearch.LengthPenalty = &v },
		func(sp *SampleParams) { v := 0.0; sp.TopP = &v },
		func(sp *SampleParams) { sp.RawPrompt = true },
	} {
		got, err := p.Complete(context.Background(), messages, nil, selection, extra)
		if got != nil || !errors.Is(err, ErrNativeBeamSearchUnsupported) {
			t.Fatalf("unsupported got=%v err=%v", got, err)
		}
	}
	p.metal = true
	if p.NativeBeamSearchSupported() {
		t.Fatal("device route admitted")
	}
	got, err := p.Complete(context.Background(), messages, nil, selection)
	if got != nil || !errors.Is(err, ErrNativeBeamSearchUnsupported) {
		t.Fatal("device selected")
	}
	p.metal = false
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err = p.Complete(ctx, messages, nil, selection)
	if got != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel got=%v err=%v", got, err)
	}
	calls := 0
	got, err = p.CompleteStream(context.Background(), func(string) error { calls++; return nil }, messages, nil, selection)
	if got != nil || !errors.Is(err, ErrNativeBeamSearchUnsupported) || calls != 0 {
		t.Fatal("beam streamed partial output")
	}
	if st := p.HostMemoryBudgetStats(); st.Reserved != 0 {
		t.Fatal("refusal leaked reservation")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeBeamHTTPProviderRefusesAtSingleOptionFold(t *testing.T) {
	p := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: "http://invalid.example", ModelID: "unused"}
	for _, stream := range []bool{false, true} {
		folds := 0
		opts := []SampleOpt{func(*SampleParams) { folds++ }, WithNativeBeamSearch(&NativeBeamSearchParams{Width: 2})}
		call, err := p.prepareUpstreamContext(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, stream, opts...)
		if call != nil || !errors.Is(err, ErrNativeBeamSearchUnsupported) || folds != 1 {
			t.Fatalf("provider preparation call=%v err=%v folds=%d", call, err, folds)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeBeamWorkspaceReservationIsBoundedAndIdempotent(t *testing.T) {
	p := nativeBeamTestPlanner(t)
	p.SetHostMemoryBudget(beam.SearchWorkspaceBudgetBytes-1, func() (int64, bool) { return 0, true })
	release, err := p.admitBeamWorkspace(p.hostBudget)
	if err == nil || release != nil || p.HostMemoryBudgetStats().Reserved != 0 {
		t.Fatal("over-budget workspace admitted")
	}
	p.SetHostMemoryBudget(beam.SearchWorkspaceBudgetBytes, func() (int64, bool) { return 0, true })
	release, err = p.admitBeamWorkspace(p.hostBudget)
	if err != nil {
		t.Fatal(err)
	}
	if p.HostMemoryBudgetStats().Reserved != beam.SearchWorkspaceBudgetBytes {
		t.Fatal("workspace not charged")
	}
	release()
	release()
	if p.HostMemoryBudgetStats().Reserved != 0 {
		t.Fatal("workspace release not idempotent")
	}
	old := p.hostBudget
	p.SetHostMemoryBudget(beam.SearchWorkspaceBudgetBytes, func() (int64, bool) { return 0, true })
	if release, err = p.admitBeamWorkspace(old); err == nil || release != nil {
		t.Fatal("changed host budget mixed across admission")
	}
}

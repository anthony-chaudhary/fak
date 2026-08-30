package model

import (
	"errors"
	"reflect"
	"testing"
)

type fakeQwen35MTPForwardCall struct {
	pos       int
	hidden    []float32
	embedding []float32
}

type fakeQwen35MTPForward struct {
	calls     []fakeQwen35MTPForwardCall
	closes    int
	failAtPos int
	failErr   error
	emptyAt   int
}

func (f *fakeQwen35MTPForward) Forward(pos int, hidden, embedding []float32) ([]float32, error) {
	f.calls = append(f.calls, fakeQwen35MTPForwardCall{
		pos:       pos,
		hidden:    append([]float32(nil), hidden...),
		embedding: append([]float32(nil), embedding...),
	})
	if f.failErr != nil && pos == f.failAtPos {
		return nil, f.failErr
	}
	if pos == f.emptyAt {
		return nil, nil
	}
	logits := make([]float32, 4)
	logits[(pos+1)%len(logits)] = 1
	return logits, nil
}

func (f *fakeQwen35MTPForward) Close() { f.closes++ }

type fakeQwen35MTPFactory struct {
	forwards []*fakeQwen35MTPForward
	make     func() *fakeQwen35MTPForward
}

func (f *fakeQwen35MTPFactory) newForward() (qwen35MTPDraftForward, error) {
	forward := &fakeQwen35MTPForward{failAtPos: -1, emptyAt: -1}
	if f.make != nil {
		forward = f.make()
	}
	f.forwards = append(f.forwards, forward)
	return forward, nil
}

func testQwen35MTPInputs() (Qwen35MTPTargetHidden, Qwen35MTPTokenEmbedding) {
	hidden := func(prefix []int) ([]float32, error) {
		last := prefix[len(prefix)-1]
		return []float32{float32(len(prefix)), float32(last)}, nil
	}
	embedding := func(token int) ([]float32, error) {
		return []float32{float32(token), -float32(token)}, nil
	}
	return hidden, embedding
}

func TestQwen35MTPDrafterProductionPathCallsForward(t *testing.T) {
	m := qwen35MTPTinyForwardModel(t)
	// The older tiny-forward witness uses epsilon zero and zero attention
	// projections, which can yield NaN in Q/K normalization. This adapter witness
	// needs a real, finite argmax from the shared head.
	m.Cfg.RMSNormEps = 1e-5
	control, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct control MTP forward: %v", err)
	}
	controlLogits, err := control.Forward(0, []float32{3, 4}, []float32{0, 5})
	control.Close()
	if err != nil {
		t.Fatalf("execute control MTP forward: %v", err)
	}
	d, err := NewQwen35MTPDrafter(m, 1,
		func(prefix []int) ([]float32, error) { return []float32{3, 4}, nil },
		func(token int) ([]float32, error) { return []float32{0, 5}, nil },
	)
	if err != nil {
		t.Fatalf("construct production MTP drafter: %v", err)
	}
	t.Cleanup(d.Close)

	want := []int{argmaxF32(controlLogits)}
	got := d.Drafter()([]int{9})
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("production MTP draft = %v from logits %v, want %v from control logits %v", got, d.lastLogits, want, controlLogits)
	}
	forward, ok := d.forward.(*Qwen35MTPForward)
	if !ok {
		t.Fatalf("production forward = %T, want *Qwen35MTPForward", d.forward)
	}
	if forward.draft.Cache.Len() != 1 || forward.lastPos != 0 {
		t.Fatalf("production forward state = cache %d lastPos %d, want cache 1 lastPos 0", forward.draft.Cache.Len(), forward.lastPos)
	}
	if err := d.Err(); err != nil {
		t.Fatalf("production MTP runtime error: %v", err)
	}
}

func TestQwen35MTPDrafterPrefixExtensionKeepsForward(t *testing.T) {
	factory := new(fakeQwen35MTPFactory)
	hidden, embedding := testQwen35MTPInputs()
	d, err := newQwen35MTPDrafter(2, hidden, embedding, factory.newForward)
	if err != nil {
		t.Fatalf("construct drafter: %v", err)
	}
	t.Cleanup(d.Close)

	if got := d.Propose([]int{10, 11}); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Fatalf("first draft = %v, want [2 3]", got)
	}
	extended := []int{10, 11, 2, 3, 12}
	if got := d.Propose(extended); !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("extended-prefix draft = %v, want [1 2]", got)
	}
	if len(factory.forwards) != 1 {
		t.Fatalf("forward instances = %d, want 1 for genuine prefix extension", len(factory.forwards))
	}
	calls := factory.forwards[0].calls
	if want := len(extended) + d.k - 1; len(calls) != want {
		t.Fatalf("forward calls = %d, want committed prefix plus draft lookahead = %d", len(calls), want)
	}
	for wantPos, call := range calls {
		if call.pos != wantPos {
			t.Fatalf("forward call %d position = %d, want monotonic position %d", wantPos, call.pos, wantPos)
		}
	}
	currentIndex := len(extended) - 1
	if got := calls[currentIndex].embedding; !reflect.DeepEqual(got, []float32{12, -12}) {
		t.Fatalf("current-token embedding at extension = %v, want callback result for token 12", got)
	}
}

func TestQwen35MTPDrafterRecreatesOnRewindAndDivergence(t *testing.T) {
	for _, tc := range []struct {
		name      string
		committed []int
	}{
		{name: "rewind", committed: []int{7}},
		{name: "divergence", committed: []int{7, 9}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			factory := new(fakeQwen35MTPFactory)
			hidden, embedding := testQwen35MTPInputs()
			d, err := newQwen35MTPDrafter(2, hidden, embedding, factory.newForward)
			if err != nil {
				t.Fatalf("construct drafter: %v", err)
			}
			t.Cleanup(d.Close)

			if got := d.Propose([]int{7, 8}); len(got) != 2 {
				t.Fatalf("first draft length = %d, want 2", len(got))
			}
			if got := d.Propose(tc.committed); len(got) != 2 {
				t.Fatalf("draft after %s length = %d, want 2", tc.name, len(got))
			}
			if len(factory.forwards) != 2 {
				t.Fatalf("forward instances after %s = %d, want 2", tc.name, len(factory.forwards))
			}
			if factory.forwards[0].closes != 1 {
				t.Fatalf("superseded forward closes = %d, want 1", factory.forwards[0].closes)
			}
			if got := factory.forwards[1].calls[0].pos; got != 0 {
				t.Fatalf("recreated forward first position = %d, want 0", got)
			}
		})
	}
}

func TestQwen35MTPDrafterCallbackFailureIsLatched(t *testing.T) {
	callbackErr := errors.New("target hidden unavailable")
	factory := new(fakeQwen35MTPFactory)
	failed := false
	hidden := func(prefix []int) ([]float32, error) {
		if len(prefix) == 2 && !failed {
			failed = true
			return nil, callbackErr
		}
		return []float32{1, 2}, nil
	}
	_, embedding := testQwen35MTPInputs()
	d, err := newQwen35MTPDrafter(2, hidden, embedding, factory.newForward)
	if err != nil {
		t.Fatalf("construct drafter: %v", err)
	}
	t.Cleanup(d.Close)

	if got := d.Drafter()([]int{1, 2}); got != nil {
		t.Fatalf("draft after callback failure = %v, want nil", got)
	}
	var adapterErr *Qwen35MTPDrafterError
	if !errors.As(d.Err(), &adapterErr) || adapterErr.Stage != "target hidden" || adapterErr.Position != 1 {
		t.Fatalf("latched error = %v, want typed target-hidden error at position 1", d.Err())
	}
	if !errors.Is(d.Err(), callbackErr) {
		t.Fatalf("latched error = %v, want callback cause", d.Err())
	}
	if factory.forwards[0].closes != 1 {
		t.Fatalf("forward closes after callback failure = %d, want 1", factory.forwards[0].closes)
	}
	if got := d.Propose([]int{1}); got != nil {
		t.Fatalf("proposal after latched failure = %v, want nil until reset", got)
	}
	if err := d.Reset(); err != nil {
		t.Fatalf("reset after callback failure: %v", err)
	}
	if got := d.Propose([]int{1}); len(got) != 2 {
		t.Fatalf("proposal after successful reset = %v, want two tokens", got)
	}
	if err := d.Err(); err != nil {
		t.Fatalf("runtime error after successful reset = %v, want nil", err)
	}
}

func TestQwen35MTPDrafterEmbeddingFailureIsLatched(t *testing.T) {
	callbackErr := errors.New("embedding unavailable")
	factory := new(fakeQwen35MTPFactory)
	hidden, _ := testQwen35MTPInputs()
	embedding := func(token int) ([]float32, error) { return nil, callbackErr }
	d, err := newQwen35MTPDrafter(1, hidden, embedding, factory.newForward)
	if err != nil {
		t.Fatalf("construct drafter: %v", err)
	}
	t.Cleanup(d.Close)

	if got := d.Propose([]int{1}); got != nil {
		t.Fatalf("draft after embedding failure = %v, want nil", got)
	}
	var adapterErr *Qwen35MTPDrafterError
	if !errors.As(d.Err(), &adapterErr) || adapterErr.Stage != "token embedding" || adapterErr.Position != 0 {
		t.Fatalf("latched error = %v, want typed token-embedding error at position 0", d.Err())
	}
	if !errors.Is(d.Err(), callbackErr) {
		t.Fatalf("latched error = %v, want embedding callback cause", d.Err())
	}
}

func TestQwen35MTPDrafterForwardFailureIsLatched(t *testing.T) {
	forwardErr := errors.New("native MTP failure")
	factory := &fakeQwen35MTPFactory{make: func() *fakeQwen35MTPForward {
		return &fakeQwen35MTPForward{failAtPos: 1, failErr: forwardErr, emptyAt: -1}
	}}
	hidden, embedding := testQwen35MTPInputs()
	d, err := newQwen35MTPDrafter(2, hidden, embedding, factory.newForward)
	if err != nil {
		t.Fatalf("construct drafter: %v", err)
	}
	t.Cleanup(d.Close)

	if got := d.Propose([]int{1, 2}); got != nil {
		t.Fatalf("draft after forward failure = %v, want nil", got)
	}
	var adapterErr *Qwen35MTPDrafterError
	if !errors.As(d.Err(), &adapterErr) || adapterErr.Stage != "forward" || adapterErr.Position != 1 {
		t.Fatalf("latched error = %v, want typed forward error at position 1", d.Err())
	}
	if !errors.Is(d.Err(), forwardErr) {
		t.Fatalf("latched error = %v, want native forward cause", d.Err())
	}
}

func TestQwen35MTPDrafterConfiguredLength(t *testing.T) {
	for _, k := range []int{1, 3} {
		t.Run(itoa(k), func(t *testing.T) {
			factory := new(fakeQwen35MTPFactory)
			hidden, embedding := testQwen35MTPInputs()
			d, err := newQwen35MTPDrafter(k, hidden, embedding, factory.newForward)
			if err != nil {
				t.Fatalf("construct drafter: %v", err)
			}
			t.Cleanup(d.Close)
			if got := d.Propose([]int{4}); len(got) != k {
				t.Fatalf("configured k=%d returned %d tokens: %v", k, len(got), got)
			}
		})
	}
}

func TestQwen35MTPDrafterResetAndCloseOwnership(t *testing.T) {
	factory := new(fakeQwen35MTPFactory)
	hidden, embedding := testQwen35MTPInputs()
	d, err := newQwen35MTPDrafter(1, hidden, embedding, factory.newForward)
	if err != nil {
		t.Fatalf("construct drafter: %v", err)
	}
	if got := d.Propose([]int{3, 4}); len(got) != 1 {
		t.Fatalf("first draft = %v, want one token", got)
	}
	if err := d.Reset(); err != nil {
		t.Fatalf("reset drafter: %v", err)
	}
	if len(factory.forwards) != 2 || factory.forwards[0].closes != 1 {
		t.Fatalf("reset ownership = %d forwards, old closes %d; want 2 and 1", len(factory.forwards), factory.forwards[0].closes)
	}
	if got := d.Propose([]int{3, 4}); len(got) != 1 || factory.forwards[1].calls[0].pos != 0 {
		t.Fatalf("draft after reset = %v calls=%v, want replay from position 0", got, factory.forwards[1].calls)
	}

	d.Close()
	d.Close()
	if factory.forwards[1].closes != 1 {
		t.Fatalf("current forward closes after double Close = %d, want 1", factory.forwards[1].closes)
	}
	if err := d.Reset(); !errors.Is(err, ErrQwen35MTPDrafterClosed) {
		t.Fatalf("reset after close error = %v, want ErrQwen35MTPDrafterClosed", err)
	}
	if got := d.Propose([]int{3}); got != nil {
		t.Fatalf("draft after close = %v, want nil", got)
	}
	if !errors.Is(d.Err(), ErrQwen35MTPDrafterClosed) {
		t.Fatalf("runtime error after closed proposal = %v, want ErrQwen35MTPDrafterClosed", d.Err())
	}
}

func TestQwen35MTPDrafterPreservesTargetGreedyBoundary(t *testing.T) {
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	prompt := []int{1, 2, 3, 4}
	want := m.NewSession().Generate(prompt, 12)

	factory := &fakeQwen35MTPFactory{make: func() *fakeQwen35MTPForward {
		return &fakeQwen35MTPForward{failAtPos: -1, emptyAt: -1}
	}}
	hidden, embedding := testQwen35MTPInputs()
	d, err := newQwen35MTPDrafter(3, hidden, embedding, factory.newForward)
	if err != nil {
		t.Fatalf("construct drafter: %v", err)
	}
	t.Cleanup(d.Close)

	run, err := SpecDecodeGreedyWithDrafter(m.NewSession(), prompt, 12, 3, d.Drafter())
	if err != nil {
		t.Fatalf("spec decode with MTP drafter: %v", err)
	}
	if err := d.Err(); err != nil {
		t.Fatalf("MTP drafter runtime error: %v", err)
	}
	if !reflect.DeepEqual(run.Output, want) {
		t.Fatalf("spec output = %v, want target-greedy %v", run.Output, want)
	}
}

func TestQwen35MTPDrafterRejectsInvalidSetup(t *testing.T) {
	hidden, embedding := testQwen35MTPInputs()
	factory := new(fakeQwen35MTPFactory)
	for _, tc := range []struct {
		name      string
		k         int
		hidden    Qwen35MTPTargetHidden
		embedding Qwen35MTPTokenEmbedding
		want      error
	}{
		{name: "draft length", k: 0, hidden: hidden, embedding: embedding, want: ErrQwen35MTPInvalidDraftLength},
		{name: "hidden callback", k: 1, embedding: embedding, want: ErrQwen35MTPMissingHidden},
		{name: "embedding callback", k: 1, hidden: hidden, want: ErrQwen35MTPMissingEmbedding},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := newQwen35MTPDrafter(tc.k, tc.hidden, tc.embedding, factory.newForward)
			if !errors.Is(err, tc.want) {
				t.Fatalf("setup error = %v, want %v", err, tc.want)
			}
		})
	}
}

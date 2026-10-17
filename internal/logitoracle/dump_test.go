package logitoracle

import (
	"errors"
	"math"
	"path/filepath"
	"testing"
)

const fakeVocab = 16

// fakeSession is a deterministic history-dependent model. perturb adds a
// candidate-side offset to the logits at a given decode position.
type fakeSession struct {
	history  []int
	steps    int
	perturb  func(step, id int) float32
	prefills [][]int
}

func (f *fakeSession) logits() []float32 {
	h := 0
	for i, id := range f.history {
		h += (id + 1) * (i + 3)
	}
	out := make([]float32, fakeVocab)
	for v := range out {
		out[v] = float32(4 * math.Sin(float64(h%97)*0.37+float64(v)*0.71))
		if f.perturb != nil {
			out[v] += f.perturb(f.steps, v)
		}
	}
	return out
}

func (f *fakeSession) PrefillNoLogits(ids []int) {
	f.prefills = append(f.prefills, append([]int(nil), ids...))
	f.history = append(f.history, ids...)
}

func (f *fakeSession) Prefill(ids []int) []float32 {
	f.PrefillNoLogits(ids)
	return f.logits()
}

func (f *fakeSession) Step(id int) []float32 {
	f.history = append(f.history, id)
	f.steps++
	return f.logits()
}

func factory(perturb func(step, id int) float32, sessions *[]*fakeSession) func() (Session, func(), error) {
	return func() (Session, func(), error) {
		s := &fakeSession{perturb: perturb}
		if sessions != nil {
			*sessions = append(*sessions, s)
		}
		return s, nil, nil
	}
}

func promptSet() []PromptSpec {
	return []PromptSpec{{Name: "a", IDs: []int{1, 2, 3, 4, 5}, Positions: 12}, {Name: "b", IDs: []int{9, 8}, Positions: 6}}
}

func counterSource() func() map[string]int64 {
	n := int64(0)
	return func() map[string]int64 {
		n += 10
		return map[string]int64{"calls": n}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestDumpTeacherForcedRoundTripCompares(t *testing.T) {
	dir := t.TempDir()
	var refSessions []*fakeSession
	ref, err := Dump(promptSet(), DumpOptions{Dir: filepath.Join(dir, "ref"), Label: "ref", PrefillChunk: 2,
		NewSession: factory(nil, &refSessions), Counters: counterSource()})
	if err != nil {
		t.Fatal(err)
	}
	if got := refSessions[0].prefills; len(got) != 3 || len(got[0]) != 2 || len(got[2]) != 1 {
		t.Fatalf("prefill panels %v", got)
	}
	if ref.Vocab != fakeVocab || len(ref.Prompts[0].Fed) != 11 || len(ref.Prompts[0].Argmax) != 12 || ref.Prompts[0].Forced {
		t.Fatalf("ref prompt %+v", ref.Prompts[0])
	}
	if c := ref.Prompts[0].Counters; c["prefill"]["calls"] != 10 || c["decode"]["calls"] != 10 {
		t.Fatalf("counter deltas %v", c)
	}
	for i, fed := range ref.Prompts[0].Fed {
		if fed != ref.Prompts[0].Argmax[i] {
			t.Fatalf("free run must feed its own argmax at %d", i)
		}
	}
	reread, err := ReadManifest(filepath.Join(dir, "ref"))
	if err != nil {
		t.Fatal(err)
	}

	// The candidate flips the greedy token at decode step 4 by a wide margin.
	flip := func(step, id int) float32 {
		if step == 4 && id == (refSessions[0].history[5+4]+1)%fakeVocab {
			return 50
		}
		return 1e-5 * float32(id)
	}
	cand, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "cand"), Label: "cand", PrefillChunk: 2,
		Teacher: reread, NewSession: factory(flip, nil)})
	if err != nil {
		t.Fatal(err)
	}
	if !cand.Prompts[0].Forced || !equalInts(cand.Prompts[0].Fed, ref.Prompts[0].Fed) || cand.Prompts[0].Positions != 12 {
		t.Fatalf("teacher forcing %+v", cand.Prompts[0])
	}
	if _, err := CompareManifests(reread, cand, FlipThresholds); !errors.Is(err, ErrMissingPrompt) {
		t.Fatalf("missing prompt b: %v", err)
	}
	reread.Prompts = reread.Prompts[:1]
	rep, err := CompareManifests(reread, cand, FlipThresholds)
	if err != nil {
		t.Fatal(err)
	}
	p := rep.Prompts[0]
	if p.Compared != 12 || p.Excluded != 0 || len(p.Mismatches) != 1 || p.Mismatches[0].Position != 4 || p.Mismatches[0].Tie {
		t.Fatalf("forced compare %+v", p)
	}
	if rep.Greedy != GreedyNotEquivalent || rep.Verdict != VerdictFail {
		t.Fatalf("report %+v", rep)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestDumpFreeRunningDivergenceExcludesLaterPositions(t *testing.T) {
	dir := t.TempDir()
	var refSessions []*fakeSession
	ref, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "ref"), NewSession: factory(nil, &refSessions)})
	if err != nil {
		t.Fatal(err)
	}
	target := (ref.Prompts[0].Argmax[3] + 1) % fakeVocab
	cand, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "cand"), NewSession: factory(func(step, id int) float32 {
		if step == 3 && id == target {
			return 50
		}
		return 0
	}, nil)})
	if err != nil {
		t.Fatal(err)
	}
	rep, err := CompareManifests(ref, cand, FlipThresholds)
	if err != nil {
		t.Fatal(err)
	}
	p := rep.Prompts[0]
	if p.FirstFedDiverge == nil || *p.FirstFedDiverge != 3 || p.Compared != 4 || p.Excluded != 8 {
		t.Fatalf("free-run compare %+v", p)
	}
	if rep.Verdict != VerdictFail || p.Greedy != GreedyNotEquivalent {
		t.Fatalf("report %+v", rep)
	}
	same, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "same"), NewSession: factory(nil, nil)})
	if err != nil {
		t.Fatal(err)
	}
	if same.Prompts[0].LogitsSHA256 != ref.Prompts[0].LogitsSHA256 {
		t.Fatal("a deterministic session must reproduce its logits file")
	}
	rep, err = CompareManifests(ref, same, StrictThresholds)
	if err != nil || rep.Verdict != VerdictPass || rep.StrictVerdict != VerdictPass || rep.Prompts[0].CosineMin != 1 {
		t.Fatalf("identical dumps %+v %v", rep, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDumpRejectsTeacherPromptMismatch(t *testing.T) {
	dir := t.TempDir()
	ref, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "ref"), NewSession: factory(nil, nil)})
	if err != nil {
		t.Fatal(err)
	}
	other := []PromptSpec{{Name: "a", IDs: []int{7}}}
	if _, err := Dump(other, DumpOptions{Dir: filepath.Join(dir, "c"), Teacher: ref, NewSession: factory(nil, nil)}); !errors.Is(err, ErrTeacherMismatch) {
		t.Fatalf("got %v", err)
	}
	if _, err := Dump(nil, DumpOptions{Dir: dir, NewSession: factory(nil, nil)}); !errors.Is(err, ErrNoPrompts) {
		t.Fatalf("got %v", err)
	}
	if _, err := CompareManifests(ref, ref, Thresholds{TopK: 0}); !errors.Is(err, ErrBadThresholds) {
		t.Fatalf("got %v", err)
	}
}

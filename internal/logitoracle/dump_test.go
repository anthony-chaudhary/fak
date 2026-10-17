package logitoracle

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const fakeVocab = 16

// Error categories are matched as public text so these complete regression
// tests also compile against the pre-fix production package.
const (
	checksumErrorText = "logitoracle: logits checksum"
	historyErrorText  = "logitoracle: invalid fed-token history"
)

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

// fak-test:runtime fast est=100ms lane=default
func TestCompareManifestsVerifiesWholeLogitsFiles(t *testing.T) {
	cases := []struct {
		name      string
		mutate    func(*testing.T, *Manifest)
		wantText  string
		wantError error
	}{
		{"missing checksum", func(t *testing.T, m *Manifest) { m.Prompts[0].LogitsSHA256 = "" }, checksumErrorText, nil},
		{"invalid checksum", func(t *testing.T, m *Manifest) { m.Prompts[0].LogitsSHA256 = strings.Repeat("z", 64) }, checksumErrorText, nil},
		{"short checksum", func(t *testing.T, m *Manifest) { m.Prompts[0].LogitsSHA256 = "ab" }, checksumErrorText, nil},
		{"wrong checksum", func(t *testing.T, m *Manifest) { m.Prompts[0].LogitsSHA256 = strings.Repeat("0", 64) }, checksumErrorText, nil},
		{"changed bytes", corruptLastLogit, checksumErrorText, nil},
		{"truncated file", func(t *testing.T, m *Manifest) {
			p := &m.Prompts[0]
			if err := os.Truncate(filepath.Join(m.Dir(), p.LogitsFile), int64(p.Positions*m.Vocab*4-1)); err != nil {
				t.Fatal(err)
			}
		}, "", ErrLengthMismatch},
	}
	for _, side := range []string{"reference", "candidate"} {
		for _, tc := range cases {
			t.Run(side+"/"+tc.name, func(t *testing.T) {
				dir := t.TempDir()
				ref, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "ref"), NewSession: factory(nil, nil)})
				if err != nil {
					t.Fatal(err)
				}
				cand, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "cand"), NewSession: factory(nil, nil)})
				if err != nil {
					t.Fatal(err)
				}
				if side == "reference" {
					tc.mutate(t, ref)
				} else {
					tc.mutate(t, cand)
				}
				_, err = CompareManifests(ref, cand, FlipThresholds)
				if tc.wantError != nil {
					if !errors.Is(err, tc.wantError) {
						t.Fatalf("got %v, want %v", err, tc.wantError)
					}
				} else if err == nil || !strings.Contains(err.Error(), tc.wantText) {
					t.Fatalf("got %v, want error containing %q", err, tc.wantText)
				}
			})
		}
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestCompareManifestsVerifiesExcludedSuffix(t *testing.T) {
	for _, side := range []string{"reference", "candidate"} {
		t.Run(side, func(t *testing.T) {
			dir := t.TempDir()
			ref, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "ref"), NewSession: factory(nil, nil)})
			if err != nil {
				t.Fatal(err)
			}
			cand, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "cand"), NewSession: factory(nil, nil)})
			if err != nil {
				t.Fatal(err)
			}
			m := cand
			if side == "reference" {
				m = ref
			}
			p := &m.Prompts[0]
			path := filepath.Join(m.Dir(), p.LogitsFile)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			// The excluded suffix extends beyond openRows' 1 MiB read-ahead.
			rowBytes := m.Vocab * 4
			data = bytes.Repeat(data[:rowBytes], (1<<20)/rowBytes+2)
			firstFed := p.Fed[0]
			p.Positions = len(data) / rowBytes
			p.Fed = make([]int, p.Positions-1)
			p.Fed[0] = (firstFed + 1) % m.Vocab
			sum := sha256.Sum256(data)
			p.LogitsSHA256 = hex.EncodeToString(sum[:])
			if err := os.WriteFile(path, data, 0o644); err != nil {
				t.Fatal(err)
			}
			rep, err := CompareManifests(ref, cand, FlipThresholds)
			if err != nil || rep.Verdict != VerdictIncomplete || rep.StrictVerdict != VerdictIncomplete || rep.Prompts[0].Compared != 1 {
				t.Fatalf("valid excluded suffix: report=%+v err=%v", rep, err)
			}
			corruptLastLogit(t, m)
			if _, err := CompareManifests(ref, cand, FlipThresholds); err == nil || !strings.Contains(err.Error(), checksumErrorText) {
				t.Fatalf("corrupt excluded suffix: got %v, want error containing %q", err, checksumErrorText)
			}
		})
	}
}

func corruptLastLogit(t *testing.T, m *Manifest) {
	t.Helper()
	path := filepath.Join(m.Dir(), m.Prompts[0].LogitsFile)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Change only a mantissa bit, preserving length and finite float32 values.
	data[len(data)-4] ^= 1
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestDumpAndCompareRequireCompleteFedHistory(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*PromptDump)
	}{
		{"missing", func(p *PromptDump) { p.Fed = nil }},
		{"short", func(p *PromptDump) { p.Fed = p.Fed[:len(p.Fed)-1] }},
		{"long", func(p *PromptDump) { p.Fed = append(p.Fed, 0) }},
		{"zero positions", func(p *PromptDump) { p.Positions = 0 }},
		{"negative positions", func(p *PromptDump) { p.Positions = -1 }},
		{"negative token", func(p *PromptDump) { p.Fed[0] = -1 }},
		{"out-of-vocabulary token", func(p *PromptDump) { p.Fed[0] = fakeVocab }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			ref, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "ref"), NewSession: factory(nil, nil)})
			if err != nil {
				t.Fatal(err)
			}
			cand, err := Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "cand"), NewSession: factory(nil, nil)})
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(&cand.Prompts[0])
			for _, pair := range [][2]*Manifest{{ref, cand}, {cand, ref}, {cand, cand}} {
				if _, err := CompareManifests(pair[0], pair[1], FlipThresholds); err == nil || !strings.Contains(err.Error(), historyErrorText) {
					t.Fatalf("got %v, want error containing %q", err, historyErrorText)
				}
			}
			started := false
			_, err = Dump(promptSet()[:1], DumpOptions{Dir: filepath.Join(dir, "forced"), Teacher: cand,
				NewSession: func() (Session, func(), error) {
					started = true
					return &fakeSession{}, nil, nil
				}})
			if !errors.Is(err, ErrTeacherMismatch) || started {
				t.Fatalf("invalid teacher must fail before session creation: started=%t err=%v", started, err)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDumpTeacherAcceptsFedTokenBounds(t *testing.T) {
	dir := t.TempDir()
	specs := []PromptSpec{{Name: "bounds", IDs: []int{1, 2}, Positions: 3}}
	ref, err := Dump(specs, DumpOptions{Dir: filepath.Join(dir, "ref"), NewSession: factory(nil, nil)})
	if err != nil {
		t.Fatal(err)
	}
	ref.Prompts[0].Fed = []int{0, fakeVocab - 1}
	cand, err := Dump(specs, DumpOptions{Dir: filepath.Join(dir, "cand"), Teacher: ref, NewSession: factory(nil, nil)})
	if err != nil {
		t.Fatal(err)
	}
	if !equalInts(cand.Prompts[0].Fed, ref.Prompts[0].Fed) {
		t.Fatalf("boundary tokens were not consumed: %+v", cand.Prompts[0])
	}
	rep, err := CompareManifests(cand, cand, FlipThresholds)
	if err != nil || rep.Verdict != VerdictPass || rep.StrictVerdict != VerdictPass {
		t.Fatalf("valid token bounds: report=%+v err=%v", rep, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDumpSinglePositionNeedsNoFedTokens(t *testing.T) {
	dir := t.TempDir()
	specs := []PromptSpec{{Name: "one", IDs: []int{1, 2}, Positions: 1}}
	ref, err := Dump(specs, DumpOptions{Dir: filepath.Join(dir, "ref"), NewSession: factory(nil, nil)})
	if err != nil {
		t.Fatal(err)
	}
	cand, err := Dump(specs, DumpOptions{Dir: filepath.Join(dir, "cand"), Teacher: ref, NewSession: factory(nil, nil)})
	if err != nil {
		t.Fatal(err)
	}
	if len(ref.Prompts[0].Fed) != 0 || len(cand.Prompts[0].Fed) != 0 || !cand.Prompts[0].Forced {
		t.Fatalf("single-position history: ref=%+v cand=%+v", ref.Prompts[0], cand.Prompts[0])
	}
	rep, err := CompareManifests(ref, cand, FlipThresholds)
	if err != nil || rep.Verdict != VerdictPass || rep.StrictVerdict != VerdictPass || rep.Prompts[0].Compared != 1 {
		t.Fatalf("single-position compare: report=%+v err=%v", rep, err)
	}
}

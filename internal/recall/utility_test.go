package recall

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// recordTwoBenign records two equally-relevant benign pages ("refund fee" overlaps
// both with score 2) so a utility credit is the ONLY thing that can reorder them —
// isolating the phase-2 re-rank from phase-1 relevance.
func recordTwoBenign(t *testing.T) *Recorder {
	t.Helper()
	ctx := context.Background()
	r := NewRecorder("utility-two")
	r.Record(ctx, "doc_a", []byte("alpha refund fee summary"))
	r.Record(ctx, "doc_b", []byte("beta refund fee summary"))
	return r
}

func recallSteps(slices []Slice) []int {
	out := make([]int, len(slices))
	for i, sl := range slices {
		out[i] = sl.Step
	}
	return out
}

// TestUtilityDefaultNeutralIsByteIdenticalAndOrderPreserving — acceptance #1: a
// never-credited page carries no utility key in manifest.json (omitempty), and the
// phase-2 re-rank with all-neutral utility yields the exact lexical-only order.
func TestUtilityDefaultNeutralIsByteIdenticalAndOrderPreserving(t *testing.T) {
	ctx := context.Background()
	r := recordTwoBenign(t)

	dir := t.TempDir()
	if err := r.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	mb, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if strings.Contains(string(mb), `"utility":`) {
		t.Fatalf("default-neutral page must not write a utility key (omitempty); manifest:\n%s", mb)
	}

	s, err := Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got := recallSteps(s.Recall(ctx, "refund fee", 5))
	want := []int{0, 1} // stable lexical order: equal score, page order preserved
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("neutral utility changed lexical ranking: got %v want %v", got, want)
	}
}

// TestUtilityWitnessedCreditReranksWithinRelevantSet — acceptance #1 (phase 2): a
// WITNESSED positive outcome lifts the page that helped above an equally-relevant
// page that did not, and the lift survives persist+reload.
func TestUtilityWitnessedCreditReranksWithinRelevantSet(t *testing.T) {
	ctx := context.Background()
	s := persistAndReload(t, recordTwoBenign(t))

	// Before crediting: stable lexical order is [0, 1].
	if got := recallSteps(s.Recall(ctx, "refund fee", 5)); got[0] != 0 {
		t.Fatalf("pre-credit order unexpected: %v", got)
	}

	witness := "recall-test:" + t.Name() + ":ship"
	n, err := s.Credit([]int{1}, Outcome{Witness: witness, Reward: 1.0})
	if err != nil {
		t.Fatalf("witnessed credit refused: %v", err)
	}
	if n != 1 {
		t.Fatalf("expected 1 page credited, got %d", n)
	}
	if got := recallSteps(s.Recall(ctx, "refund fee", 5)); got[0] != 1 {
		t.Fatalf("utility did not lift the witnessed-helpful page to the front: %v", got)
	}

	// Persisted and reloaded, the learned utility (and the lift) survives.
	dir := t.TempDir()
	if err := s.Persist(dir); err != nil {
		t.Fatalf("persist credited session: %v", err)
	}
	mb, _ := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if !strings.Contains(string(mb), `"utility":`) {
		t.Fatalf("credited utility was not persisted to manifest.json:\n%s", mb)
	}
	s2, err := Load(dir)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if got := recallSteps(s2.Recall(ctx, "refund fee", 5)); got[0] != 1 {
		t.Fatalf("utility lift did not survive reload: %v", got)
	}
}

// TestUtilityRefusesSelfAssertedAndRevokedOutcome — acceptance #2: the update path
// takes ONLY a witnessed outcome. A self-asserted (empty-witness) success is refused
// (ErrUnwitnessed) and a revoked-witness outcome is refused (ErrSealed); neither
// mutates any utility.
func TestUtilityRefusesSelfAssertedAndRevokedOutcome(t *testing.T) {
	s := persistAndReload(t, recordTwoBenign(t))

	if _, err := s.Credit([]int{0, 1}, Outcome{Witness: "", Reward: 1.0}); err == nil {
		t.Fatal("self-asserted outcome (empty witness) must be refused")
	} else if !errors.Is(err, ErrUnwitnessed) {
		t.Fatalf("expected ErrUnwitnessed, got %v", err)
	}

	witness := "recall-test:" + t.Name() + ":refuted"
	vdso.Default.Revoke(witness)
	if _, err := s.Credit([]int{0, 1}, Outcome{Witness: witness, Reward: 1.0}); err == nil {
		t.Fatal("revoked-witness outcome must be refused")
	} else if !isSealed(err) {
		t.Fatalf("expected ErrSealed for revoked outcome witness, got %v", err)
	}

	for i, p := range s.Manifest.Pages {
		if p.Utility != 0 {
			t.Fatalf("a refused credit mutated utility on page %d: %v", i, p.Utility)
		}
	}
}

// TestUtilityProvenanceGatesLearning — acceptance #3: a poisoned page can never
// ACCRUE positive utility (Credit skips quarantined and revoked-admission-witness
// pages), even under an otherwise-valid witnessed outcome.
func TestUtilityProvenanceGatesLearning(t *testing.T) {
	ctx := context.Background()

	// recordAirline steps 1 and 3 are quarantined (poison + secret leak).
	sQ := persistAndReload(t, recordAirline(t))
	witness := "recall-test:" + t.Name() + ":ship"
	n, err := sQ.Credit([]int{1, 3}, Outcome{Witness: witness, Reward: 2.0})
	if err != nil {
		t.Fatalf("credit errored: %v", err)
	}
	if n != 0 {
		t.Fatalf("a quarantined page must never accrue utility; credited %d", n)
	}
	for _, step := range []int{1, 3} {
		if u := sQ.Manifest.Pages[step].Utility; u != 0 {
			t.Fatalf("quarantined page %d accrued utility %v", step, u)
		}
	}

	// A benign page whose own admission witness is revoked is also skipped.
	admWitness := "recall-test:" + t.Name() + ":adm"
	r := NewRecorder("util-adm")
	r.RecordWithWitness(ctx, "doc_a", []byte("alpha refund fee summary"), admWitness)
	s := persistAndReload(t, r)
	vdso.Default.Revoke(admWitness)
	n, err = s.Credit([]int{0}, Outcome{Witness: witness, Reward: 2.0})
	if err != nil {
		t.Fatalf("credit errored: %v", err)
	}
	if n != 0 || s.Manifest.Pages[0].Utility != 0 {
		t.Fatalf("a revoked-admission-witness page accrued utility (n=%d util=%v)", n, s.Manifest.Pages[0].Utility)
	}
}

// TestUtilityZeroedWhenWitnessRevoked — acceptance #3 (retention): a page that
// legitimately accrued utility while clean cannot RETAIN it once its witness is
// revoked. The dream seal path zeroes utility as it re-seals the refuted page.
func TestUtilityZeroedWhenWitnessRevoked(t *testing.T) {
	ctx := context.Background()
	admWitness := "recall-test:" + t.Name() + ":adm"

	r := NewRecorder("util-revoke-retain")
	r.RecordWithWitness(ctx, "doc_a", []byte("alpha refund fee summary"), admWitness)
	dir := t.TempDir()
	if err := r.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	s, err := Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Earn utility while the witness is still live.
	if _, err := s.Credit([]int{0}, Outcome{Witness: "recall-test:" + t.Name() + ":ship", Reward: 3.0}); err != nil {
		t.Fatalf("credit: %v", err)
	}
	if s.Manifest.Pages[0].Utility == 0 {
		t.Fatal("page did not accrue utility before revocation")
	}
	if err := s.Persist(dir); err != nil {
		t.Fatalf("persist credited: %v", err)
	}

	// Revoke the admission witness and run the dream seal pass.
	vdso.Default.Revoke(admWitness)
	out := t.TempDir()
	rep, err := Dream(ctx, dir, DreamOptions{OutputDir: out})
	if err != nil {
		t.Fatalf("dream: %v", err)
	}
	if rep.RevokedSeals == 0 {
		t.Fatal("dream did not seal the refuted-witness page")
	}
	sealed, err := Load(out)
	if err != nil {
		t.Fatalf("load sealed image: %v", err)
	}
	p := sealed.Manifest.Pages[0]
	if !p.Quarantined {
		t.Fatal("refuted-witness page was not re-sealed")
	}
	if p.Utility != 0 {
		t.Fatalf("sealed page retained utility %v — a refuted page must not keep its learned weight", p.Utility)
	}
}

// TestUtilityNeverResurrectsIrrelevantPage — phase 1 stays a HARD filter: utility
// re-ranks WITHIN the relevant set and can never pull a score-0 (irrelevant) page
// into the working set, even with maximal accrued utility.
func TestUtilityNeverResurrectsIrrelevantPage(t *testing.T) {
	ctx := context.Background()
	r := NewRecorder("util-irrelevant")
	r.Record(ctx, "doc_topic", []byte("refund fee summary"))      // relevant
	r.Record(ctx, "doc_other", []byte("unrelated weather notes")) // irrelevant to "refund fee"
	s := persistAndReload(t, r)

	if _, err := s.Credit([]int{1}, Outcome{Witness: "recall-test:" + t.Name() + ":ship", Reward: UtilityMax}); err != nil {
		t.Fatalf("credit: %v", err)
	}
	got := recallSteps(s.Recall(ctx, "refund fee", 5))
	for _, step := range got {
		if step == 1 {
			t.Fatalf("utility resurrected an irrelevant page into the working set: %v", got)
		}
	}
}

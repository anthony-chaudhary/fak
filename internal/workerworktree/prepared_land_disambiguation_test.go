package workerworktree

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type preparedDisambiguationObservation struct {
	reads       int
	postApplies int
}

func installPreparedDisambiguationOracle(t *testing.T) *preparedDisambiguationObservation {
	t.Helper()
	oldHas := hasDisambiguationContract
	oldRead := readDisambiguation
	t.Cleanup(func() {
		hasDisambiguationContract = oldHas
		readDisambiguation = oldRead
	})

	obs := &preparedDisambiguationObservation{}
	hasDisambiguationContract = func(context.Context, string, string) (bool, error) {
		return true, nil
	}
	readDisambiguation = func(ctx context.Context, repo string, tree string, setSubphase func(string)) DisambiguationWitness {
		obs.reads++
		if tree != "HEAD" {
			obs.postApplies++
		}
		setSubphase("test-oracle")
		resolved, err := resolveDisambiguationTree(ctx, repo, tree)
		if err != nil {
			return DisambiguationWitness{Tree: tree, Detail: err.Error()}
		}
		return DisambiguationWitness{
			Tree:           tree,
			Fresh:          true,
			SemanticValid:  true,
			CriticalClean:  true,
			Coverage:       100,
			FamilyCoverage: map[string]float64{"loop": 100},
			CacheIdentity:  disambiguationCacheKey(resolved),
			CacheState:     "test",
		}
	}
	return obs
}

func rehashPreparedDisambiguationReceipt(t *testing.T, f preparedFixture, r PreparedLandReceipt) string {
	t.Helper()
	oldPath, err := PreparedLandReceiptPath(f.root, r.ReceiptID, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(oldPath)
	if err != nil {
		t.Fatal(err)
	}
	var stored PreparedLandReceipt
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatal(err)
	}
	newID, err := preparedReceiptID(stored)
	if err != nil {
		t.Fatal(err)
	}
	var receipt map[string]json.RawMessage
	if err := json.Unmarshal(raw, &receipt); err != nil {
		t.Fatal(err)
	}
	receipt["receipt_id"], err = json.Marshal(newID)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	newPath, err := PreparedLandReceiptPath(f.root, newID, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(newPath, append(raw, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	return newID
}

func preparedDisambiguationReceiptKey(t *testing.T, raw map[string]json.RawMessage) string {
	t.Helper()
	for key := range raw {
		if strings.Contains(strings.ToLower(key), "disambiguation") {
			return key
		}
	}
	t.Fatal("prepared receipt did not bind a disambiguation witness")
	return ""
}

func rewritePreparedDisambiguationJSON(t *testing.T, f preparedFixture, r PreparedLandReceipt, mutate func(*testing.T, map[string]json.RawMessage, string)) {
	t.Helper()
	path, err := PreparedLandReceiptPath(f.root, r.ReceiptID, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var receipt map[string]json.RawMessage
	if err := json.Unmarshal(raw, &receipt); err != nil {
		t.Fatal(err)
	}
	key := preparedDisambiguationReceiptKey(t, receipt)
	mutate(t, receipt, key)
	raw, err = json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func alterPreparedDisambiguationIdentity(t *testing.T, raw json.RawMessage) json.RawMessage {
	t.Helper()
	var witness any
	if err := json.Unmarshal(raw, &witness); err != nil {
		t.Fatal(err)
	}
	var alter func(any) bool
	alter = func(value any) bool {
		switch value := value.(type) {
		case map[string]any:
			for _, key := range []string{"analyzer_identity", "cache_identity"} {
				if _, ok := value[key]; ok {
					value[key] = "different-analyzer"
					return true
				}
			}
			for _, child := range value {
				if alter(child) {
					return true
				}
			}
		case []any:
			for _, child := range value {
				if alter(child) {
					return true
				}
			}
		}
		return false
	}
	if !alter(witness) {
		t.Fatal("prepared disambiguation witness has no analyzer identity binding")
	}
	altered, err := json.Marshal(witness)
	if err != nil {
		t.Fatal(err)
	}
	return altered
}

func preparedLandState(t *testing.T, f preparedFixture) (head, recoveryRefs, index string) {
	t.Helper()
	head = strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
	recoveryRefs = mustGit(t, f.root, "for-each-ref", "--format=%(refname) %(objectname)", "refs/fak/recovery")
	index = strings.TrimSpace(mustGit(t, f.root, "write-tree"))
	return head, recoveryRefs, index
}

// fak-test:runtime slow est=90s lane=default
func TestAcceptPreparedLandDisambiguationReusesBoundWitness(t *testing.T) {
	obs := installPreparedDisambiguationOracle(t)
	f := newPreparedFixture(t, "internal/workerworktree/prepared_candidate.go")
	r, prep, verifyObs := prepareCandidate(t, f)
	if !prep.OK || prep.Code != LandResultPrepared {
		t.Fatalf("prepare relevant candidate: %+v", prep)
	}
	if obs.postApplies != 1 || obs.reads != 3 {
		t.Fatalf("prepare oracle operations: post-apply=%d reads=%d, want 1 and 3", obs.postApplies, obs.reads)
	}

	path, err := PreparedLandReceiptPath(f.root, r.ReceiptID, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var stored map[string]json.RawMessage
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatal(err)
	}
	preparedDisambiguationReceiptKey(t, stored)

	reads, postApplies := obs.reads, obs.postApplies
	verifyCalls, prospectiveCalls := verifyObs.verify, verifyObs.prospective
	got := AcceptPreparedLand(f.root, f.wt, preparedWant(f, r), nil)
	if !got.OK || got.Code != LandResultSuccess || !got.Applied || !got.Committed {
		t.Fatalf("accept bound disambiguation witness: %+v", got)
	}
	if obs.reads != reads || obs.postApplies != postApplies {
		t.Fatalf("accept reran disambiguation: reads %d -> %d, post-apply %d -> %d", reads, obs.reads, postApplies, obs.postApplies)
	}
	if verifyObs.verify != verifyCalls || verifyObs.prospective != prospectiveCalls {
		t.Fatal("accept reran candidate verification")
	}
	if head := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); head != r.CandidateSHA {
		t.Fatalf("accepted head=%s candidate=%s", head, r.CandidateSHA)
	}
}

// fak-test:runtime integration est=4m lane=default
func TestAcceptPreparedLandDisambiguationRejectsAlteredWitnessWithoutRefMovement(t *testing.T) {
	tests := []struct {
		name   string
		rehash bool
		mutate func(*testing.T, map[string]json.RawMessage, string)
	}{
		{"missing", false, func(_ *testing.T, receipt map[string]json.RawMessage, key string) {
			delete(receipt, key)
		}},
		{"malformed", false, func(_ *testing.T, receipt map[string]json.RawMessage, key string) {
			receipt[key] = json.RawMessage(`null`)
		}},
		{"altered identity", true, func(t *testing.T, receipt map[string]json.RawMessage, key string) {
			receipt[key] = alterPreparedDisambiguationIdentity(t, receipt[key])
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			obs := installPreparedDisambiguationOracle(t)
			f := newPreparedFixture(t, "internal/workerworktree/prepared_candidate.go")
			r, prep, verifyObs := prepareCandidate(t, f)
			if !prep.OK {
				t.Fatalf("prepare relevant candidate: %+v", prep)
			}
			if obs.postApplies != 1 {
				t.Fatalf("prepare post-apply oracle operations=%d, want 1", obs.postApplies)
			}
			rewritePreparedDisambiguationJSON(t, f, r, tc.mutate)
			want := preparedWant(f, r)
			if tc.rehash {
				want.ReceiptID = rehashPreparedDisambiguationReceipt(t, f, r)
			}
			beforeHead, beforeRecovery, beforeIndex := preparedLandState(t, f)
			reads, postApplies := obs.reads, obs.postApplies
			verifyCalls, prospectiveCalls := verifyObs.verify, verifyObs.prospective

			got := AcceptPreparedLand(f.root, f.wt, want, nil)
			if got.OK || got.Code != LandResultPreparedMismatch || !got.Preserved || got.CommitSHA != "" {
				t.Fatalf("%s witness accepted: %+v", tc.name, got)
			}
			afterHead, afterRecovery, afterIndex := preparedLandState(t, f)
			if afterHead != beforeHead || afterRecovery != beforeRecovery || afterIndex != beforeIndex {
				t.Fatalf("%s witness mutated repository state: head %q -> %q recovery %q -> %q index %q -> %q",
					tc.name, beforeHead, afterHead, beforeRecovery, afterRecovery, beforeIndex, afterIndex)
			}
			if obs.reads != reads || obs.postApplies != postApplies || verifyObs.verify != verifyCalls || verifyObs.prospective != prospectiveCalls {
				t.Fatalf("%s witness refusal reran an accepted gate", tc.name)
			}
		})
	}
}

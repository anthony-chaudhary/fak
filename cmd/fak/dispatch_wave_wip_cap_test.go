package main

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchWaveFreshStartCapLimitsNewFrontsButNotAttemptedWIP(t *testing.T) {
	router := dispatchtick.RouterPayload{
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"a": {Count: 1, Issues: []int{10}, Tree: []string{"internal/a"}},
			"b": {Count: 1, Issues: []int{20}, Tree: []string{"internal/b"}},
		},
	}

	t.Run("default cap admits one fresh front", func(t *testing.T) {
		price, err := priceDispatchWavePayloadFilteredWithFreshCap(t.TempDir(), router, 2, 2, "", nil, 0, nil, dispatchWaveDefaultFreshCap, dispatchGoalProfileThroughput)
		if err != nil {
			t.Fatal(err)
		}
		if len(price.RunTargets) != 1 || price.FreshStarts != 1 || price.FreshStartCap != 1 {
			t.Fatalf("price = %+v, want one fresh target under cap 1", price)
		}
		held := 0
		for _, cand := range price.Candidates {
			if cand.Reason == dispatchWaveReasonFreshWIPCap {
				held++
			}
		}
		if held != 1 {
			t.Fatalf("fresh-cap held rows = %d, want 1; candidates=%+v", held, price.Candidates)
		}
	})

	t.Run("attempted WIP keeps full safe concurrency", func(t *testing.T) {
		root := t.TempDir()
		writeDispatchAttemptWitness(t, root, 10)
		writeDispatchAttemptWitness(t, root, 20)
		price, err := priceDispatchWavePayloadFilteredWithFreshCap(root, router, 2, 2, "", nil, 0, nil, dispatchWaveDefaultFreshCap, dispatchGoalProfileThroughput)
		if err != nil {
			t.Fatal(err)
		}
		if len(price.RunTargets) != 2 || price.FreshStarts != 0 {
			t.Fatalf("targets=%+v fresh_starts=%d, want both attempted issues and zero fresh starts", price.RunTargets, price.FreshStarts)
		}
	})

	t.Run("explicit breadth override admits two fresh fronts", func(t *testing.T) {
		price, err := priceDispatchWavePayloadFilteredWithFreshCap(t.TempDir(), router, 2, 2, "", nil, 0, nil, 2, dispatchGoalProfileThroughput)
		if err != nil {
			t.Fatal(err)
		}
		if len(price.RunTargets) != 2 || price.FreshStarts != 2 || price.FreshStartCap != 2 {
			t.Fatalf("price = %+v, want two fresh targets under explicit cap 2", price)
		}
	})
}

func TestDispatchWaveDivergenceCapPreservesFinishers(t *testing.T) {
	root := t.TempDir()
	writeDispatchAttemptWitness(t, root, 10)
	router := dispatchtick.RouterPayload{Lanes: map[string]dispatchtick.RouterLaneGroup{
		"finish": {Count: 1, Issues: []int{10}, Tree: []string{"internal/finish"}},
		"fresh":  {Count: 1, Issues: []int{20}, Tree: []string{"internal/fresh"}},
	}}
	admission := evaluateDispatchFinishFirstAdmission(dispatchFinishFirstAdmissionInput{
		EvidenceAvailable: true, GitHubAvailable: true, WIPFilesDelta: 86, WIPLinesDelta: 500,
		OldestWIPMinutes: 5210, RequestedFreshStarts: 2, Finishers: 2,
	})
	price, err := priceDispatchWavePayloadFilteredWithFreshCap(root, router, 2, 2, "", nil, 0, nil, admission.AllowedFreshStarts, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.RunTargets) != 1 || price.RunTargets[0].Issue != 10 || price.FreshStarts != 0 {
		t.Fatalf("price = %+v, want attempted issue 10 only", price)
	}
	if admission.AllowedFinishers != 2 || admission.DeniedFreshStarts != 2 {
		t.Fatalf("admission = %+v", admission)
	}
}

func writeDispatchAttemptWitness(t *testing.T, root string, issue int) {
	t.Helper()
	runs := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runs, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(runs, "resolve-"+strconv.Itoa(issue)+"-previous.witness")
	if err := os.WriteFile(path, []byte("finished\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	attemptedAt := time.Now().Add(-3 * time.Hour).Truncate(time.Second)
	if err := os.Chtimes(path, attemptedAt, attemptedAt); err != nil {
		t.Fatal(err)
	}
}

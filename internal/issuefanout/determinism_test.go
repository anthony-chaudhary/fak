package issuefanout

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// TestDeterminismCandidateEvaluation runs candidate and fanout evaluation twice with
// identical inputs and asserts deep equality of output structures, rendered fields,
// strict policy reviews, and serialized representations.
func TestDeterminismCandidateEvaluation(t *testing.T) {
	cases := []struct {
		name  string
		input Input
	}{
		{
			name: "full strict production fanout",
			input: Input{
				Title:              "strict live fanout contracts",
				Leaf:               "issuefanout",
				SpineRef:           "8dd2bd480c4e8ba87d31d6fbffd2150ce05f3324",
				ParentIssue:        9512,
				ParentBaseline:     8.0,
				CompletionStandard: "production",
				TargetEnvelope:     "- strict-reviewed generated children: >= 3 issues\n- partial live writes on contract refusal: = 0 issues",
				WitnessedEnvelope:  "- strict-reviewed generated children: 3 issues\n- partial live writes on contract refusal: 0 issues",
			},
		},
		{
			name: "qa area filter",
			input: Input{
				Title:              "strict live fanout qa",
				Leaf:               "issuefanout",
				SpineRef:           "8dd2bd480c4e8ba87d31d6fbffd2150ce05f3324",
				ParentIssue:        9512,
				ParentBaseline:     8.0,
				CompletionStandard: "production",
				TargetEnvelope:     "- strict-reviewed generated children: >= 3 issues",
				WitnessedEnvelope:  "- strict-reviewed generated children: 3 issues",
				Areas:              []string{"qa"},
			},
		},
		{
			name: "capped to floor with demo maturity",
			input: Input{
				Title:              "strict live fanout demo",
				Leaf:               "issuefanout",
				SpineRef:           "8dd2bd480c4e8ba87d31d6fbffd2150ce05f3324",
				ParentIssue:        9512,
				ParentBaseline:     8.0,
				CompletionStandard: "demo",
				Max:                MinFanout,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			firstPlan, err1 := Build(tc.input)
			if err1 != nil {
				t.Fatalf("first Build: %v", err1)
			}
			secondPlan, err2 := Build(tc.input)
			if err2 != nil {
				t.Fatalf("second Build: %v", err2)
			}

			if !reflect.DeepEqual(firstPlan, secondPlan) {
				t.Fatalf("Build produced different plans on identical input:\nfirst:  %+v\nsecond: %+v", firstPlan, secondPlan)
			}

			firstBytes, err := json.Marshal(firstPlan)
			if err != nil {
				t.Fatalf("marshal first plan: %v", err)
			}
			secondBytes, err := json.Marshal(secondPlan)
			if err != nil {
				t.Fatalf("marshal second plan: %v", err)
			}
			if !bytes.Equal(firstBytes, secondBytes) {
				t.Fatalf("plans serialized to different JSON bytes:\nfirst:  %s\nsecond: %s", firstBytes, secondBytes)
			}

			strictLiveOpts := issuepolicy.Options{
				Live:              true,
				DedupeChecked:     true,
				DedupeCap:         DefaultDedupeCap,
				StrictModelTier:   true,
				StrictScale:       true,
				StrictWitness:     true,
				StrictBornRouted:  true,
				StrictProjectWork: tc.input.ParentIssue > 0 && tc.input.ParentBaseline > 0,
			}

			for i := range firstPlan.Candidates {
				c1 := firstPlan.Candidates[i]
				c2 := secondPlan.Candidates[i]

				if !reflect.DeepEqual(c1, c2) {
					t.Fatalf("candidate %d differed:\nfirst:  %+v\nsecond: %+v", i, c1, c2)
				}

				draft1 := liveIssueDraft(c1)
				draft2 := liveIssueDraft(c2)
				if !reflect.DeepEqual(draft1, draft2) {
					t.Fatalf("candidate %d liveIssueDraft differed:\nfirst:  %+v\nsecond: %+v", i, draft1, draft2)
				}

				body1 := LiveBody(c1)
				body2 := LiveBody(c2)
				if body1 != body2 {
					t.Fatalf("candidate %d LiveBody differed:\nfirst:  %s\nsecond: %s", i, body1, body2)
				}

				labels1 := LiveLabels(c1)
				labels2 := LiveLabels(c2)
				if !reflect.DeepEqual(labels1, labels2) {
					t.Fatalf("candidate %d LiveLabels differed:\nfirst:  %v\nsecond: %v", i, labels1, labels2)
				}

				review1 := issuepolicy.ReviewIssueDraft(draft1, strictLiveOpts)
				review2 := issuepolicy.ReviewIssueDraft(draft2, strictLiveOpts)
				if !reflect.DeepEqual(review1, review2) {
					t.Fatalf("candidate %d strict ReviewIssueDraft differed:\nfirst:  %+v\nsecond: %+v", i, review1, review2)
				}
				if !review1.OK || review1.Dispatchability != issuepolicy.Dispatchable {
					t.Fatalf("candidate %d not dispatchable under strict review: verdict=%s reasons=%v missing=%v",
						i, review1.Verdict, review1.Reasons, review1.MissingFields)
				}
			}
		})
	}
}

// TestDeterminismStrictLiveFanoutContracts evaluates live filing behavior twice
// against identical inputs (plan, existing issues, and runner responses) and
// asserts byte-identical, deeply equal filing outcomes.
func TestDeterminismStrictLiveFanoutContracts(t *testing.T) {
	plan, err := Build(Input{
		Title:              "strict live fanout contracts",
		Leaf:               "issuefanout",
		SpineRef:           "8dd2bd480c4e8ba87d31d6fbffd2150ce05f3324",
		ParentIssue:        9512,
		ParentBaseline:     8.0,
		CompletionStandard: "production",
		TargetEnvelope:     "- strict-reviewed generated children: >= 3 issues\n- partial live writes on contract refusal: = 0 issues",
		WitnessedEnvelope:  "- strict-reviewed generated children: 3 issues\n- partial live writes on contract refusal: 0 issues",
		Areas:              []string{"qa"},
	})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	seenKey := plan.Candidates[0].Key
	existing := []Issue{
		{Number: 42, Body: "prior issue carrying " + seenKey + " in discussion"},
	}

	runFiling := func() (LiveResult, [][]string, error) {
		calls := make([][]string, 0)
		runner := func(args []string) (string, string, bool) {
			calls = append(calls, append([]string(nil), args...))
			return fmt.Sprintf("https://github.com/o/r/issues/%d", 9550+len(calls)), "", true
		}
		res, err := FileLive(plan, append([]Issue(nil), existing...), LiveOptions{
			DedupeCap: 50,
			Runner:    runner,
		})
		return res, calls, err
	}

	firstRes, firstCalls, err1 := runFiling()
	if err1 != nil {
		t.Fatalf("first FileLive: %v", err1)
	}
	secondRes, secondCalls, err2 := runFiling()
	if err2 != nil {
		t.Fatalf("second FileLive: %v", err2)
	}

	if !reflect.DeepEqual(firstRes, secondRes) {
		t.Fatalf("FileLive produced different LiveResults on identical inputs:\nfirst:  %+v\nsecond: %+v", firstRes, secondRes)
	}
	if !reflect.DeepEqual(firstCalls, secondCalls) {
		t.Fatalf("FileLive invoked different runner calls:\nfirst:  %v\nsecond: %v", firstCalls, secondCalls)
	}

	firstRender := RenderLive(firstRes)
	secondRender := RenderLive(secondRes)
	if firstRender != secondRender {
		t.Fatalf("RenderLive produced different strings:\nfirst:  %s\nsecond: %s", firstRender, secondRender)
	}

	firstJSON, err := json.Marshal(firstRes)
	if err != nil {
		t.Fatalf("marshal first live result: %v", err)
	}
	secondJSON, err := json.Marshal(secondRes)
	if err != nil {
		t.Fatalf("marshal second live result: %v", err)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatalf("live results serialized to different JSON bytes:\nfirst:  %s\nsecond: %s", firstJSON, secondJSON)
	}
}

// TestDeterminismStrictLiveConcurrentRaceWitness executes candidate expansion,
// strict review, and live filing adjudication across concurrent workers to witness
// that strict live fanout contract evaluation is race-free and deterministic.
func TestDeterminismStrictLiveConcurrentRaceWitness(t *testing.T) {
	in := Input{
		Title:              "strict live fanout contracts",
		Leaf:               "issuefanout",
		SpineRef:           "8dd2bd480c4e8ba87d31d6fbffd2150ce05f3324",
		ParentIssue:        9512,
		ParentBaseline:     8.0,
		CompletionStandard: "production",
		TargetEnvelope:     "- strict-reviewed generated children: >= 3 issues\n- partial live writes on contract refusal: = 0 issues",
		WitnessedEnvelope:  "- strict-reviewed generated children: 3 issues\n- partial live writes on contract refusal: 0 issues",
		Areas:              []string{"qa"},
	}

	refPlan, err := Build(in)
	if err != nil {
		t.Fatalf("reference Build: %v", err)
	}

	seenKey := refPlan.Candidates[0].Key
	existing := []Issue{
		{Number: 42, Body: "existing marker " + seenKey},
	}

	refCalls := make([][]string, 0)
	refRunner := func(args []string) (string, string, bool) {
		refCalls = append(refCalls, append([]string(nil), args...))
		return fmt.Sprintf("https://github.com/o/r/issues/%d", 9600+len(refCalls)), "", true
	}
	refRes, err := FileLive(refPlan, append([]Issue(nil), existing...), LiveOptions{
		DedupeCap: 50,
		Runner:    refRunner,
	})
	if err != nil {
		t.Fatalf("reference FileLive: %v", err)
	}

	const workers = 32
	var wg sync.WaitGroup
	errCh := make(chan error, workers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			plan, buildErr := Build(in)
			if buildErr != nil {
				errCh <- fmt.Errorf("concurrent Build: %w", buildErr)
				return
			}
			if !reflect.DeepEqual(plan, refPlan) {
				errCh <- fmt.Errorf("concurrent Plan differed from reference")
				return
			}

			strictOpts := issuepolicy.Options{
				Live:              true,
				DedupeChecked:     true,
				DedupeCap:         DefaultDedupeCap,
				StrictModelTier:   true,
				StrictScale:       true,
				StrictWitness:     true,
				StrictBornRouted:  true,
				StrictProjectWork: true,
			}
			for i, c := range plan.Candidates {
				draft := liveIssueDraft(c)
				review := issuepolicy.ReviewIssueDraft(draft, strictOpts)
				if !review.OK || review.Dispatchability != issuepolicy.Dispatchable {
					errCh <- fmt.Errorf("candidate %d failed strict review: %v", i, review.Reasons)
					return
				}
			}

			calls := make([][]string, 0)
			runner := func(args []string) (string, string, bool) {
				calls = append(calls, append([]string(nil), args...))
				return fmt.Sprintf("https://github.com/o/r/issues/%d", 9600+len(calls)), "", true
			}
			res, fileErr := FileLive(plan, append([]Issue(nil), existing...), LiveOptions{
				DedupeCap: 50,
				Runner:    runner,
			})
			if fileErr != nil {
				errCh <- fmt.Errorf("concurrent FileLive: %w", fileErr)
				return
			}
			if !reflect.DeepEqual(res, refRes) {
				errCh <- fmt.Errorf("concurrent LiveResult differed from reference")
				return
			}
			if !reflect.DeepEqual(calls, refCalls) {
				errCh <- fmt.Errorf("concurrent runner calls differed from reference")
				return
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
}

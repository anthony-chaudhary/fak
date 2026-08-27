package studyforge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func pointedInt(value *int) int {
	if value == nil {
		return -1
	}
	return *value
}

func TestCaptureRecordsNonAtomicDeltaForConcurrentCreationAndDeletion(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{11, 12}, []int64{12, 13})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(corpus); err != nil {
		t.Fatal(err)
	}
	delta := corpus.Receipt.NonAtomicDelta
	if delta == nil || !delta.Accepted || delta.Type != NonAtomicDeltaType || delta.EvidenceMode != NonAtomicDeltaEvidenceModeExactIdentity || delta.IdentityBasis != identityBasisCaptured {
		t.Fatalf("non_atomic_delta = %+v", delta)
	}
	if delta.MixedCount != 2 || delta.DedicatedCount != 2 || pointedInt(delta.OverlapCount) != 1 || pointedInt(delta.OnlyInMixedCount) != 1 || pointedInt(delta.OnlyInDedicatedCount) != 1 {
		t.Fatalf("non_atomic_delta counts = %+v", delta)
	}
	assertIdentityIDs(t, delta.Overlap, 12)
	assertIdentityIDs(t, delta.OnlyInMixed, 11)
	assertIdentityIDs(t, delta.OnlyInDedicated, 13)
	if delta.MixedCrawl.StartedAt == "" || delta.MixedCrawl.EndedAt == "" || delta.DedicatedCrawl.StartedAt == "" || delta.DedicatedCrawl.EndedAt == "" {
		t.Fatalf("missing crawl windows: %+v", delta)
	}
	if got := recordsForSource(corpus.Records, "issues"); len(got) != 0 {
		t.Fatalf("mixed endpoint PR-shaped rows leaked into issue-only records: %+v", got)
	}
	if got := recordsForSource(corpus.Records, "pulls"); len(got) != 2 || got[0].ID != 12 || got[1].ID != 13 {
		t.Fatalf("dedicated pulls are not authoritative normalized rows: %+v", got)
	}
}

func TestValidateRejectsMissingOrContradictoryNonAtomicDelta(t *testing.T) {
	base, err := captureReconciliationFixture(t, []int64{11, 12}, []int64{12, 13})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		want string
		edit func(*Corpus)
	}{
		{name: "missing", want: "require non_atomic_delta", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta = nil }},
		{name: "contradictory set", want: "identity sets contradict", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.OnlyInDedicated[0].ID = 99 }},
		{name: "contradictory count", want: "counts contradict", edit: func(c *Corpus) { *c.Receipt.NonAtomicDelta.OverlapCount++ }},
		{name: "contradictory verdict", want: "accepted verdict contradicts", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.Accepted = false }},
		{name: "unbounded policy", want: "policy is missing or exceeds", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.Policy.MaxTotal = DefaultNonAtomicDeltaLimit + 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			corpus := cloneCorpus(base)
			tt.edit(&corpus)
			if err := Validate(corpus); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestValidateAcceptsAndRewritesAlreadyUpgradedExactLegacyProjection(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{11, 12}, []int64{12, 13})
	if err != nil {
		t.Fatal(err)
	}
	delta := corpus.Receipt.NonAtomicDelta
	delta.EvidenceMode = ""
	delta.EvidenceReason = ""
	delta.IdentityBasis = identityBasisLegacyProjection
	delta.MixedEvidence = ""
	delta.DedicatedEvidence = ""
	delta.RelationEvidence = ""
	delta.SymmetricDifferenceLowerBound = 0
	delta.SymmetricDifferenceUpperBound = 0
	delta.Verdict = ""
	if err := Validate(corpus); err != nil {
		t.Fatalf("already-upgraded exact corpus rejected: %v", err)
	}
	if err := reconcileNonAtomicDelta(&corpus); err != nil {
		t.Fatalf("rewrite exact projection: %v", err)
	}
	got := corpus.Receipt.NonAtomicDelta
	if got.EvidenceMode != NonAtomicDeltaEvidenceModeExactIdentity || got.IdentityBasis != identityBasisLegacyProjection || got.RelationEvidence != CrossEndpointEvidenceExactIdentities {
		t.Fatalf("rewritten exact projection = %+v", got)
	}
}

func TestCaptureRecordsPolicyOverflowAsPartial(t *testing.T) {
	dedicated := make([]int64, DefaultNonAtomicDeltaLimit+1)
	for i := range dedicated {
		dedicated[i] = int64(i + 1)
	}
	corpus, err := captureReconciliationFixture(t, nil, dedicated)
	if err == nil || !strings.Contains(err.Error(), "exceeds policy") {
		t.Fatalf("Capture error = %v", err)
	}
	if corpus.Receipt.Status != StatusPartial || corpus.Receipt.NonAtomicDelta == nil || corpus.Receipt.NonAtomicDelta.Accepted {
		t.Fatalf("overflow receipt = %+v", corpus.Receipt)
	}
	if err := validateCheckpoint(corpus); err != nil {
		t.Fatalf("overflow partial must remain a resumable, internally consistent checkpoint: %v", err)
	}
	if err := Validate(corpus); err == nil || !strings.Contains(err.Error(), "policy overflow") {
		t.Fatalf("Validate error = %v, want policy overflow", err)
	}
}

func TestCaptureRejectsDuplicateMixedPullIdentities(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{7, 7}, []int64{7})
	if err == nil || !strings.Contains(err.Error(), "duplicate issues id 7") {
		t.Fatalf("Capture error = %v", err)
	}
	issues, _ := sourceByName(corpus.Receipt.Sources, "issues")
	if len(issues.Pages) != 0 || issues.ClassifiedPullCount != 0 || len(issues.ClassifiedPullIdentities) != 0 {
		t.Fatalf("duplicate identity page was accepted: %+v", issues)
	}
}

func TestCaptureResumesLegacyCheckpointWithoutRefetchingCompleteEndpoints(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var (
		mu              sync.Mutex
		calls           = map[string]int{}
		failDiscussions = true
		server          *httptest.Server
	)
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.Path]++
		fail := failDiscussions
		mu.Unlock()
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"legacy-revision"}`)
		case "/repos/acme/widget/issues":
			writeRows(w, []int64{21}, true)
		case "/repos/acme/widget/pulls":
			writeRows(w, []int64{21}, false)
		case "/repos/acme/widget/discussions":
			if fail {
				http.Error(w, "interrupted", http.StatusInternalServerError)
				return
			}
			fmt.Fprint(w, `[]`)
		case "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.MaxRetries = 0
	collector.Now = func() time.Time { return cutoff }
	partial, err := collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
	if err == nil || partial.Receipt.Status != StatusPartial {
		t.Fatalf("initial Capture = %v status=%s", err, partial.Receipt.Status)
	}
	partial.Receipt.NonAtomicDelta = nil
	for i := range partial.Receipt.Sources {
		partial.Receipt.Sources[i].CrawlStartedAt = ""
		partial.Receipt.Sources[i].CrawlEndedAt = ""
		if partial.Receipt.Sources[i].Name == "issues" {
			partial.Receipt.Sources[i].ClassifiedPullIdentities = nil
			partial.Receipt.Sources[i].ClassifiedPullChecksum = ""
		}
	}
	if err := validateResumeCheckpoint(partial); err != nil {
		t.Fatalf("legacy checkpoint rejected: %v", err)
	}
	mu.Lock()
	issuesBefore, pullsBefore := calls["/repos/acme/widget/issues"], calls["/repos/acme/widget/pulls"]
	failDiscussions = false
	mu.Unlock()
	var checkpoints []Corpus
	completed, err := collector.Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: cutoff, Resume: &partial,
		Checkpoint: func(c Corpus) error {
			checkpoints = append(checkpoints, cloneCorpus(c))
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(completed); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	issuesAfter, pullsAfter := calls["/repos/acme/widget/issues"], calls["/repos/acme/widget/pulls"]
	mu.Unlock()
	if issuesAfter != issuesBefore || pullsAfter != pullsBefore {
		t.Fatalf("resume refetched complete endpoints: issues %d->%d pulls %d->%d", issuesBefore, issuesAfter, pullsBefore, pullsAfter)
	}
	if completed.Receipt.NonAtomicDelta == nil || completed.Receipt.NonAtomicDelta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly || completed.Receipt.NonAtomicDelta.IdentityBasis != identityBasisLegacyCountOnly {
		t.Fatalf("legacy reconciliation evidence = %+v", completed.Receipt.NonAtomicDelta)
	}
	if len(checkpoints) == 0 {
		t.Fatal("resume did not persist an upgraded checkpoint")
	}
	persistedIssues, _ := sourceByName(checkpoints[0].Receipt.Sources, "issues")
	if persistedIssues.ClassifiedPullIdentities != nil || persistedIssues.ClassifiedPullChecksum != "" {
		t.Fatalf("first resumed checkpoint fabricated mixed identities: %+v", persistedIssues)
	}
	if checkpoints[0].Receipt.NonAtomicDelta == nil || checkpoints[0].Receipt.NonAtomicDelta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly {
		t.Fatalf("first resumed checkpoint delta = %+v", checkpoints[0].Receipt.NonAtomicDelta)
	}
}

func TestCaptureMigratesLegacyCheckpointAfterCompletingPullCensus(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var (
		mu        sync.Mutex
		calls     = map[string]int{}
		failPage2 = true
		server    *httptest.Server
	)
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.Path+"?"+r.URL.RawQuery]++
		fail := failPage2
		mu.Unlock()
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"legacy-page-revision"}`)
		case "/repos/acme/widget/issues":
			writeRows(w, []int64{31, 32}, true)
		case "/repos/acme/widget/pulls":
			if r.URL.Query().Get("page") == "2" {
				if fail {
					http.Error(w, "interrupted", http.StatusInternalServerError)
					return
				}
				w.Header().Set("Link", `<`+server.URL+`/repos/acme/widget/pulls?page=3>; rel="next"`)
				writeRows(w, []int64{32}, false)
				return
			}
			if r.URL.Query().Get("page") == "3" {
				fmt.Fprint(w, `[]`)
				return
			}
			w.Header().Set("Link", `<`+server.URL+`/repos/acme/widget/pulls?page=2>; rel="next"`)
			writeRows(w, []int64{31}, false)
		case "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.MaxRetries = 0
	collector.Now = func() time.Time { return cutoff }
	partial, err := collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
	if err == nil {
		t.Fatal("initial capture unexpectedly completed")
	}
	legacyizeMixedPullCheckpoint(&partial)
	if err := validateResumeCheckpoint(partial); err != nil {
		t.Fatalf("legacy checkpoint rejected: %v", err)
	}
	if err := validateCheckpoint(partial); err == nil || !strings.Contains(err.Error(), "classified pull identity count") {
		t.Fatalf("ordinary checkpoint validation error = %v, want strict marker rejection", err)
	}
	legacyPath := filepath.Join(t.TempDir(), "legacy.json")
	legacyJSON, err := json.Marshal(partial)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacyPath, legacyJSON, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(legacyPath); err == nil {
		t.Fatal("ordinary Read accepted a legacy markerless checkpoint")
	}
	loaded, err := ReadResume(legacyPath)
	if err != nil {
		t.Fatalf("ReadResume rejected legacy checkpoint: %v", err)
	}
	partial = loaded
	page1Key := "/repos/acme/widget/pulls?"
	issuesKey := "/repos/acme/widget/issues?"
	mu.Lock()
	page1Before, issuesBefore := calls[page1Key], calls[issuesKey]
	page2Before := calls["/repos/acme/widget/pulls?page=2"]
	page3Before := calls["/repos/acme/widget/pulls?page=3"]
	anyCheckpointWithoutTypedEvidence := false
	failPage2 = false
	mu.Unlock()
	var persisted []Corpus
	completed, err := collector.Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: cutoff, Resume: &partial,
		Checkpoint: func(c Corpus) error {
			if c.Receipt.NonAtomicDelta == nil || c.Receipt.NonAtomicDelta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly {
				anyCheckpointWithoutTypedEvidence = true
			}
			persisted = append(persisted, cloneCorpus(c))
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	page1After, issuesAfter := calls[page1Key], calls[issuesKey]
	page2After := calls["/repos/acme/widget/pulls?page=2"]
	page3After := calls["/repos/acme/widget/pulls?page=3"]
	mu.Unlock()
	if page1After != page1Before || issuesAfter != issuesBefore || page2After != page2Before+1 || page3After != page3Before+1 {
		t.Fatalf("resume requests: issues %d->%d pull page1 %d->%d pull page2 %d->%d pull page3 %d->%d", issuesBefore, issuesAfter, page1Before, page1After, page2Before, page2After, page3Before, page3After)
	}
	if anyCheckpointWithoutTypedEvidence || len(persisted) == 0 {
		t.Fatalf("persisted=%d checkpoint_without_typed_evidence=%t", len(persisted), anyCheckpointWithoutTypedEvidence)
	}
	issues, _ := sourceByName(completed.Receipt.Sources, "issues")
	if issues.ClassifiedPullIdentities != nil || issues.ClassifiedPullChecksum != "" {
		t.Fatalf("completed legacy evidence fabricated mixed identities: %+v", issues)
	}
	if completed.Receipt.NonAtomicDelta == nil || completed.Receipt.NonAtomicDelta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly {
		t.Fatalf("completed legacy delta = %+v", completed.Receipt.NonAtomicDelta)
	}
}

func TestLegacyMixedPullMigrationNeverProjectsDedicatedIdentities(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{41}, []int64{41, 42})
	if err != nil {
		t.Fatal(err)
	}
	legacyizeMixedPullCheckpoint(&corpus)
	requests := 0
	collector := NewCollector(&http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		requests++
		return nil, errors.New("unexpected request")
	})})
	collector.BaseURL = corpus.Receipt.APIBase
	completed, err := collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoffTime(t, corpus.Receipt.Cutoff), Resume: &corpus})
	if err != nil {
		t.Fatalf("Capture error = %v", err)
	}
	if requests != 0 {
		t.Fatalf("terminal resume made %d requests", requests)
	}
	issues, _ := sourceByName(completed.Receipt.Sources, "issues")
	if issues.ClassifiedPullIdentities != nil || issues.ClassifiedPullChecksum != "" {
		t.Fatalf("dedicated identities were projected into mixed evidence: %+v", issues)
	}
	if delta := completed.Receipt.NonAtomicDelta; delta == nil || delta.RelationEvidence != CrossEndpointEvidenceUnavailable || delta.Overlap != nil || delta.OnlyInMixed != nil || delta.OnlyInDedicated != nil {
		t.Fatalf("legacy relation evidence = %+v", delta)
	}
}

func TestLegacyCountOnlyBoundsAndPolicyVerdicts(t *testing.T) {
	tests := []struct {
		name        string
		mixed       int
		dedicated   int
		wantLower   int
		wantUpper   int
		wantVerdict string
	}{
		{name: "observed partial counts", mixed: 35528, dedicated: 34800, wantLower: 728, wantUpper: 70328, wantVerdict: NonAtomicDeltaVerdictCompatibleUnproven},
		{name: "observed temporal drift", mixed: 35528, dedicated: 35504, wantLower: 24, wantUpper: 71032, wantVerdict: NonAtomicDeltaVerdictCompatibleUnproven},
		{name: "absurd forced delta", mixed: 35528, dedicated: 34000, wantLower: 1528, wantUpper: 69528, wantVerdict: NonAtomicDeltaVerdictRejected},
		{name: "small corpus proven", mixed: 10, dedicated: 9, wantLower: 1, wantUpper: 19, wantVerdict: NonAtomicDeltaVerdictAccepted},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			corpus := Corpus{Records: make([]Record, tt.dedicated)}
			for i := range corpus.Records {
				corpus.Records[i].Source = "pulls"
			}
			reconcileLegacyCountOnlyDelta(&corpus,
				SourceReceipt{Name: "issues", ClassifiedPullCount: tt.mixed},
				SourceReceipt{Name: "pulls"},
			)
			delta := corpus.Receipt.NonAtomicDelta
			if delta.SymmetricDifferenceLowerBound != tt.wantLower || delta.SymmetricDifferenceUpperBound != tt.wantUpper || delta.Verdict != tt.wantVerdict || delta.Accepted != (tt.wantVerdict == NonAtomicDeltaVerdictAccepted) {
				t.Fatalf("delta = %+v", delta)
			}
			if delta.Overlap != nil || delta.OnlyInMixed != nil || delta.OnlyInDedicated != nil || delta.OverlapCount != nil || delta.OnlyInMixedCount != nil || delta.OnlyInDedicatedCount != nil {
				t.Fatalf("count-only relation sets became available: %+v", delta)
			}
		})
	}
}

func TestValidateLegacyCountOnlyRejectsContradictoryShapes(t *testing.T) {
	base, err := captureReconciliationFixture(t, []int64{61}, []int64{61})
	if err != nil {
		t.Fatal(err)
	}
	legacyizeMixedPullCheckpoint(&base)
	if upgraded, err := upgradeLegacyMixedPullEvidence(&base); err != nil || !upgraded {
		t.Fatalf("upgrade = %t, %v", upgraded, err)
	}
	if err := validateCheckpoint(base); err != nil {
		t.Fatalf("valid count-only checkpoint rejected: %v", err)
	}
	tests := []struct {
		name string
		want string
		edit func(*Corpus)
	}{
		{name: "forged reason", want: "contradictory provenance", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.EvidenceReason = "projected" }},
		{name: "forged mixed identity", want: "explicitly unavailable", edit: func(c *Corpus) { c.Receipt.Sources[0].ClassifiedPullIdentities = []CrossEndpointIdentity{} }},
		{name: "forged empty overlap", want: "must keep relation identity sets", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.Overlap = []CrossEndpointIdentity{} }},
		{name: "forged bound", want: "bounds contradict counts", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.SymmetricDifferenceUpperBound++ }},
		{name: "forged verdict", want: "verdict contradicts", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.Verdict = NonAtomicDeltaVerdictRejected }},
		{name: "projection basis", want: "contradictory provenance", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.IdentityBasis = "legacy_checkpoint_projection" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			corpus := cloneCorpus(base)
			tt.edit(&corpus)
			if err := validateCheckpoint(corpus); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateCheckpoint error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestLegacyCheckpointRecognitionIsExact(t *testing.T) {
	base, err := captureReconciliationFixture(t, []int64{71}, []int64{71})
	if err != nil {
		t.Fatal(err)
	}
	legacyizeMixedPullCheckpoint(&base)
	issues, _ := sourceByName(base.Receipt.Sources, "issues")
	if !isLegacyMixedPullCheckpoint(base.Receipt, issues) {
		t.Fatal("precise legacy checkpoint was not recognized")
	}
	tests := []struct {
		name string
		edit func(*Receipt, *SourceReceipt)
	}{
		{name: "failed receipt", edit: func(r *Receipt, _ *SourceReceipt) { r.Status = StatusFailed }},
		{name: "empty present identity list", edit: func(_ *Receipt, s *SourceReceipt) { s.ClassifiedPullIdentities = []CrossEndpointIdentity{} }},
		{name: "present checksum", edit: func(_ *Receipt, s *SourceReceipt) { s.ClassifiedPullChecksum = identityDigest(nil) }},
		{name: "zero classified count", edit: func(_ *Receipt, s *SourceReceipt) { s.ClassifiedPullCount = 0 }},
		{name: "count contradiction", edit: func(_ *Receipt, s *SourceReceipt) { s.FetchedCount++ }},
		{name: "typed evidence already present", edit: func(r *Receipt, _ *SourceReceipt) { r.NonAtomicDelta = &NonAtomicDeltaEvidence{} }},
		{name: "missing pulls", edit: func(r *Receipt, _ *SourceReceipt) { r.Sources = r.Sources[:1] }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt := cloneCorpus(base).Receipt
			candidate, _ := sourceByName(receipt.Sources, "issues")
			tt.edit(&receipt, &candidate)
			if isLegacyMixedPullCheckpoint(receipt, candidate) {
				t.Fatal("contradictory shape recognized as legacy")
			}
		})
	}
}

func TestLegacyMixedPullMigrationRejectsContradictoryMarkerBeforeRequests(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{45}, []int64{45})
	if err != nil {
		t.Fatal(err)
	}
	legacyizeMixedPullCheckpoint(&corpus)
	issues, _ := sourceByName(corpus.Receipt.Sources, "issues")
	issues.ClassifiedPullIdentities = []CrossEndpointIdentity{{ID: 45, Number: 45, NodeID: "PR_45"}}
	upsertSource(&corpus.Receipt.Sources, issues)
	requests := 0
	collector := NewCollector(&http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		requests++
		return nil, errors.New("unexpected request")
	})})
	collector.BaseURL = corpus.Receipt.APIBase
	_, err = collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoffTime(t, corpus.Receipt.Cutoff), Resume: &corpus})
	if err == nil || !strings.Contains(err.Error(), "classified pull identity checksum mismatch") {
		t.Fatalf("Capture error = %v, want contradictory marker failure", err)
	}
	if requests != 0 {
		t.Fatalf("contradictory resume made %d requests", requests)
	}
}

func TestLegacyMixedPullMigrationIsIdempotentForUpgradedCheckpoint(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{51}, []int64{51})
	if err != nil {
		t.Fatal(err)
	}
	want := cloneCorpus(corpus)
	for i := 0; i < 2; i++ {
		upgraded, err := upgradeLegacyMixedPullEvidence(&corpus)
		if err != nil || upgraded {
			t.Fatalf("upgrade current checkpoint = %t, %v", upgraded, err)
		}
	}
	if !reflect.DeepEqual(corpus, want) {
		t.Fatal("already-upgraded checkpoint changed")
	}
}

func legacyizeMixedPullCheckpoint(c *Corpus) {
	c.Receipt.Status = StatusPartial
	c.Receipt.CompletedAt = ""
	c.Receipt.NonAtomicDelta = nil
	for i := range c.Receipt.Sources {
		if c.Receipt.Sources[i].Name == "issues" {
			c.Receipt.Sources[i].ClassifiedPullIdentities = nil
			c.Receipt.Sources[i].ClassifiedPullChecksum = ""
		}
	}
}

func cutoffTime(t *testing.T, value string) time.Time {
	t.Helper()
	cutoff, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		t.Fatal(err)
	}
	return cutoff
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func captureReconciliationFixture(t *testing.T, mixed, dedicated []int64) (Corpus, error) {
	t.Helper()
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"delta-revision"}`)
		case "/repos/acme/widget/issues":
			writeRows(w, mixed, true)
		case "/repos/acme/widget/pulls":
			writeRows(w, dedicated, false)
		case "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.Now = func() time.Time { return cutoff }
	return collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
}

func writeRows(w http.ResponseWriter, ids []int64, mixed bool) {
	rows := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		row := map[string]any{
			"id": id, "node_id": fmt.Sprintf("PR_%d", id), "number": int(id),
			"title": fmt.Sprintf("pull %d", id), "created_at": "2026-08-25T00:00:00Z",
		}
		if mixed {
			row["pull_request"] = map[string]any{}
		}
		rows = append(rows, row)
	}
	_ = json.NewEncoder(w).Encode(rows)
}

func assertIdentityIDs(t *testing.T, got []CrossEndpointIdentity, want ...int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("identity count = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i] {
			t.Fatalf("identity[%d] = %d, want %d", i, got[i].ID, want[i])
		}
	}
}

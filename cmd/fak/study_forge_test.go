package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

func TestStudyForgeCaptureCLIResumesWithinSourceCheckpoint(t *testing.T) {
	var (
		mu        sync.Mutex
		page1Hits int
		failPage2 = true
		server    *httptest.Server
	)
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"cli-revision"}`)
		case "/repos/acme/widget/issues":
			if r.URL.Query().Get("page") == "2" {
				mu.Lock()
				fail := failPage2
				mu.Unlock()
				if fail {
					http.Error(w, "stop after checkpoint", http.StatusInternalServerError)
					return
				}
				fmt.Fprint(w, `[{"id":2,"created_at":"2026-08-25T00:00:00Z"}]`)
				return
			}
			mu.Lock()
			page1Hits++
			mu.Unlock()
			w.Header().Set("Link", `<`+server.URL+`/repos/acme/widget/issues?page=2>; rel="next"`)
			fmt.Fprint(w, `[{"id":1,"created_at":"2026-08-25T00:00:00Z"}]`)
		case "/repos/acme/widget/pulls", "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	out := filepath.Join(t.TempDir(), "corpus.json")
	args := []string{
		"capture", "--repository", "acme/widget", "--cutoff", "2026-08-26T12:00:00Z",
		"--out", out, "--base-url", server.URL, "--retries", "0",
	}
	var stdout, stderr bytes.Buffer
	if code := runStudyForge(&stdout, &stderr, args); code != 1 {
		t.Fatalf("first capture exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	partial, err := studyforge.Read(out)
	if err != nil {
		t.Fatal(err)
	}
	if partial.Receipt.Status != studyforge.StatusPartial || len(partial.Receipt.Sources[0].Pages) != 1 {
		t.Fatalf("partial checkpoint = %+v", partial.Receipt)
	}

	mu.Lock()
	failPage2 = false
	mu.Unlock()
	stdout.Reset()
	stderr.Reset()
	if code := runStudyForge(&stdout, &stderr, append(args, "--resume")); code != 0 {
		t.Fatalf("resume exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	mu.Lock()
	gotPage1Hits := page1Hits
	mu.Unlock()
	if gotPage1Hits != 1 {
		t.Fatalf("resume refetched page 1: hits=%d", gotPage1Hits)
	}
	completed, err := studyforge.Read(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := studyforge.Validate(completed); err != nil {
		t.Fatal(err)
	}
}

func TestStudyForgeCaptureCLIResumesPreMetricLegacyCountOnlyCheckpoint(t *testing.T) {
	const (
		mixedCount     = 35528
		dedicatedCount = 35551
	)
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var (
		mu                     sync.Mutex
		calls                  = map[string]int{}
		persistedBeforeNext    bool
		persistedBeforeNextErr error
		out                    string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.Path]++
		mu.Unlock()
		if r.URL.Path == "/repos/acme/widget/releases" {
			checkpoint, err := studyforge.Read(out)
			if err == nil {
				err = validateMigratedLegacyCountOnlyCheckpoint(checkpoint)
			}
			mu.Lock()
			persistedBeforeNext = true
			persistedBeforeNextErr = err
			mu.Unlock()
		}
		switch r.URL.Path {
		case "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.Error(w, "completed source was refetched", http.StatusConflict)
		}
	}))
	defer server.Close()

	out = filepath.Join(t.TempDir(), "legacy-count-only.json")
	legacyJSON := writePreMetricLegacyCountOnlyCheckpoint(t, out, server.URL, cutoff, mixedCount, dedicatedCount)
	if bytes.Contains(legacyJSON, []byte(`"metric"`)) || bytes.Contains(legacyJSON, []byte(`"observed_endpoint_cardinality_delta"`)) {
		t.Fatal("pre-metric fixture unexpectedly contains current metric evidence")
	}
	if _, err := studyforge.Read(out); err == nil {
		t.Fatal("ordinary Read accepted the resume-only pre-metric checkpoint")
	}

	invalidPath := filepath.Join(filepath.Dir(out), "invalid-legacy-count-only.json")
	var invalid studyforge.Corpus
	if err := json.Unmarshal(legacyJSON, &invalid); err != nil {
		t.Fatal(err)
	}
	invalid.Receipt.NonAtomicDelta.SymmetricDifferenceUpperBound++
	invalidJSON, err := json.MarshalIndent(invalid, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(invalidPath, append(invalidJSON, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	args := []string{
		"capture", "--repository", "acme/widget", "--cutoff", cutoff.Format(time.RFC3339),
		"--base-url", server.URL, "--retries", "0", "--resume",
	}
	var stdout, stderr bytes.Buffer
	if code := runStudyForge(&stdout, &stderr, append(args, "--out", invalidPath)); code != 1 {
		t.Fatalf("invalid resume exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	mu.Lock()
	requestsAfterInvalid := len(calls)
	mu.Unlock()
	if requestsAfterInvalid != 0 {
		t.Fatalf("invalid resume reached network: calls=%v", calls)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runStudyForge(&stdout, &stderr, append(args, "--out", out)); code != 0 {
		t.Fatalf("legacy resume exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	mu.Lock()
	gotPersistedBeforeNext := persistedBeforeNext
	gotPersistedBeforeNextErr := persistedBeforeNextErr
	issuesCalls := calls["/repos/acme/widget/issues"]
	pullsCalls := calls["/repos/acme/widget/pulls"]
	mu.Unlock()
	if !gotPersistedBeforeNext || gotPersistedBeforeNextErr != nil {
		t.Fatalf("first resumed checkpoint was not strict current shape before the next source request: observed=%t err=%v", gotPersistedBeforeNext, gotPersistedBeforeNextErr)
	}
	if issuesCalls != 0 || pullsCalls != 0 {
		t.Fatalf("resume refetched completed pages: issues=%d pulls=%d", issuesCalls, pullsCalls)
	}

	completed, err := studyforge.Read(out)
	if err != nil {
		t.Fatalf("strict Read rejected migrated checkpoint: %v", err)
	}
	if err := studyforge.Validate(completed); err != nil {
		t.Fatalf("migrated completed checkpoint did not validate: %v", err)
	}
	if err := validateMigratedLegacyCountOnlyCheckpoint(completed); err != nil {
		t.Fatal(err)
	}
}

func writePreMetricLegacyCountOnlyCheckpoint(t *testing.T, path, baseURL string, cutoff time.Time, mixedCount, dedicatedCount int) []byte {
	t.Helper()
	fetchedAt := cutoff.Format(time.RFC3339Nano)
	repositoryURL := baseURL + "/repos/acme/widget"
	issuesEndpoint := repositoryURL + "/issues?state=all&sort=created&direction=asc&per_page=100&page=1"
	pullsEndpoint := repositoryURL + "/pulls?state=all&sort=created&direction=asc&per_page=100&page=1"
	digest := "sha256:" + strings.Repeat("0", 64)
	identitySetsAvailable := false
	observedDelta := dedicatedCount - mixedCount
	if observedDelta < 0 {
		observedDelta = -observedDelta
	}
	records := make([]studyforge.Record, dedicatedCount)
	for i := range records {
		records[i] = studyforge.Record{Source: "pulls", Kind: "pull", ID: int64(i + 1), Number: i + 1}
	}
	checkpoint := studyforge.Corpus{
		Schema: studyforge.CorpusSchema,
		Receipt: studyforge.Receipt{
			Schema:     studyforge.ReceiptSchema,
			Repository: "acme/widget",
			Revision:   "pre-metric-revision",
			Cutoff:     fetchedAt,
			APIBase:    baseURL,
			StartedAt:  fetchedAt,
			Status:     studyforge.StatusPartial,
			Sources: []studyforge.SourceReceipt{
				{
					Name: "issues", Endpoint: issuesEndpoint, Status: studyforge.StatusComplete,
					CrawlStartedAt: fetchedAt, CrawlEndedAt: fetchedAt,
					Pages:               []studyforge.PageReceipt{{Number: 1, URL: issuesEndpoint, ItemCount: mixedCount, Checksum: digest, FetchedAt: fetchedAt, StatusCode: http.StatusOK}},
					FetchedCount:        mixedCount,
					ClassifiedPullCount: mixedCount,
				},
				{
					Name: "pulls", Endpoint: pullsEndpoint, Status: studyforge.StatusComplete,
					CrawlStartedAt: fetchedAt, CrawlEndedAt: fetchedAt,
					Pages:           []studyforge.PageReceipt{{Number: 1, URL: pullsEndpoint, ItemCount: dedicatedCount, Checksum: digest, FetchedAt: fetchedAt, StatusCode: http.StatusOK}},
					FetchedCount:    dedicatedCount,
					NormalizedCount: dedicatedCount,
					UniqueCount:     dedicatedCount,
				},
			},
			NonAtomicDelta: &studyforge.NonAtomicDeltaEvidence{
				Type:                             studyforge.NonAtomicDeltaType,
				MixedSource:                      "issues",
				DedicatedSource:                  "pulls",
				EvidenceMode:                     studyforge.NonAtomicDeltaEvidenceModeLegacyCountOnly,
				EvidenceReason:                   "legacy_mixed_pull_identities_not_retained",
				IdentitySetsAvailable:            &identitySetsAvailable,
				IdentityBasis:                    "legacy_checkpoint_counts",
				MixedEvidence:                    studyforge.CrossEndpointEvidenceExactCountOnly,
				DedicatedEvidence:                studyforge.CrossEndpointEvidenceExactIdentities,
				RelationEvidence:                 studyforge.CrossEndpointEvidenceUnavailable,
				MixedCrawl:                       studyforge.CrawlWindow{StartedAt: fetchedAt, EndedAt: fetchedAt},
				DedicatedCrawl:                   studyforge.CrawlWindow{StartedAt: fetchedAt, EndedAt: fetchedAt},
				MixedCount:                       mixedCount,
				DedicatedCount:                   dedicatedCount,
				ObservedEndpointCardinalityDelta: &observedDelta,
				SymmetricDifferenceLowerBound:    observedDelta,
				SymmetricDifferenceUpperBound:    mixedCount + dedicatedCount,
				Policy: studyforge.NonAtomicDeltaPolicy{
					Type: studyforge.NonAtomicDeltaPolicyType, Metric: studyforge.NonAtomicDeltaPolicyMetricEndpointCardinalityDelta,
					MaxOnlyInMixed: studyforge.DefaultNonAtomicDeltaLimit, MaxOnlyInDedicated: studyforge.DefaultNonAtomicDeltaLimit, MaxTotal: studyforge.DefaultNonAtomicDeltaLimit,
				},
				Verdict:  studyforge.NonAtomicDeltaVerdictAccepted,
				Accepted: true,
			},
			API: []studyforge.APIReceipt{
				{Purpose: "repository", URL: repositoryURL, FetchedAt: fetchedAt, StatusCode: http.StatusOK, Checksum: digest},
				{Purpose: "revision", URL: repositoryURL + "/commits/main", FetchedAt: fetchedAt, StatusCode: http.StatusOK, Checksum: digest},
			},
		},
		Records: records,
	}
	if err := studyforge.Write(path, checkpoint); err != nil {
		t.Fatalf("write strict fixture seed: %v", err)
	}
	checkpoint, err := studyforge.Read(path)
	if err != nil {
		t.Fatalf("read strict fixture seed: %v", err)
	}
	checkpoint.Receipt.NonAtomicDelta.Policy.Metric = ""
	checkpoint.Receipt.NonAtomicDelta.ObservedEndpointCardinalityDelta = nil
	checkpoint.Receipt.NonAtomicDelta.Verdict = studyforge.NonAtomicDeltaVerdictCompatibleUnproven
	checkpoint.Receipt.NonAtomicDelta.Accepted = false
	legacyJSON, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	legacyJSON = append(legacyJSON, '\n')
	if err := os.WriteFile(path, legacyJSON, 0600); err != nil {
		t.Fatal(err)
	}
	return legacyJSON
}

func validateMigratedLegacyCountOnlyCheckpoint(checkpoint studyforge.Corpus) error {
	delta := checkpoint.Receipt.NonAtomicDelta
	if delta == nil {
		return fmt.Errorf("non_atomic_delta is missing")
	}
	if delta.EvidenceMode != studyforge.NonAtomicDeltaEvidenceModeLegacyCountOnly || delta.EvidenceReason != "legacy_mixed_pull_identities_not_retained" || delta.IdentitySetsAvailable == nil || *delta.IdentitySetsAvailable || delta.IdentityBasis != "legacy_checkpoint_counts" {
		return fmt.Errorf("legacy provenance was not preserved: %+v", delta)
	}
	if delta.MixedCount != 35528 || delta.DedicatedCount != 35551 || delta.SymmetricDifferenceLowerBound != 23 || delta.SymmetricDifferenceUpperBound != 71079 {
		return fmt.Errorf("legacy counts or bounds changed: %+v", delta)
	}
	if delta.Overlap != nil || delta.OnlyInMixed != nil || delta.OnlyInDedicated != nil || delta.OverlapCount != nil || delta.OnlyInMixedCount != nil || delta.OnlyInDedicatedCount != nil {
		return fmt.Errorf("migration fabricated unavailable identity relations: %+v", delta)
	}
	if delta.Policy.Type != studyforge.NonAtomicDeltaPolicyType || delta.Policy.Metric != studyforge.NonAtomicDeltaPolicyMetricEndpointCardinalityDelta || delta.Policy.MaxOnlyInMixed != studyforge.DefaultNonAtomicDeltaLimit || delta.Policy.MaxOnlyInDedicated != studyforge.DefaultNonAtomicDeltaLimit || delta.Policy.MaxTotal != studyforge.DefaultNonAtomicDeltaLimit {
		return fmt.Errorf("migration policy = %+v", delta.Policy)
	}
	if delta.ObservedEndpointCardinalityDelta == nil || *delta.ObservedEndpointCardinalityDelta != 23 || delta.Verdict != studyforge.NonAtomicDeltaVerdictAccepted || !delta.Accepted {
		return fmt.Errorf("migration verdict evidence = %+v", delta)
	}
	return nil
}

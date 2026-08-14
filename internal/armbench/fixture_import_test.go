package armbench

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestPinnedFixtureDeclarationsMatchIssue6677(t *testing.T) {
	tests := []struct {
		suite FixtureSuite
		sha   string
		want  []string
	}{
		{
			suite: FixtureSuiteCaveman,
			sha:   CavemanFixtureSHA,
			want: []string{
				"benchmarks/run.py",
				"benchmarks/prompts.json",
				"skills/caveman/SKILL.md",
				"benchmarks/requirements.txt",
				"docs/HONEST-NUMBERS.md",
				"evals/README.md",
				"tests/test_benchmark_contract.py",
				"LICENSE",
				"LICENSING.md",
			},
		},
		{
			suite: FixtureSuitePonytail,
			sha:   PonytailFixtureSHA,
			want: []string{
				"benchmarks/promptfooconfig.yaml",
				"benchmarks/promptfooconfig.gpt.yaml",
				"benchmarks/promptfooconfig.gpt-newest.yaml",
				"benchmarks/promptfooconfig.gemini.yaml",
				"benchmarks/prompts.json",
				"benchmarks/arms/baseline.js",
				"benchmarks/arms/caveman.js",
				"benchmarks/arms/ponytail.js",
				"benchmarks/arms/caveman-SKILL.md",
				"skills/ponytail/SKILL.md",
				"benchmarks/loc.js",
				"benchmarks/behavior.js",
				"benchmarks/behavior.yaml",
				"benchmarks/correctness.js",
				"benchmarks/robustness-audit.js",
				"benchmarks/agentic/tasks.py",
				"benchmarks/agentic/judge.py",
				"LICENSE",
			},
		},
	}
	for _, tc := range tests {
		t.Run(string(tc.suite), func(t *testing.T) {
			declarations := PinnedFixtureDeclarations(tc.suite)
			if err := validateDeclarations(tc.suite, declarations); err != nil {
				t.Fatalf("declared fixture set is invalid: %v", err)
			}
			got := make([]string, 0, len(declarations))
			for _, declaration := range declarations {
				got = append(got, declaration.Path)
				if declaration.SHA != tc.sha {
					t.Fatalf("%s pins %s, want %s", declaration.Path, declaration.SHA, tc.sha)
				}
				if declaration.Normalization != normalizationNone {
					t.Fatalf("%s normalization=%q", declaration.Path, declaration.Normalization)
				}
				if !validSHA256(declaration.ExpectedHash) {
					t.Fatalf("%s hash=%q", declaration.Path, declaration.ExpectedHash)
				}
			}
			sort.Strings(got)
			sort.Strings(tc.want)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("paths differ\n got: %v\nwant: %v", got, tc.want)
			}
		})
	}
	for _, declaration := range PinnedFixtureDeclarations(FixtureSuiteCaveman) {
		if declaration.ReviewToken != CavemanLicenseReviewToken {
			t.Fatalf("%s does not require the revision-bound Caveman review token", declaration.Path)
		}
	}
	for _, declaration := range PinnedFixtureDeclarations(FixtureSuitePonytail) {
		if declaration.ReviewToken != "" {
			t.Fatalf("%s unexpectedly requires explicit review despite the asserted root MIT license", declaration.Path)
		}
	}
}

func TestFixtureImporterLocalServerProducesRunnableArmbenchInput(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	server, requested := fixtureServer(t, bodies)
	defer server.Close()

	workspace := t.TempDir()
	store := t.TempDir()
	opts := ImportOptions{
		StoreRoot: store, WorkspaceRoot: workspace, BaseURL: server.URL,
		FakVersion: "test-6677", Client: server.Client(),
		LicenseReviews: []string{"review-caveman-test"},
	}
	result, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if result.SourceCount != len(declarations) || result.InputID == "" || result.ManifestIdentity == "" {
		t.Fatalf("incomplete result: %+v", result)
	}
	gotRequests := requested()
	wantRequests := make([]string, 0, len(declarations))
	for _, declaration := range declarations {
		wantRequests = append(wantRequests, "/"+declaration.Repo+"/"+declaration.SHA+"/"+declaration.Path)
	}
	sort.Strings(gotRequests)
	sort.Strings(wantRequests)
	if !reflect.DeepEqual(gotRequests, wantRequests) {
		t.Fatalf("fetch set differs\n got: %v\nwant: %v", gotRequests, wantRequests)
	}

	manifestBlob, err := os.ReadFile(filepath.Join(store, filepath.FromSlash(result.ManifestPath)))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := UnmarshalManifest(manifestBlob)
	if err != nil {
		t.Fatalf("manifest is not armbench-consumable: %v", err)
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("manifest validation: %v", err)
	}
	if manifest.Identity() != result.ManifestIdentity {
		t.Fatalf("identity %s, result %s", manifest.Identity(), result.ManifestIdentity)
	}
	for _, source := range manifest.Sources {
		if source.URL == "" || source.License != "MIT" || source.LicenseBoundary == "" ||
			source.LicenseBoundaryHash == "" || source.LicenseReview != "explicit:review-caveman-test" ||
			source.RetrievedAt != "2026-08-13" || source.Normalization != normalizationNone ||
			source.LocalPath == "" {
			t.Fatalf("source omitted issue-required provenance: %+v", source)
		}
		materialized, err := os.ReadFile(filepath.Join(store, filepath.FromSlash(result.InputDir), filepath.FromSlash(source.LocalPath)))
		if err != nil {
			t.Fatal(err)
		}
		if digestBytes(materialized) != source.ContentHash {
			t.Fatalf("%s materialized hash drifted", source.Name)
		}
	}

	corpusBlob, err := os.ReadFile(filepath.Join(store, filepath.FromSlash(result.CorpusPath)))
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := UnmarshalCorpus(corpusBlob)
	if err != nil {
		t.Fatalf("corpus is not armbench-consumable: %v", err)
	}
	if err := ValidateCorpus(manifest, corpus); err != nil {
		t.Fatalf("corpus validation: %v", err)
	}
	if _, err := Execute(context.Background(), manifest, corpus.Tasks, &FakeProvider{}, &FakeGrader{}, Options{}); err != nil {
		t.Fatalf("emitted input did not execute through armbench: %v", err)
	}

	// An unchanged re-import reuses the same content address.
	again, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	if err != nil {
		t.Fatalf("re-import: %v", err)
	}
	if again.InputID != result.InputID || again.InputDir != result.InputDir {
		t.Fatalf("unchanged import moved content address: first=%+v second=%+v", result, again)
	}
}

func TestFixtureImporterFailsClosedOnHashMismatch(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	bodies[pathFor(declarations[0])] = []byte("different bytes")
	server, _ := fixtureServer(t, bodies)
	defer server.Close()
	_, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, importTestOptions(t, server), "2026-08-13")
	requireReason(t, err, ReasonFixtureHashMismatch)
}

func TestFixtureImporterFailsClosedWhenPathMoved(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	delete(bodies, pathFor(declarations[0]))
	server, _ := fixtureServer(t, bodies)
	defer server.Close()
	_, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, importTestOptions(t, server), "2026-08-13")
	requireReason(t, err, ReasonFixturePathMoved)
}

func TestFixtureImporterFailsClosedWhenPinnedPathRedirects(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	first := pathFor(declarations[0])
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == first {
			http.Redirect(w, r, "/moved", http.StatusFound)
			return
		}
		if r.URL.Path == "/moved" {
			_, _ = w.Write(bodies[first])
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()
	_, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, importTestOptions(t, server), "2026-08-13")
	requireReason(t, err, ReasonFixturePathMoved)
}

func TestFixtureImporterFailsClosedOnMissingLicenseMetadataBeforeNetwork(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	declarations[0].LicenseBoundary = ""
	server, requested := fixtureServer(t, bodies)
	defer server.Close()
	_, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, importTestOptions(t, server), "2026-08-13")
	requireReason(t, err, ReasonFixtureLicenseMissing)
	if got := requested(); len(got) != 0 {
		t.Fatalf("network was used before the license gate: %v", got)
	}
}

func TestFixtureImporterRequiresExplicitCavemanLicenseReviewBeforeNetwork(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	server, requested := fixtureServer(t, bodies)
	defer server.Close()
	opts := importTestOptions(t, server)
	opts.LicenseReviews = nil
	_, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	requireReason(t, err, ReasonFixtureLicenseReview)
	if got := requested(); len(got) != 0 {
		t.Fatalf("network was used before explicit review: %v", got)
	}
}

func TestFixtureImporterRefusesUnknownLicenseReviewTokenBeforeNetwork(t *testing.T) {
	server, requested := fixtureServer(t, nil)
	defer server.Close()
	opts := ImportOptions{
		StoreRoot: t.TempDir(), WorkspaceRoot: t.TempDir(), Client: server.Client(), BaseURL: server.URL,
		LicenseReviews: []string{"unbound-license-conclusion"},
	}
	_, err := ImportFixtures(context.Background(), FixtureSuitePonytail, opts)
	requireReason(t, err, ReasonFixtureLicenseReview)
	if got := requested(); len(got) != 0 {
		t.Fatalf("network was reached before rejecting unknown review token: %v", got)
	}
}

func TestFixtureImporterFailsClosedOnDirtyLocalMutation(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	server, _ := fixtureServer(t, bodies)
	defer server.Close()
	opts := importTestOptions(t, server)
	result, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(opts.StoreRoot, filepath.FromSlash(result.InputDir), filepath.FromSlash(fixtureCopyPath(declarations[0])))
	if err := os.Chmod(path, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("local edit"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	requireReason(t, err, ReasonFixtureLocalMutation)
}

func TestFixtureImporterFailsClosedOnWritableLocalCopy(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	server, _ := fixtureServer(t, bodies)
	defer server.Close()
	opts := importTestOptions(t, server)
	result, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(opts.StoreRoot, filepath.FromSlash(result.InputDir), "manifest.json")
	if err := os.Chmod(path, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	requireReason(t, err, ReasonFixtureLocalMutation)
}

func TestFixtureImporterFailsClosedOnUnexpectedLocalDirectory(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	server, _ := fixtureServer(t, bodies)
	defer server.Close()
	opts := importTestOptions(t, server)
	result, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	if err != nil {
		t.Fatal(err)
	}
	extra := filepath.Join(opts.StoreRoot, filepath.FromSlash(result.InputDir), "locally-added-empty-directory")
	if err := os.Mkdir(extra, 0o755); err != nil {
		t.Fatal(err)
	}
	_, err = importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, opts, "2026-08-13")
	requireReason(t, err, ReasonFixtureLocalMutation)
}

func TestFixtureImporterRefusesRepositoryLocalStore(t *testing.T) {
	workspace := t.TempDir()
	err := validateStoreRoot(filepath.Join(workspace, "_scratch", "fixtures"), workspace)
	requireReason(t, err, ReasonFixtureStoreInsideRepo)
}

func TestFixtureImporterRefusesExternalSymlinkIntoRepository(t *testing.T) {
	workspace := t.TempDir()
	target := filepath.Join(workspace, "_scratch", "fixtures")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "external-looking-store")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	requireReason(t, validateStoreRoot(link, workspace), ReasonFixtureStoreInsideRepo)
}

func TestFixtureImporterRefusesCavemanRestrictedRuntimePath(t *testing.T) {
	declarations, _ := testCavemanDeclarations()
	declarations = append(declarations, FixtureDeclaration{
		Name: "restricted", Suite: FixtureSuiteCaveman, Repo: "JuliusBrussee/caveman",
		SHA: CavemanFixtureSHA, Path: "engine/compressors/text.go", ExpectedHash: digestBytes([]byte("x")),
		Role: "runtime", License: "BSL-1.1", LicenseBoundary: "LICENSE.BSL",
		LicenseBoundaryHash: digestBytes([]byte("license")), ReviewToken: "review-caveman-test",
		Normalization: normalizationNone,
	})
	requireReason(t, validateDeclarations(FixtureSuiteCaveman, declarations), ReasonFixtureRestrictedPath)
}

func TestFixtureImporterRefusesLicenseBoundaryMetadataMismatch(t *testing.T) {
	declarations, bodies := testCavemanDeclarations()
	for i := range declarations {
		declarations[i].LicenseBoundaryHash = digestBytes([]byte("not-the-fetched-license-boundary"))
	}
	server, _ := fixtureServer(t, bodies)
	defer server.Close()
	_, err := importFixtureSuite(context.Background(), FixtureSuiteCaveman, declarations, importTestOptions(t, server), "2026-08-13")
	requireReason(t, err, ReasonFixtureLicenseMissing)
}

func TestMarshalFixtureImportReportStable(t *testing.T) {
	report := &FixtureImportReport{
		Schema: FixtureImportSchema, RetrievedAt: "2026-08-13",
		Store:   "content-addressed external store (paths relative to store root)",
		Results: []FixtureImportResult{{Suite: FixtureSuiteCaveman, InputID: "sha256:" + strings.Repeat("a", 64)}},
	}
	a, err := MarshalFixtureImportReport(report)
	if err != nil {
		t.Fatal(err)
	}
	var decoded FixtureImportReport
	if err := json.Unmarshal(a, &decoded); err != nil {
		t.Fatal(err)
	}
	b, err := MarshalFixtureImportReport(&decoded)
	if err != nil {
		t.Fatal(err)
	}
	if string(a) != string(b) {
		t.Fatalf("report is not stable\n%s\n%s", a, b)
	}
}

func TestCommittedFixtureImportWitnessMatchesPinnedDeclarations(t *testing.T) {
	blob, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "armbench-import-2026-08-14.json"))
	if err != nil {
		t.Fatal(err)
	}
	var report FixtureImportReport
	if err := json.Unmarshal(blob, &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != FixtureImportSchema || report.RetrievedAt != "2026-08-14" {
		t.Fatalf("unexpected witness header: %+v", report)
	}
	if len(report.Results) != 2 {
		t.Fatalf("witness has %d results, want Caveman + Ponytail", len(report.Results))
	}
	for i, suite := range []FixtureSuite{FixtureSuiteCaveman, FixtureSuitePonytail} {
		declarations := PinnedFixtureDeclarations(suite)
		fetched := make([]fetchedFixture, len(declarations))
		for j, declaration := range declarations {
			fetched[j].Declaration = declaration
		}
		result := report.Results[i]
		if result.Suite != suite || result.SourceCount != len(declarations) {
			t.Fatalf("result %d does not cover %s declarations: %+v", i, suite, result)
		}
		if result.SourceSetHash != declarationSetHash(fetched) {
			t.Fatalf("%s witness source set is stale: got %s", suite, result.SourceSetHash)
		}
		if !validSHA256(result.InputID) || !validSHA256(result.ManifestIdentity) {
			t.Fatalf("%s witness lacks content addresses: %+v", suite, result)
		}
	}
}

func testCavemanDeclarations() ([]FixtureDeclaration, map[string][]byte) {
	const (
		repo  = "JuliusBrussee/caveman"
		sha   = CavemanFixtureSHA
		token = "review-caveman-test"
	)
	licenseBody := []byte("MIT License\n")
	licensingBody := []byte("skills/ and benchmarks/ are MIT.\n")
	boundaryHash := digestParts(licenseBody, licensingBody)
	bodies := map[string][]byte{}
	add := func(name, path, role string, body []byte) FixtureDeclaration {
		declaration := FixtureDeclaration{
			Name: name, Suite: FixtureSuiteCaveman, Repo: repo, SHA: sha, Path: path,
			ExpectedHash: digestBytes(body), Role: role, License: "MIT",
			LicenseBoundary: "LICENSE + LICENSING.md", LicenseBoundaryHash: boundaryHash,
			ReviewToken: token, Normalization: normalizationNone,
		}
		bodies[pathFor(declaration)] = body
		return declaration
	}
	declarations := []FixtureDeclaration{
		add("caveman-benchmark-runner", "benchmarks/run.py", "harness", []byte("NORMAL_SYSTEM = \"You are a helpful assistant.\"\n")),
		add("caveman-prompts", "benchmarks/prompts.json", "corpus", []byte(`{"version":1,"prompts":[{"id":"one","category":"test","prompt":"first task"},{"id":"two","category":"test","prompt":"second task"}]}`)),
		add("caveman-active-skill", "skills/caveman/SKILL.md", "system_prompt", []byte("Respond tersely.\n")),
		add("caveman-honest-numbers", "docs/HONEST-NUMBERS.md", "eval_metadata", []byte("Manual quality review required.\n")),
		add("caveman-license", "LICENSE", "license", licenseBody),
		add("caveman-licensing-map", "LICENSING.md", "license", licensingBody),
	}
	return declarations, bodies
}

func pathFor(declaration FixtureDeclaration) string {
	return "/" + declaration.Repo + "/" + declaration.SHA + "/" + declaration.Path
}

func fixtureServer(t *testing.T, bodies map[string][]byte) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var requested []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requested = append(requested, r.URL.Path)
		mu.Unlock()
		body, ok := bodies[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(body)
	}))
	return server, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), requested...)
	}
}

func importTestOptions(t *testing.T, server *httptest.Server) ImportOptions {
	t.Helper()
	return ImportOptions{
		StoreRoot: t.TempDir(), WorkspaceRoot: t.TempDir(), BaseURL: server.URL,
		FakVersion: "test", Client: server.Client(),
		Now:            func() time.Time { return time.Date(2026, 8, 13, 12, 0, 0, 0, time.UTC) },
		LicenseReviews: []string{"review-caveman-test"},
	}
}

package armbench

// fixture_import.go is the immutable-input importer for benchmark epic #6674
// and issue #6677. It fetches only an audited, exact-path allowlist at the two
// issue-pinned commits; checks every response against a committed SHA-256;
// records the applicable license boundary and any explicit review; stores the
// byte-exact bodies in an out-of-repository content-addressed store; and emits
// an ordinary fak.armbench.manifest/1 + corpus pair.
//
// The importer is intentionally narrower than a git clone. A clone can quietly
// pull new paths, submodules, generated artifacts, or mixed-license runtime
// code. This importer has no "path" argument: adding one upstream file is a
// source change whose hash and license boundary must be reviewed here first.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const (
	// FixtureImportSchema tags the machine-readable command result.
	FixtureImportSchema = "fak.armbench.fixture-import/1"

	// The exact revisions named by #6677.
	CavemanFixtureSHA  = "c72984e4392c7a154e55c11dbf445f01ce5c35d4"
	PonytailFixtureSHA = "2ed6c52c9d7e5e56942508591085fd45dea277d3"

	// CavemanLicenseReviewToken is deliberately long and revision-bound. The
	// Caveman repository uses a mixed MIT/BSL boundary and GitHub reports the
	// repository license as NOASSERTION, so importing its benchmark surfaces
	// requires the operator to state the reviewed revision and MIT conclusion.
	CavemanLicenseReviewToken = "JuliusBrussee/caveman@" + CavemanFixtureSHA + "=MIT"

	defaultFixtureBaseURL = "https://raw.githubusercontent.com"
	maxFixtureBytes       = 8 << 20

	normalizationNone = "none; byte-exact upstream response"
)

// FixtureSuite is the closed importer suite vocabulary.
type FixtureSuite string

const (
	FixtureSuiteCaveman  FixtureSuite = "caveman"
	FixtureSuitePonytail FixtureSuite = "ponytail"
	FixtureSuiteAll      FixtureSuite = "all"
)

// FixtureDeclaration is one reviewed upstream path. ExpectedHash is over the
// raw response body, before any newline or text normalization (there is none).
type FixtureDeclaration struct {
	Name                string
	Suite               FixtureSuite
	Repo                string
	SHA                 string
	Path                string
	ExpectedHash        string
	Role                string
	License             string
	LicenseBoundary     string
	LicenseBoundaryHash string
	ReviewToken         string
	Normalization       string
}

// ImportOptions contains the impure edges. Tests inject an httptest server,
// clock, and temp roots, so CI never needs the network.
type ImportOptions struct {
	StoreRoot      string
	WorkspaceRoot  string
	BaseURL        string
	FakVersion     string
	Client         *http.Client
	Now            func() time.Time
	LicenseReviews []string
}

// LicenseReview records how one repository's license boundary was admitted.
type LicenseReview struct {
	Repo     string `json:"repo"`
	SHA      string `json:"sha"`
	License  string `json:"license"`
	Boundary string `json:"boundary"`
	Status   string `json:"status"`
}

// FixtureImportResult is one suite's content-addressed armbench input. Paths
// are relative to StoreRoot so committed proof output contains no host path.
type FixtureImportResult struct {
	Suite            FixtureSuite    `json:"suite"`
	InputID          string          `json:"input_id"`
	InputDir         string          `json:"input_dir"`
	ManifestPath     string          `json:"manifest_path"`
	CorpusPath       string          `json:"corpus_path"`
	ManifestIdentity string          `json:"manifest_identity"`
	SourceSetHash    string          `json:"source_set_hash"`
	SourceCount      int             `json:"source_count"`
	SourceBytes      int64           `json:"source_bytes"`
	LicenseReviews   []LicenseReview `json:"license_reviews"`
}

// FixtureImportReport is the stable JSON shape emitted by the command.
type FixtureImportReport struct {
	Schema      string                `json:"schema"`
	RetrievedAt string                `json:"retrieved_at"`
	Store       string                `json:"store"`
	Results     []FixtureImportResult `json:"results"`
}

// Additional importer-specific refusal reasons. They share RefusalError with
// the runner so the CLI retains one exit-code/refusal-token contract.
const (
	ReasonFixtureHashMismatch       = "FIXTURE_HASH_MISMATCH"
	ReasonFixturePathMoved          = "FIXTURE_PATH_MOVED"
	ReasonFixtureFetchFailed        = "FIXTURE_FETCH_FAILED"
	ReasonFixtureLicenseMissing     = "FIXTURE_LICENSE_METADATA_MISSING"
	ReasonFixtureLicenseReview      = "FIXTURE_LICENSE_REVIEW_REQUIRED"
	ReasonFixtureLocalMutation      = "FIXTURE_LOCAL_MUTATION"
	ReasonFixtureStoreInsideRepo    = "FIXTURE_STORE_INSIDE_REPO"
	ReasonFixtureRestrictedPath     = "FIXTURE_RESTRICTED_PATH"
	ReasonFixtureDeclarationInvalid = "FIXTURE_DECLARATION_INVALID"
)

const (
	cavemanLicenseBoundary      = "LICENSE + LICENSING.md"
	cavemanLicenseBoundaryHash  = "sha256:695deeb180b5a1e28a4eafd822d6a86b8673a642eee5e2865e1a3cc6cf43d3df"
	ponytailLicenseBoundary     = "LICENSE"
	ponytailLicenseBoundaryHash = "sha256:df8847f0cfdbc2f8d3b5a322e9fc8c6f3f411729c00e531d267fa5f78da51ae1"
)

// PinnedFixtureDeclarations returns a copy of the audited source allowlist.
func PinnedFixtureDeclarations(suite FixtureSuite) []FixtureDeclaration {
	var in []FixtureDeclaration
	switch suite {
	case FixtureSuiteCaveman:
		in = cavemanFixtureDeclarations()
	case FixtureSuitePonytail:
		in = ponytailFixtureDeclarations()
	case FixtureSuiteAll:
		in = append(cavemanFixtureDeclarations(), ponytailFixtureDeclarations()...)
	default:
		return nil
	}
	return append([]FixtureDeclaration(nil), in...)
}

func cavemanFixtureDeclarations() []FixtureDeclaration {
	const repo = "JuliusBrussee/caveman"
	makeDecl := func(name, path, hash, role string) FixtureDeclaration {
		return FixtureDeclaration{
			Name: name, Suite: FixtureSuiteCaveman, Repo: repo, SHA: CavemanFixtureSHA,
			Path: path, ExpectedHash: hash, Role: role, License: "MIT",
			LicenseBoundary: cavemanLicenseBoundary, LicenseBoundaryHash: cavemanLicenseBoundaryHash,
			ReviewToken: CavemanLicenseReviewToken, Normalization: normalizationNone,
		}
	}
	return []FixtureDeclaration{
		makeDecl("caveman-benchmark-runner", "benchmarks/run.py", "sha256:530a387918418713e64ded97794f41a1ffe6a01e833a69d2cb447bf4640facce", "harness"),
		makeDecl("caveman-prompts", "benchmarks/prompts.json", "sha256:773e557f9187363c44e7e5aae2d27268720bcd8772865e119825078b06da93d7", "corpus"),
		makeDecl("caveman-active-skill", "skills/caveman/SKILL.md", "sha256:daf9cec496ebd039809d8236f99f17fa1b4beaadf8ce4e2d532d0da51d70afce", "system_prompt"),
		makeDecl("caveman-benchmark-requirements", "benchmarks/requirements.txt", "sha256:994d50c7e2e135d7621b812c929bb6efca7d8f1ddd1d41476caa71ec3f1eecd1", "eval_metadata"),
		makeDecl("caveman-honest-numbers", "docs/HONEST-NUMBERS.md", "sha256:740ac5e8bb6722c7a5d45f82d8308c5fa6ced93a06eb97040f712363882f7c59", "eval_metadata"),
		makeDecl("caveman-evals-readme", "evals/README.md", "sha256:253bde57a74c83f57d0a5f53ec5ed77cb2cbae1c23a364999f44259c444ab376", "eval_metadata"),
		makeDecl("caveman-benchmark-contract", "tests/test_benchmark_contract.py", "sha256:27a37ae418f00555761ccdb085078933750c87bd9a4c44f93209f29e0b18c678", "eval_metadata"),
		makeDecl("caveman-license", "LICENSE", "sha256:f0abc56b6f49ab2e285bb6e6723f028abb7ebd4fe0e242bbdc2b4dded0ace8b9", "license"),
		makeDecl("caveman-licensing-map", "LICENSING.md", "sha256:d4804b40d29ec31ee03b163e68eec134e1967c7d2cc53d8068ea5c3fabbbf7b4", "license"),
	}
}

func ponytailFixtureDeclarations() []FixtureDeclaration {
	const repo = "DietrichGebert/ponytail"
	makeDecl := func(name, path, hash, role string) FixtureDeclaration {
		return FixtureDeclaration{
			Name: name, Suite: FixtureSuitePonytail, Repo: repo, SHA: PonytailFixtureSHA,
			Path: path, ExpectedHash: hash, Role: role, License: "MIT",
			LicenseBoundary: ponytailLicenseBoundary, LicenseBoundaryHash: ponytailLicenseBoundaryHash,
			Normalization: normalizationNone,
		}
	}
	return []FixtureDeclaration{
		makeDecl("ponytail-promptfoo-claude", "benchmarks/promptfooconfig.yaml", "sha256:fc292f68c5727f306f5ba1ce74b161631e416ce71b66a67d9241a59556c3e00d", "promptfoo_config"),
		makeDecl("ponytail-promptfoo-gpt", "benchmarks/promptfooconfig.gpt.yaml", "sha256:1e196e71efc3c77c54efb10d58634a9e1d1fe58c7e8755ee712825e1c6b940fb", "promptfoo_config"),
		makeDecl("ponytail-promptfoo-gpt-newest", "benchmarks/promptfooconfig.gpt-newest.yaml", "sha256:e3bcfbf34eb148d81bfd3c985d88ea7e7f71c9989b74b9588c8136488ed18f72", "promptfoo_config"),
		makeDecl("ponytail-promptfoo-gemini", "benchmarks/promptfooconfig.gemini.yaml", "sha256:2c1578b259fc3efb755d1c1534f2a01ebe75ea05e6fc2cd6701ea68e99affce6", "promptfoo_config"),
		makeDecl("ponytail-prompts", "benchmarks/prompts.json", "sha256:6137aae013b057d38c90b38af17e6b22d57011b85d0be5d549045df196fa37ac", "corpus"),
		makeDecl("ponytail-arm-baseline", "benchmarks/arms/baseline.js", "sha256:ef0f81f670425705ab3195609947aa64890890cb078d7669780afe2228da8740", "arm"),
		makeDecl("ponytail-arm-caveman", "benchmarks/arms/caveman.js", "sha256:b3793a7549c9217e9e9b89b2ed5e94813a9c71297386c77a177d4e2ef3e13d1a", "arm"),
		makeDecl("ponytail-arm-ponytail", "benchmarks/arms/ponytail.js", "sha256:9091a2e794ad1c5c7c4c9b68a3bb4aa995fcce6d0f09aad37a2febdd95640b82", "arm"),
		makeDecl("ponytail-caveman-skill", "benchmarks/arms/caveman-SKILL.md", "sha256:09ebdef35a85d058f8eba04c6a3d91079ac8dabc3d45d265694f128c808f7648", "system_prompt"),
		makeDecl("ponytail-active-skill", "skills/ponytail/SKILL.md", "sha256:1316a2f3f95741d2300b116fe0c2d81ce4a9568656ed0a62643f54aaf09957f2", "system_prompt"),
		makeDecl("ponytail-loc-definition", "benchmarks/loc.js", "sha256:c3c346f836eec0ba759a100a5c1166be2362fcf37017470f3c9f9b872ab44234", "correctness"),
		makeDecl("ponytail-behavior-definition", "benchmarks/behavior.js", "sha256:9364957a4fade600cd423098e115e2187d12ebcaefc54900f4b368819f33efd2", "behavior"),
		makeDecl("ponytail-behavior-config", "benchmarks/behavior.yaml", "sha256:7f0edb875c56872fa14707a5e096a0f951dfc0b5f6de82fb7bfe505b9fd45c4f", "behavior"),
		makeDecl("ponytail-correctness-definition", "benchmarks/correctness.js", "sha256:7a17d8c904eab1813a220279c31ef51598214735923d178bed9beb92e9cf0230", "correctness"),
		makeDecl("ponytail-robustness-definition", "benchmarks/robustness-audit.js", "sha256:e9a7e9eb6b087e60e2fed07e61370fbd1bfca76b16bb95c86170cffedcebff96", "robustness"),
		makeDecl("ponytail-agentic-tasks", "benchmarks/agentic/tasks.py", "sha256:68f473557695f69a036cd6fdb5d8e9ec51aef407183416545b29823c1e3190a2", "agentic_tasks"),
		makeDecl("ponytail-agentic-judge", "benchmarks/agentic/judge.py", "sha256:c845548290c062dac5bef93ba3231e26b35be62a07fdebabec61833c4a20b6c0", "agentic_judge"),
		makeDecl("ponytail-license", "LICENSE", "sha256:fb1bc6909ac3ef82d5c22106e32ef682b0cff66788fa915fb9b53b15c9d2f3ab", "license"),
	}
}

// DefaultFixtureStore is intentionally outside repository scratch. The caller
// still passes WorkspaceRoot and the importer re-checks the resolved path.
func DefaultFixtureStore() (string, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "fak", "armbench-fixtures"), nil
}

// ImportFixtures materializes one or both pinned suites.
func ImportFixtures(ctx context.Context, suite FixtureSuite, opts ImportOptions) (*FixtureImportReport, error) {
	if suite != FixtureSuiteCaveman && suite != FixtureSuitePonytail && suite != FixtureSuiteAll {
		return nil, refuse(ReasonFixtureDeclarationInvalid, "suite %q is not caveman, ponytail, or all", suite)
	}
	if opts.StoreRoot == "" {
		root, err := DefaultFixtureStore()
		if err != nil {
			return nil, err
		}
		opts.StoreRoot = root
	}
	if err := validateStoreRoot(opts.StoreRoot, opts.WorkspaceRoot); err != nil {
		return nil, err
	}
	if err := validateSuppliedReviewTokens(PinnedFixtureDeclarations(suite), opts.LicenseReviews); err != nil {
		return nil, err
	}
	now := time.Now
	if opts.Now != nil {
		now = opts.Now
	}
	retrievedAt := now().UTC().Format("2006-01-02")
	report := &FixtureImportReport{
		Schema: FixtureImportSchema, RetrievedAt: retrievedAt,
		Store: "content-addressed external store (paths relative to store root)",
	}
	suites := []FixtureSuite{suite}
	if suite == FixtureSuiteAll {
		suites = []FixtureSuite{FixtureSuiteCaveman, FixtureSuitePonytail}
	}
	for _, selected := range suites {
		result, err := importFixtureSuite(ctx, selected, PinnedFixtureDeclarations(selected), opts, retrievedAt)
		if err != nil {
			return nil, err
		}
		report.Results = append(report.Results, result)
	}
	return report, nil
}

type fetchedFixture struct {
	Declaration FixtureDeclaration
	Body        []byte
}

func importFixtureSuite(ctx context.Context, suite FixtureSuite, declarations []FixtureDeclaration, opts ImportOptions, retrievedAt string) (FixtureImportResult, error) {
	if err := validateDeclarations(suite, declarations); err != nil {
		return FixtureImportResult{}, err
	}
	reviewStatus, err := validateLicenseReviews(declarations, opts.LicenseReviews)
	if err != nil {
		return FixtureImportResult{}, err
	}
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultFixtureBaseURL
	}
	client := opts.Client
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	fetched := make([]fetchedFixture, 0, len(declarations))
	var sourceBytes int64
	for _, declaration := range declarations {
		body, err := fetchFixture(ctx, client, baseURL, declaration)
		if err != nil {
			return FixtureImportResult{}, err
		}
		fetched = append(fetched, fetchedFixture{Declaration: declaration, Body: body})
		sourceBytes += int64(len(body))
	}
	if err := os.MkdirAll(filepath.Join(opts.StoreRoot, "blobs", "sha256"), 0o755); err != nil {
		return FixtureImportResult{}, err
	}
	for _, fixture := range fetched {
		if err := ensureBlob(opts.StoreRoot, fixture.Declaration.ExpectedHash, fixture.Body); err != nil {
			return FixtureImportResult{}, err
		}
	}
	corpus, manifest, sourceSetHash, err := buildImportedInput(suite, fetched, reviewStatus, retrievedAt, opts.FakVersion)
	if err != nil {
		return FixtureImportResult{}, err
	}
	manifestBytes, err := MarshalManifest(manifest)
	if err != nil {
		return FixtureImportResult{}, err
	}
	corpusBytes, err := MarshalCorpus(corpus)
	if err != nil {
		return FixtureImportResult{}, err
	}
	inputID := digestParts(manifestBytes, corpusBytes)
	inputHex := strings.TrimPrefix(inputID, "sha256:")
	inputRel := filepath.ToSlash(filepath.Join("inputs", "sha256", inputHex))
	manifestRel := filepath.ToSlash(filepath.Join(inputRel, "manifest.json"))
	corpusRel := filepath.ToSlash(filepath.Join(inputRel, "corpus.json"))
	expected := map[string][]byte{
		"manifest.json": manifestBytes,
		"corpus.json":   corpusBytes,
	}
	for _, fixture := range fetched {
		rel := fixtureCopyPath(fixture.Declaration)
		expected[rel] = fixture.Body
	}
	inputDir := filepath.Join(opts.StoreRoot, filepath.FromSlash(inputRel))
	if err := ensureInputDirectory(inputDir, expected); err != nil {
		return FixtureImportResult{}, err
	}
	return FixtureImportResult{
		Suite: suite, InputID: inputID, InputDir: inputRel,
		ManifestPath: manifestRel, CorpusPath: corpusRel,
		ManifestIdentity: manifest.Identity(), SourceSetHash: sourceSetHash,
		SourceCount: len(fetched), SourceBytes: sourceBytes,
		LicenseReviews: reviewStatus,
	}, nil
}

func validateDeclarations(suite FixtureSuite, declarations []FixtureDeclaration) error {
	if len(declarations) == 0 {
		return refuse(ReasonFixtureDeclarationInvalid, "suite %q has no declared paths", suite)
	}
	seenNames := map[string]bool{}
	seenPaths := map[string]bool{}
	for i, declaration := range declarations {
		if declaration.Suite != suite {
			return refuse(ReasonFixtureDeclarationInvalid, "declaration %d suite %q does not match %q", i, declaration.Suite, suite)
		}
		if strings.TrimSpace(declaration.Name) == "" || seenNames[declaration.Name] {
			return refuse(ReasonFixtureDeclarationInvalid, "declaration %d has empty or duplicate name %q", i, declaration.Name)
		}
		seenNames[declaration.Name] = true
		if strings.TrimSpace(declaration.Repo) == "" || !validCommitSHA(declaration.SHA) {
			return refuse(ReasonFixtureDeclarationInvalid, "declaration %q has invalid repo/SHA", declaration.Name)
		}
		clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(declaration.Path)))
		if clean != declaration.Path || clean == "." || strings.HasPrefix(clean, "../") || filepath.IsAbs(filepath.FromSlash(declaration.Path)) {
			return refuse(ReasonFixtureDeclarationInvalid, "declaration %q path %q is not a clean repository-relative path", declaration.Name, declaration.Path)
		}
		pathKey := declaration.Repo + "@" + declaration.SHA + ":" + declaration.Path
		if seenPaths[pathKey] {
			return refuse(ReasonFixtureDeclarationInvalid, "path %s is declared twice", pathKey)
		}
		seenPaths[pathKey] = true
		if !validSHA256(declaration.ExpectedHash) {
			return refuse(ReasonFixtureDeclarationInvalid, "declaration %q has invalid expected hash %q", declaration.Name, declaration.ExpectedHash)
		}
		if strings.TrimSpace(declaration.Role) == "" {
			return refuse(ReasonFixtureDeclarationInvalid, "declaration %q has no role", declaration.Name)
		}
		if strings.TrimSpace(declaration.License) == "" || strings.TrimSpace(declaration.LicenseBoundary) == "" || !validSHA256(declaration.LicenseBoundaryHash) || strings.TrimSpace(declaration.Normalization) == "" {
			return refuse(ReasonFixtureLicenseMissing, "declaration %q (%s@%s:%s) lacks license, boundary hash, or normalization metadata", declaration.Name, declaration.Repo, declaration.SHA, declaration.Path)
		}
		if isRestrictedCavemanPath(declaration) {
			return refuse(ReasonFixtureRestrictedPath, "Caveman path %q crosses the reviewed benchmark/adoption boundary into mixed-license runtime code", declaration.Path)
		}
	}
	return nil
}

func isRestrictedCavemanPath(declaration FixtureDeclaration) bool {
	if !strings.EqualFold(declaration.Repo, "JuliusBrussee/caveman") {
		return false
	}
	for _, prefix := range []string{
		"engine/", "proxy/", "cache" + "engine/", "rewriter/", "browse/", "mcp/",
		"shrink/", "mem/", "shared/platform/",
	} {
		if strings.HasPrefix(declaration.Path, prefix) {
			return true
		}
	}
	return false
}

func validateLicenseReviews(declarations []FixtureDeclaration, supplied []string) ([]LicenseReview, error) {
	have := map[string]bool{}
	for _, token := range supplied {
		token = strings.TrimSpace(token)
		have[token] = true
	}
	type reviewKey struct{ repo, sha, license, boundary, token string }
	unique := map[reviewKey]bool{}
	for _, declaration := range declarations {
		key := reviewKey{declaration.Repo, declaration.SHA, declaration.License, declaration.LicenseBoundary, declaration.ReviewToken}
		unique[key] = true
	}
	keys := make([]reviewKey, 0, len(unique))
	for key := range unique {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i].repo < keys[j].repo })
	out := make([]LicenseReview, 0, len(keys))
	for _, key := range keys {
		status := "repository-asserted"
		if key.token != "" {
			if !have[key.token] {
				return nil, refuse(ReasonFixtureLicenseReview, "%s@%s is mixed-license/NOASSERTION; review its pinned LICENSE + LICENSING.md and re-run with --review-license %q", key.repo, key.sha, key.token)
			}
			status = "explicit:" + key.token
		}
		out = append(out, LicenseReview{Repo: key.repo, SHA: key.sha, License: key.license, Boundary: key.boundary, Status: status})
	}
	return out, nil
}

func validateSuppliedReviewTokens(declarations []FixtureDeclaration, supplied []string) error {
	known := map[string]bool{}
	for _, declaration := range declarations {
		if declaration.ReviewToken != "" {
			known[declaration.ReviewToken] = true
		}
	}
	for _, token := range supplied {
		token = strings.TrimSpace(token)
		if token == "" {
			return refuse(ReasonFixtureLicenseReview, "empty --review-license token")
		}
		if !known[token] {
			return refuse(ReasonFixtureLicenseReview, "--review-license token %q does not match a review boundary in the selected suite", token)
		}
	}
	return nil
}

func fetchFixture(ctx context.Context, client *http.Client, baseURL string, declaration FixtureDeclaration) ([]byte, error) {
	fetchURL, err := sourceURL(baseURL, declaration)
	if err != nil {
		return nil, refuse(ReasonFixtureDeclarationInvalid, "build URL for %s: %v", declaration.Name, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fetchURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "fak-armbench-fixture-import/1")
	resp, err := client.Do(req)
	if err != nil {
		return nil, refuse(ReasonFixtureFetchFailed, "%s: %v", canonicalSourceURL(declaration), err)
	}
	defer resp.Body.Close()
	if resp.Request != nil && resp.Request.URL.String() != fetchURL {
		return nil, refuse(ReasonFixturePathMoved, "%s redirected to %s; a pinned path may not move", canonicalSourceURL(declaration), resp.Request.URL)
	}
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusGone {
		return nil, refuse(ReasonFixturePathMoved, "%s returned HTTP %d; pinned path moved or vanished", canonicalSourceURL(declaration), resp.StatusCode)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, refuse(ReasonFixtureFetchFailed, "%s returned HTTP %d", canonicalSourceURL(declaration), resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFixtureBytes+1))
	if err != nil {
		return nil, refuse(ReasonFixtureFetchFailed, "%s: %v", canonicalSourceURL(declaration), err)
	}
	if len(body) > maxFixtureBytes {
		return nil, refuse(ReasonFixtureFetchFailed, "%s exceeds %d bytes", canonicalSourceURL(declaration), maxFixtureBytes)
	}
	got := digestBytes(body)
	if got != declaration.ExpectedHash {
		return nil, refuse(ReasonFixtureHashMismatch, "%s hashed to %s, want %s", canonicalSourceURL(declaration), got, declaration.ExpectedHash)
	}
	return body, nil
}

func sourceURL(base string, declaration FixtureDeclaration) (string, error) {
	u, err := url.Parse(strings.TrimRight(base, "/"))
	if err != nil {
		return "", err
	}
	for _, segment := range strings.Split(declaration.Repo+"/"+declaration.SHA+"/"+declaration.Path, "/") {
		u.Path += "/" + url.PathEscape(segment)
	}
	return u.String(), nil
}

func canonicalSourceURL(declaration FixtureDeclaration) string {
	u, _ := sourceURL(defaultFixtureBaseURL, declaration)
	return u
}

func buildImportedInput(suite FixtureSuite, fetched []fetchedFixture, reviews []LicenseReview, retrievedAt, fakVersion string) (*CorpusFile, *Manifest, string, error) {
	if fakVersion == "" {
		fakVersion = "unknown"
	}
	byPath := make(map[string]fetchedFixture, len(fetched))
	for _, fixture := range fetched {
		byPath[fixture.Declaration.Path] = fixture
	}
	if err := validateFetchedLicenseBoundary(suite, byPath); err != nil {
		return nil, nil, "", err
	}
	tasks, err := importedTasks(suite, byPath)
	if err != nil {
		return nil, nil, "", err
	}
	corpus := &CorpusFile{
		Schema: CorpusSchema,
		ID:     fmt.Sprintf("%s-upstream-prompts@%s", suite, fixtureShortSHA(fetched[0].Declaration.SHA)),
		Tasks:  tasks,
	}
	sourceSetHash := declarationSetHash(fetched)
	sources := make([]Source, 0, len(fetched))
	reviewByRepo := map[string]string{}
	for _, review := range reviews {
		reviewByRepo[review.Repo] = review.Status
	}
	for _, fixture := range fetched {
		declaration := fixture.Declaration
		sources = append(sources, Source{
			Name: declaration.Name, Repo: declaration.Repo, URL: canonicalSourceURL(declaration),
			SHA: declaration.SHA, Path: declaration.Path, ContentHash: declaration.ExpectedHash,
			License: declaration.License, LicenseBoundary: declaration.LicenseBoundary,
			LicenseBoundaryHash: declaration.LicenseBoundaryHash, LicenseReview: reviewByRepo[declaration.Repo],
			RetrievedAt: retrievedAt, Normalization: declaration.Normalization,
			LocalPath: fixtureCopyPath(declaration),
		})
	}
	m := &Manifest{
		Schema:  ManifestSchema,
		ID:      fmt.Sprintf("%s-upstream-fixtures-%s", suite, strings.TrimPrefix(sourceSetHash, "sha256:")[:12]),
		Sources: sources,
		Model: Model{
			Provider: "fake", Snapshot: fmt.Sprintf("fixture-import-%s@%s", suite, fixtureShortSHA(fetched[0].Declaration.SHA)),
			Region: "offline", Sampling: Sampling{TopP: 1}, MaxTokens: 4096,
		},
		Corpus: Corpus{ID: corpus.ID, Hash: HashTasks(tasks), TaskCount: len(tasks)},
		Trials: Trials{Count: 3, Seed: 6677, Order: OrderCounterbalanced, Concurrency: 1},
		Environment: Environment{
			OS: runtime.GOOS, Arch: runtime.GOARCH, HostClass: "fixture-import",
			FakVersion: fakVersion, PricingDate: retrievedAt,
		},
	}
	switch suite {
	case FixtureSuiteCaveman:
		m.Model.Sampling.Temperature = 0
		m.Model.MaxTokens = 4096
		m.Judge = Judge{
			ID:   "caveman-explicit-quality-review",
			Hash: mustHash(byPath, "docs/HONEST-NUMBERS.md"),
			Kind: "manual-review-boundary",
		}
		m.Arms = []Arm{
			{ID: "baseline", Kind: ArmBaseline, PromptHash: digestBytes([]byte("You are a helpful assistant."))},
			{ID: "caveman", Kind: ArmUpstreamTreatment, PromptHash: mustHash(byPath, "skills/caveman/SKILL.md"), SourceName: "caveman-active-skill"},
		}
	case FixtureSuitePonytail:
		m.Model.Sampling.Temperature = 1
		m.Model.MaxTokens = 8192
		m.Trials.Count = 10
		m.Judge = Judge{
			ID: "ponytail-correctness-behavior-robustness-agentic",
			Hash: digestParts(
				byPath["benchmarks/correctness.js"].Body,
				byPath["benchmarks/behavior.js"].Body,
				byPath["benchmarks/robustness-audit.js"].Body,
				byPath["benchmarks/agentic/judge.py"].Body,
			),
			Kind: "upstream-multi-gate",
		}
		m.Arms = []Arm{
			{ID: "baseline", Kind: ArmBaseline, PromptHash: mustHash(byPath, "benchmarks/arms/baseline.js")},
			{ID: "caveman", Kind: ArmUpstreamTreatment, PromptHash: mustHash(byPath, "benchmarks/arms/caveman-SKILL.md"), SourceName: "ponytail-caveman-skill"},
			{ID: "ponytail", Kind: ArmUpstreamTreatment, PromptHash: mustHash(byPath, "skills/ponytail/SKILL.md"), SourceName: "ponytail-active-skill"},
		}
	default:
		return nil, nil, "", refuse(ReasonFixtureDeclarationInvalid, "cannot build suite %q", suite)
	}
	if err := m.Validate(); err != nil {
		return nil, nil, "", err
	}
	return corpus, m, sourceSetHash, nil
}

func validateFetchedLicenseBoundary(suite FixtureSuite, byPath map[string]fetchedFixture) error {
	var (
		got      string
		want     string
		required []string
	)
	switch suite {
	case FixtureSuiteCaveman:
		required = []string{"LICENSE", "LICENSING.md"}
		for _, path := range required {
			if _, ok := byPath[path]; !ok {
				return refuse(ReasonFixtureLicenseMissing, "Caveman license boundary is missing pinned path %s", path)
			}
		}
		got = digestParts(byPath["LICENSE"].Body, byPath["LICENSING.md"].Body)
		want = byPath["LICENSE"].Declaration.LicenseBoundaryHash
	case FixtureSuitePonytail:
		required = []string{"LICENSE"}
		if _, ok := byPath["LICENSE"]; !ok {
			return refuse(ReasonFixtureLicenseMissing, "Ponytail license boundary is missing pinned path LICENSE")
		}
		got = digestParts(byPath["LICENSE"].Body)
		want = byPath["LICENSE"].Declaration.LicenseBoundaryHash
	default:
		return refuse(ReasonFixtureDeclarationInvalid, "unknown suite %q", suite)
	}
	if got != want {
		return refuse(ReasonFixtureLicenseMissing, "%s license boundary %v hashed to %s, want reviewed boundary %s", suite, required, got, want)
	}
	for _, fixture := range byPath {
		if fixture.Declaration.LicenseBoundaryHash != got {
			return refuse(ReasonFixtureLicenseMissing, "%s license boundary metadata %s disagrees with fetched boundary %s", fixture.Declaration.Name, fixture.Declaration.LicenseBoundaryHash, got)
		}
	}
	return nil
}

func importedTasks(suite FixtureSuite, byPath map[string]fetchedFixture) ([]Task, error) {
	switch suite {
	case FixtureSuiteCaveman:
		var src struct {
			Version int `json:"version"`
			Prompts []struct {
				ID       string `json:"id"`
				Category string `json:"category"`
				Prompt   string `json:"prompt"`
			} `json:"prompts"`
		}
		if err := decodeStrict(byPath["benchmarks/prompts.json"].Body, &src); err != nil {
			return nil, refuse(ReasonFixtureDeclarationInvalid, "decode Caveman prompts: %v", err)
		}
		tasks := make([]Task, 0, len(src.Prompts))
		for _, prompt := range src.Prompts {
			tasks = append(tasks, Task{ID: prompt.ID, Input: prompt.Prompt, Expect: prompt.ID})
		}
		if err := validateTasks(tasks); err != nil {
			return nil, err
		}
		return tasks, nil
	case FixtureSuitePonytail:
		var src struct {
			Method  string   `json:"method"`
			Configs []string `json:"configs"`
			Tasks   []struct {
				ID     string `json:"id"`
				Prompt string `json:"prompt"`
			} `json:"tasks"`
		}
		if err := decodeStrict(byPath["benchmarks/prompts.json"].Body, &src); err != nil {
			return nil, refuse(ReasonFixtureDeclarationInvalid, "decode Ponytail prompts: %v", err)
		}
		tasks := make([]Task, 0, len(src.Tasks))
		for _, prompt := range src.Tasks {
			tasks = append(tasks, Task{ID: prompt.ID, Input: prompt.Prompt, Expect: prompt.ID})
		}
		if err := validateTasks(tasks); err != nil {
			return nil, err
		}
		return tasks, nil
	default:
		return nil, refuse(ReasonFixtureDeclarationInvalid, "unknown suite %q", suite)
	}
}

func mustHash(byPath map[string]fetchedFixture, path string) string {
	return byPath[path].Declaration.ExpectedHash
}

func declarationSetHash(fetched []fetchedFixture) string {
	sorted := append([]fetchedFixture(nil), fetched...)
	sort.Slice(sorted, func(i, j int) bool {
		a, b := sorted[i].Declaration, sorted[j].Declaration
		if a.Repo != b.Repo {
			return a.Repo < b.Repo
		}
		return a.Path < b.Path
	})
	var parts [][]byte
	for _, fixture := range sorted {
		declaration := fixture.Declaration
		parts = append(parts, []byte(strings.Join([]string{
			declaration.Repo, declaration.SHA, declaration.Path, declaration.ExpectedHash,
			declaration.License, declaration.LicenseBoundary, declaration.LicenseBoundaryHash,
			declaration.Normalization,
		}, "\x00")))
	}
	return digestParts(parts...)
}

func fixtureCopyPath(declaration FixtureDeclaration) string {
	return filepath.ToSlash(filepath.Join("sources", filepath.FromSlash(declaration.Repo), declaration.SHA, filepath.FromSlash(declaration.Path)))
}

func validateStoreRoot(storeRoot, workspaceRoot string) error {
	storeAbs, err := resolveForContainment(storeRoot)
	if err != nil {
		return err
	}
	if workspaceRoot == "" {
		return nil
	}
	workspaceAbs, err := resolveForContainment(workspaceRoot)
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(workspaceAbs, storeAbs)
	if err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return refuse(ReasonFixtureStoreInsideRepo, "fixture store %s resolves inside repository %s; use the user-cache default or another out-of-repository path", storeAbs, workspaceAbs)
	}
	return nil
}

// resolveForContainment follows every symlink in the existing prefix and then
// appends any not-yet-created suffix. A lexical filepath.Abs check alone would
// let an external-looking store symlink back into repository scratch.
func resolveForContainment(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err == nil {
		return resolved, nil
	}
	if !os.IsNotExist(err) {
		return "", err
	}
	var suffix []string
	current := abs
	for {
		parent := filepath.Dir(current)
		if parent == current {
			return abs, nil
		}
		suffix = append([]string{filepath.Base(current)}, suffix...)
		current = parent
		resolved, err = filepath.EvalSymlinks(current)
		if err == nil {
			return filepath.Join(append([]string{resolved}, suffix...)...), nil
		}
		if !os.IsNotExist(err) {
			return "", err
		}
	}
}

func ensureBlob(storeRoot, expectedHash string, body []byte) error {
	path := filepath.Join(storeRoot, "blobs", "sha256", strings.TrimPrefix(expectedHash, "sha256:"))
	return ensureFile(path, body)
}

func ensureInputDirectory(inputDir string, expected map[string][]byte) error {
	if info, err := os.Stat(inputDir); err == nil {
		if !info.IsDir() {
			return refuse(ReasonFixtureLocalMutation, "%s exists but is not a directory", inputDir)
		}
		return verifyDirectory(inputDir, expected)
	} else if !os.IsNotExist(err) {
		return err
	}
	parent := filepath.Dir(inputDir)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(parent, ".import-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	for rel, body := range expected {
		if err := ensureFile(filepath.Join(tmp, filepath.FromSlash(rel)), body); err != nil {
			return err
		}
	}
	if err := os.Rename(tmp, inputDir); err != nil {
		if _, statErr := os.Stat(inputDir); statErr == nil {
			return verifyDirectory(inputDir, expected)
		}
		return err
	}
	return verifyDirectory(inputDir, expected)
}

func verifyDirectory(root string, expected map[string][]byte) error {
	seen := map[string]bool{}
	allowedDirectories := map[string]bool{".": true}
	for rel := range expected {
		dir := filepath.ToSlash(filepath.Dir(filepath.FromSlash(rel)))
		for dir != "." && dir != "/" {
			allowedDirectories[dir] = true
			dir = filepath.ToSlash(filepath.Dir(filepath.FromSlash(dir)))
		}
	}
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if entry.IsDir() {
			if !allowedDirectories[rel] {
				return refuse(ReasonFixtureLocalMutation, "content-addressed input contains unexpected local directory %s", rel)
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return refuse(ReasonFixtureLocalMutation, "content-addressed input contains local symlink %s", rel)
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || info.Mode().Perm()&0o222 != 0 {
			return refuse(ReasonFixtureLocalMutation, "content-addressed input path %s has local mode %s, want read-only regular file", rel, info.Mode())
		}
		want, ok := expected[rel]
		if !ok {
			return refuse(ReasonFixtureLocalMutation, "content-addressed input contains unexpected local path %s", rel)
		}
		seen[rel] = true
		got, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if !bytes.Equal(got, want) {
			return refuse(ReasonFixtureLocalMutation, "%s was locally modified: got %s, want %s", rel, digestBytes(got), digestBytes(want))
		}
		return nil
	})
	if err != nil {
		return err
	}
	for rel := range expected {
		if !seen[filepath.ToSlash(rel)] {
			return refuse(ReasonFixtureLocalMutation, "content-addressed input is missing %s", rel)
		}
	}
	return nil
}

func ensureFile(path string, want []byte) error {
	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() {
			return refuse(ReasonFixtureLocalMutation, "%s exists with local mode %s, want a regular file", path, info.Mode())
		}
		if info.Mode().Perm()&0o222 != 0 {
			return refuse(ReasonFixtureLocalMutation, "%s exists with writable local mode %s, want read-only content-addressed bytes", path, info.Mode())
		}
		got, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if !bytes.Equal(got, want) {
			return refuse(ReasonFixtureLocalMutation, "%s was locally modified: got %s, want %s", path, digestBytes(got), digestBytes(want))
		}
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".fixture-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(want); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmpName, 0o444); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		if _, statErr := os.Lstat(path); statErr == nil {
			return ensureFile(path, want)
		}
		return err
	}
	return nil
}

func digestBytes(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func digestParts(parts ...[]byte) string {
	h := sha256.New()
	var length [8]byte
	for _, part := range parts {
		n := uint64(len(part))
		for i := 0; i < len(length); i++ {
			length[i] = byte(n >> (8 * i))
		}
		_, _ = h.Write(length[:])
		_, _ = h.Write(part)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func fixtureShortSHA(sha string) string {
	if len(sha) <= 12 {
		return sha
	}
	return sha[:12]
}

// MarshalFixtureImportReport renders stable JSON for command proof.
func MarshalFixtureImportReport(report *FixtureImportReport) ([]byte, error) {
	b, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

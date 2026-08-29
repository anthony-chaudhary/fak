package studylink

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

type BuildOptions struct {
	IndexPath     string
	ForgePath     string
	AdjacencyPath string
	RepoRoot      string
}

type ValidateOptions struct {
	LedgerPath    string
	IndexPath     string
	ForgePath     string
	AdjacencyPath string
	RepoRoot      string
}

type issueMatch struct {
	Record ForgeRecord
	Score  int
	Basis  string
	Exact  bool
}

type claim struct {
	Cluster int
	Score   int
}

type witnessSeed struct {
	ClusterID string
	Issue     int
	Paths     []string
	Mode      Disposition
	Reason    string
	Source    string
}

var witnessSeeds = []witnessSeed{
	{
		ClusterID: "apis_tool_calling_structured_output:body:guided-decoding",
		Issue:     26,
		Paths:     []string{"internal/guideddecode"},
		Mode:      Landed,
		Source:    "docs/notes/VLLM-OPTIMIZATION-REUSE-TICKET-MAP-2026-06-30.md#ticket-map-row-5",
	},
	{
		ClusterID: "kv_cache:title:paged-attention",
		Issue:     486,
		Paths:     []string{"internal/compute/cuda_kernels.cu", "internal/compute/cuda_flash_test.go"},
		Mode:      Landed,
		Source:    "captured issue #486 plus issue-linked repository history",
	},
	{
		ClusterID: "kernels_compilation:title:torch-compile",
		Issue:     1731,
		Paths:     []string{"internal/vllmcompile"},
		Mode:      Landed,
		Source:    "docs/notes/VLLM-OPTIMIZATION-REUSE-TICKET-MAP-2026-06-30.md#ticket-map-row-11",
	},
	{
		ClusterID: "observability_operations:body:helm-chart",
		Issue:     2665,
		Paths:     []string{"deploy/k8s/README.md"},
		Mode:      OpenExact,
		Source:    "captured issue #2665 and the checked-in no-Helm-yet deployment witness",
	},
	{
		ClusterID: "observability_operations:title:kubernetes",
		Issue:     2662,
		Paths:     []string{"deploy/k8s"},
		Mode:      Landed,
		Source:    "captured issue #2662 plus issue-linked repository history",
	},
	{
		ClusterID: "kv_cache:title:prefix-cache",
		Issue:     1551,
		Paths:     []string{"internal/engine/vllm_cache_observe.go"},
		Mode:      Partial,
		Reason:    "the exact witness is a vLLM prefix-cache observation adapter, not the complete upstream prefix-cache mechanism",
		Source:    "docs/notes/VLLM-OPTIMIZATION-REUSE-TICKET-MAP-2026-06-30.md#ticket-map-rows-1-and-6",
	},
	{
		ClusterID: "distributed_parallelism:title:tensor-parallel",
		Issue:     4542,
		Paths:     []string{"internal/quality/tensor_parallel.go"},
		Mode:      Partial,
		Reason:    "the exact witness is an output-parity gate; a test does not establish the whole tensor-parallel mechanism",
		Source:    "captured issue #4542 plus issue-linked repository history",
	},
}

var pathPattern = regexp.MustCompile(`(?:internal|cmd|docs|examples|experiments|pkg)/[A-Za-z0-9_.+@/\-]+`)

func Build(opts BuildOptions) (Ledger, Summary, error) {
	index, indexBytes, err := readIndex(opts.IndexPath)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	forge, forgeBytes, err := readForge(opts.ForgePath)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	adjacency, adjacencyBytes, err := readAdjacency(opts.AdjacencyPath)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	repoRoot, err := cleanRepoRoot(opts.RepoRoot)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	revision, trackedPaths, err := repositoryState(repoRoot)
	if err != nil {
		return Ledger{}, Summary{}, fmt.Errorf("studylink: repository state: %w", err)
	}

	issues, err := issueRecords(forge)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	issueByNumber := make(map[int]ForgeRecord, len(issues))
	for _, issue := range issues {
		issueByNumber[issue.Number] = issue
	}
	allMatches := make([][]issueMatch, len(index.Clusters))
	claimsByIssue := map[int][]claim{}
	seedOwners := map[int]int{}
	for _, seed := range witnessSeeds {
		for i, cluster := range index.Clusters {
			if cluster.Key == seed.ClusterID {
				seedOwners[seed.Issue] = i
				break
			}
		}
	}
	for i, cluster := range index.Clusters {
		matches := matchIssues(cluster, issues)
		allMatches[i] = matches
		for _, match := range matches {
			if match.Exact {
				claimsByIssue[match.Record.Number] = append(claimsByIssue[match.Record.Number], claim{i, ownershipScore(cluster, match)})
			}
		}
	}
	owners := exactOwners(claimsByIssue)
	for issue, cluster := range seedOwners {
		owners[issue] = cluster
	}

	ledger := Ledger{
		Schema:         Schema,
		Cutoff:         forge.Receipt.Cutoff,
		SourceRevision: "anthony-chaudhary/fak@" + forge.Receipt.Revision,
		Sources: Sources{
			IndexPath: sourcePath(opts.IndexPath, repoRoot), IndexSHA256: digestBytes(indexBytes),
			IndexClusterDigest: firstNonEmpty(index.CompactClustersChecksum, index.ClustersChecksum),
			ForgePath:          sourcePath(opts.ForgePath, repoRoot), ForgeSHA256: digestBytes(forgeBytes), ForgeSchema: forge.Schema,
			ForgeRevision: forge.Receipt.Revision, ForgeCutoff: forge.Receipt.Cutoff, ForgeRecordCount: len(forge.Records),
			AdjacencyPath: sourcePath(opts.AdjacencyPath, repoRoot), AdjacencySHA256: digestBytes(adjacencyBytes),
			AdjacencyID: adjacency.ID, AdjacencyMembers: len(adjacency.Members),
			RepositoryRoot: ".", RepositoryRevision: revision,
		},
	}
	captured := map[int]CapturedIssue{}
	for i, cluster := range index.Clusters {
		join := buildJoin(cluster, allMatches[i], owners, i, trackedPaths, repoRoot, revision, issueByNumber)
		join.Evidence.Digest = evidenceDigest(join)
		ledger.Joins = append(ledger.Joins, join)
		for _, artifact := range join.Artifacts {
			if artifact.Kind != "issue" {
				continue
			}
			n, _ := strconv.Atoi(artifact.ID)
			captured[n] = CapturedIssue{Number: n, State: artifact.State, Title: artifact.Title, URL: artifact.URL, RecordDigest: artifact.RecordDigest}
		}
	}
	for _, issue := range captured {
		ledger.CapturedIssues = append(ledger.CapturedIssues, issue)
	}
	sort.Slice(ledger.CapturedIssues, func(i, j int) bool { return ledger.CapturedIssues[i].Number < ledger.CapturedIssues[j].Number })
	if err := validateAll(ledger, index, forge, adjacency, repoRoot, indexBytes, forgeBytes, adjacencyBytes); err != nil {
		return Ledger{}, Summary{}, err
	}
	summary, err := Summarize(ledger)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	return ledger, summary, nil
}

func ValidateFiles(opts ValidateOptions) error {
	ledger, err := ReadLedger(opts.LedgerPath)
	if err != nil {
		return err
	}
	index, indexBytes, err := readIndex(opts.IndexPath)
	if err != nil {
		return err
	}
	forge, forgeBytes, err := readForge(opts.ForgePath)
	if err != nil {
		return err
	}
	adjacency, adjacencyBytes, err := readAdjacency(opts.AdjacencyPath)
	if err != nil {
		return err
	}
	repoRoot, err := cleanRepoRoot(opts.RepoRoot)
	if err != nil {
		return err
	}
	return validateAll(ledger, index, forge, adjacency, repoRoot, indexBytes, forgeBytes, adjacencyBytes)
}

func validateAll(ledger Ledger, index CompactIndex, forge ForgeCorpus, adjacency AdjacencyManifest, repoRoot string, indexBytes, forgeBytes, adjacencyBytes []byte) error {
	if err := ValidateStructure(ledger, &index, repoRoot); err != nil {
		return err
	}
	if ledger.Sources.IndexSHA256 != digestBytes(indexBytes) || ledger.Sources.ForgeSHA256 != digestBytes(forgeBytes) || ledger.Sources.AdjacencySHA256 != digestBytes(adjacencyBytes) {
		return invalidf("source checksum mismatch")
	}
	if ledger.Sources.ForgeRevision != forge.Receipt.Revision || ledger.Sources.ForgeCutoff != forge.Receipt.Cutoff || ledger.Sources.ForgeRecordCount != len(forge.Records) {
		return invalidf("forge receipt mismatch")
	}
	if ledger.Sources.AdjacencyID != adjacency.ID || ledger.Sources.AdjacencyMembers != len(adjacency.Members) {
		return invalidf("adjacency manifest mismatch")
	}
	issues, err := issueRecords(forge)
	if err != nil {
		return err
	}
	issueByNumber := map[int]ForgeRecord{}
	for _, issue := range issues {
		issueByNumber[issue.Number] = issue
	}
	for _, captured := range ledger.CapturedIssues {
		issue, ok := issueByNumber[captured.Number]
		if !ok {
			return invalidf("captured issue #%d missing from forge corpus", captured.Number)
		}
		if issue.State != captured.State {
			return invalidf("captured issue #%d state %s != forge %s", captured.Number, captured.State, issue.State)
		}
		if got := issueRecordDigest(issue); got != captured.RecordDigest {
			return invalidf("captured issue #%d record digest mismatch", captured.Number)
		}
	}
	trackedAtRevision, err := repositoryPathsAtRevision(repoRoot, ledger.Sources.RepositoryRevision)
	if err != nil {
		return invalidf("cannot read repository paths at %s: %v", ledger.Sources.RepositoryRevision, err)
	}
	for _, join := range ledger.Joins {
		for _, artifact := range join.Artifacts {
			if artifact.Kind != "repo_path" {
				continue
			}
			if !pathTracked(artifact.Path, trackedAtRevision) {
				return invalidf("cluster %s repo path %s is not tracked at %s", join.ClusterID, artifact.Path, ledger.Sources.RepositoryRevision)
			}
		}
	}
	return nil
}

func buildJoin(cluster Cluster, matches []issueMatch, owners map[int]int, clusterIndex int, trackedPaths []string, repoRoot, revision string, issueByNumber map[int]ForgeRecord) Join {
	join := Join{
		ClusterID: cluster.Key, Mechanism: cluster.Mechanism, Signal: cluster.Signal, Rule: cluster.Rule,
		Actionable: cluster.Actionable, Confidence: "none", MembersChecksum: cluster.MembersChecksum,
		Evidence: Evidence{Query: queryFor(cluster)},
	}
	allEvidenceMatches := make([]string, 0, len(matches)+8)
	ownedExact := []issueMatch{}
	for _, match := range matches {
		allEvidenceMatches = append(allEvidenceMatches, issueMatchString(match))
		if match.Exact && owners[match.Record.Number] == clusterIndex {
			ownedExact = append(ownedExact, match)
		}
	}
	pathMatches := matchPaths(cluster, trackedPaths)
	for _, path := range pathMatches {
		allEvidenceMatches = append(allEvidenceMatches, "repo_path:"+path+" basis=path_tokens")
	}
	if seed, ok := seedForCluster(cluster.Key); ok {
		if _, present := issueByNumber[seed.Issue]; present {
			seedJoin, seedMatches := buildSeedJoin(join, seed, issueByNumber, repoRoot, revision)
			allEvidenceMatches = append(seedMatches, allEvidenceMatches...)
			setBoundedEvidence(&seedJoin, allEvidenceMatches)
			return seedJoin
		}
	}
	if !cluster.Actionable {
		if cluster.Mechanism == "explicit_non_candidate" {
			join.Disposition = Obsolete
			join.Confidence = "source-explicit-noncandidate"
		} else {
			join.Disposition, join.Confidence, join.ManualReview = Partial, "source-nonactionable-only", true
			join.ManualReason = "the upstream classifier marked this cluster non-actionable, which is not evidence that the FAK mechanism is obsolete"
		}
		setBoundedEvidence(&join, allEvidenceMatches)
		return join
	}
	if len(ownedExact) > 1 {
		join.Disposition, join.Confidence, join.ManualReview = Conflict, "ambiguous", true
		join.ManualReason = "multiple FAK issue titles contain the exact cluster signal; no exact issue is selected automatically"
		join.Artifacts = issueArtifacts(firstIssueMatches(ownedExact, 4), false)
		setBoundedEvidence(&join, allEvidenceMatches)
		return join
	}
	if len(ownedExact) == 1 {
		match := ownedExact[0]
		issueArtifact := artifactForIssue(match.Record, true)
		paths := referencedExistingPaths(match.Record.Body, repoRoot)
		join.Artifacts = append(join.Artifacts, issueArtifact)
		historyPaths := 0
		for _, path := range paths {
			artifactRevision := revision
			exactPath := false
			if historyRevision := issueLinkedHistory(repoRoot, match.Record.Number, path); historyRevision != "" {
				artifactRevision = historyRevision
				exactPath = true
				historyPaths++
				allEvidenceMatches = append(allEvidenceMatches, fmt.Sprintf("git:%s issue=#%d path=%s", historyRevision, match.Record.Number, path))
			}
			join.Artifacts = append(join.Artifacts, Artifact{Kind: "repo_path", ID: path, Path: path, Revision: artifactRevision, Exact: exactPath})
		}
		if match.Record.State == "open" {
			join.Disposition, join.Confidence = OpenExact, "exact-title"
			setBoundedEvidence(&join, allEvidenceMatches)
			return join
		}
		if match.Record.State == "closed" && historyPaths > 0 {
			join.Disposition, join.Confidence = Landed, "exact-title+captured-closed+issue-linked-history"
			setBoundedEvidence(&join, allEvidenceMatches)
			return join
		}
		join.Disposition, join.Confidence, join.ManualReview = Partial, "exact-title-only", true
		join.ManualReason = "the exact issue is closed but no existing repository path is linked to that issue by commit history"
		setBoundedEvidence(&join, allEvidenceMatches)
		return join
	}

	weak := make([]issueMatch, 0, 4)
	for _, match := range matches {
		if !match.Exact && len(weak) < 4 {
			weak = append(weak, match)
		}
	}
	if len(weak) > 0 || len(pathMatches) > 0 {
		join.Disposition, join.Confidence, join.ManualReview = Partial, "lexical-candidate", true
		join.ManualReason = "lexical evidence is reproducible but insufficient for an exact semantic join"
		join.Artifacts = append(join.Artifacts, issueArtifacts(weak, false)...)
		for _, path := range firstStrings(pathMatches, 4) {
			join.Artifacts = append(join.Artifacts, Artifact{Kind: "repo_path", ID: path, Path: path, Revision: revision})
		}
		setBoundedEvidence(&join, allEvidenceMatches)
		return join
	}
	join.Disposition = Uncovered
	join.Confidence = "no-reproducible-match"
	setBoundedEvidence(&join, allEvidenceMatches)
	return join
}

func seedForCluster(clusterID string) (witnessSeed, bool) {
	for _, seed := range witnessSeeds {
		if seed.ClusterID == clusterID {
			return seed, true
		}
	}
	return witnessSeed{}, false
}

func buildSeedJoin(join Join, seed witnessSeed, issueByNumber map[int]ForgeRecord, repoRoot, revision string) (Join, []string) {
	issue, ok := issueByNumber[seed.Issue]
	if !ok {
		join.Disposition, join.Confidence, join.ManualReview = Conflict, "missing-explicit-witness", true
		join.ManualReason = fmt.Sprintf("explicit witness issue #%d is absent from the captured issue corpus", seed.Issue)
		return join, []string{fmt.Sprintf("explicit_witness:%s issue=#%d missing", seed.Source, seed.Issue)}
	}
	exact := seed.Mode == Landed || seed.Mode == OpenExact
	join.Artifacts = append(join.Artifacts, artifactForIssue(issue, exact))
	evidence := []string{fmt.Sprintf("explicit_witness:%s issue=#%d state=%s record=%s", seed.Source, issue.Number, issue.State, issueRecordDigest(issue))}
	historyFound := false
	for _, path := range seed.Paths {
		if _, err := os.Stat(filepath.Join(repoRoot, filepath.FromSlash(path))); err != nil {
			join.Disposition, join.Confidence, join.ManualReview = Conflict, "broken-explicit-witness", true
			join.ManualReason = fmt.Sprintf("explicit witness path %s is missing", path)
			evidence = append(evidence, "repo_path:"+path+" basis=explicit_witness missing")
			continue
		}
		artifactRevision := revision
		if seed.Mode == Landed || seed.Mode == Partial {
			if historyRevision := issueLinkedHistory(repoRoot, issue.Number, path); historyRevision != "" {
				artifactRevision = historyRevision
				historyFound = true
				evidence = append(evidence, fmt.Sprintf("git:%s issue=#%d path=%s", historyRevision, issue.Number, path))
			}
		}
		join.Artifacts = append(join.Artifacts, Artifact{Kind: "repo_path", ID: path, Path: path, Revision: artifactRevision, Exact: exact})
		evidence = append(evidence, "repo_path:"+path+" basis=explicit_witness")
	}
	if join.Disposition == Conflict {
		return join, evidence
	}
	switch seed.Mode {
	case Landed:
		if issue.State != "closed" {
			join.Disposition, join.Confidence, join.ManualReview = Conflict, "explicit-state-conflict", true
			join.ManualReason = fmt.Sprintf("explicit landed witness issue #%d is captured %s", issue.Number, issue.State)
		} else if !historyFound {
			join.Disposition, join.Confidence, join.ManualReview = Partial, "explicit-closed-without-history", true
			join.ManualReason = fmt.Sprintf("explicit issue #%d is closed and its path exists, but no issue-linked commit was found", issue.Number)
		} else {
			join.Disposition, join.Confidence = Landed, "explicit-map+captured-closed+history"
		}
	case OpenExact:
		if issue.State != "open" {
			join.Disposition, join.Confidence, join.ManualReview = Conflict, "explicit-state-conflict", true
			join.ManualReason = fmt.Sprintf("explicit open witness issue #%d is captured %s", issue.Number, issue.State)
		} else {
			join.Disposition, join.Confidence = OpenExact, "explicit-open-scope"
		}
	case Partial:
		join.Disposition, join.Confidence, join.ManualReview = Partial, "explicit-partial-scope", true
		join.ManualReason = seed.Reason
	case Conflict, Obsolete, Uncovered:
		join.Disposition, join.Confidence, join.ManualReview = Conflict, "invalid-explicit-witness-mode", true
		join.ManualReason = fmt.Sprintf("explicit witness mode %s cannot seed affirmative repository evidence", seed.Mode)
	}
	return join, evidence
}

func issueLinkedHistory(repoRoot string, issue int, path string) string {
	pattern := fmt.Sprintf(`(^|[^0-9])#%d([^0-9]|$)`, issue)
	cmd := exec.Command("git", "-C", repoRoot, "log", "-1", "--perl-regexp", "--format=%H", "--grep="+pattern, "--", path)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func setBoundedEvidence(join *Join, allMatches []string) {
	allMatches = stableUnique(allMatches)
	join.Evidence.MatchCount = len(allMatches)
	canonical := append([]string(nil), allMatches...)
	sort.Strings(canonical)
	joined, _ := json.Marshal(canonical)
	join.Evidence.FullMatchesDigest = digestBytes(joined)
	if len(allMatches) > 12 {
		allMatches = allMatches[:12]
	}
	join.Evidence.Matches = append([]string(nil), allMatches...)
}

func stableUnique(values []string) []string {
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func issueArtifacts(matches []issueMatch, exact bool) []Artifact {
	artifacts := make([]Artifact, 0, len(matches))
	for _, match := range matches {
		artifacts = append(artifacts, artifactForIssue(match.Record, exact))
	}
	return artifacts
}

func firstIssueMatches(matches []issueMatch, limit int) []issueMatch {
	if len(matches) <= limit {
		return matches
	}
	return matches[:limit]
}

func firstStrings(values []string, limit int) []string {
	if len(values) <= limit {
		return values
	}
	return values[:limit]
}

func artifactForIssue(issue ForgeRecord, exact bool) Artifact {
	return Artifact{Kind: "issue", ID: strconv.Itoa(issue.Number), State: issue.State, Title: issue.Title, URL: issue.URL, Exact: exact, RecordDigest: issueRecordDigest(issue)}
}

func exactOwners(byIssue map[int][]claim) map[int]int {
	owners := map[int]int{}
	for issue, claims := range byIssue {
		sort.Slice(claims, func(i, j int) bool {
			if claims[i].Score != claims[j].Score {
				return claims[i].Score > claims[j].Score
			}
			return claims[i].Cluster < claims[j].Cluster
		})
		if len(claims) == 1 || claims[0].Score > claims[1].Score {
			owners[issue] = claims[0].Cluster
		} else {
			owners[issue] = -1
		}
	}
	return owners
}

func ownershipScore(cluster Cluster, match issueMatch) int {
	terms := significantTerms(cluster.Signal)
	field := clusterField(cluster.Key)
	priority := map[string]int{"title": 300, "body": 200, "label": 100}[field]
	return priority + len(terms)*20 + len(normalize(cluster.Signal)) + match.Score
}

func matchIssues(cluster Cluster, issues []ForgeRecord) []issueMatch {
	phrase := normalize(cluster.Signal)
	terms := significantTerms(cluster.Signal)
	allowExact := clusterField(cluster.Key) == "title" && exactSignalEligible(terms)
	matches := make([]issueMatch, 0)
	for _, issue := range issues {
		title := normalize(issue.Title)
		body := normalize(issue.Body)
		m := issueMatch{Record: issue}
		switch {
		case allowExact && containsPhrase(title, phrase) && strongIssueEligible(cluster, issue):
			m.Score, m.Basis, m.Exact = 120+len(terms)*5, "title_phrase+implementation_scope", true
		case len(terms) > 0 && allTerms(title, terms):
			m.Score, m.Basis = 70+len(terms)*3, "title_tokens"
		case phrase != "" && containsPhrase(body, phrase):
			m.Score, m.Basis = 45+len(terms)*2, "body_phrase"
		case len(terms) >= 2 && allTerms(body, terms):
			m.Score, m.Basis = 25+len(terms), "body_tokens"
		default:
			continue
		}
		matches = append(matches, m)
	}
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Exact != matches[j].Exact {
			return matches[i].Exact
		}
		if matches[i].Score != matches[j].Score {
			return matches[i].Score > matches[j].Score
		}
		return matches[i].Record.Number < matches[j].Record.Number
	})
	return matches
}

func strongIssueEligible(cluster Cluster, issue ForgeRecord) bool {
	title := strings.TrimSpace(strings.ToLower(issue.Title))
	allowedPrefix := false
	for _, prefix := range []string{"feat(", "fix(", "perf(", "gpu(", "serve(", "model(", "engine(", "security(", "reliability(", "ops(", "feat:", "fix:", "perf:"} {
		if strings.HasPrefix(title, prefix) {
			allowedPrefix = true
			break
		}
	}
	if !allowedPrefix {
		return false
	}
	anchors := map[string][]string{
		"apis_tool_calling_structured_output": {"tool", "structured", "schema", "guided", "gateway", "model"},
		"architecture_runtime":                {"engine", "runtime", "worker", "executor", "model"},
		"distributed_parallelism":             {"tensor", "pipeline", "expert", "collective", "nccl", "model", "gpu"},
		"kernels_compilation":                 {"kernel", "cuda", "triton", "gemm", "compile", "gpu"},
		"kv_cache":                            {"kv", "cache", "paged", "prefix", "block"},
		"memory_residency":                    {"memory", "allocator", "oom", "offload", "swap", "gpu", "model"},
		"model_backend_hardware":              {"model", "backend", "inference", "gpu", "cpu", "cuda", "rocm", "qwen", "llama", "glm"},
		"observability_operations":            {"serving", "model", "engine", "gateway", "worker"},
		"reliability_security":                {"serving", "model", "engine", "gateway", "worker", "runtime"},
		"scheduling_batching":                 {"scheduler", "batch", "prefill", "decode", "serving", "model"},
		"speculative_decoding":                {"speculative", "draft", "eagle", "dflash", "decode", "model"},
	}
	if cluster.Mechanism == "tests_ci_docs" || cluster.Mechanism == "explicit_non_candidate" {
		return false
	}
	titleNormalized := " " + normalize(issue.Title) + " "
	for _, anchor := range anchors[cluster.Mechanism] {
		if strings.Contains(titleNormalized, " "+normalize(anchor)+" ") {
			return true
		}
	}
	return false
}

func matchPaths(cluster Cluster, tracked []string) []string {
	terms := significantTerms(cluster.Signal)
	if len(terms) == 0 {
		return nil
	}
	out := []string{}
	for _, path := range tracked {
		if strings.HasPrefix(path, "docs/research/vllm-fak-join-") || strings.HasPrefix(path, "docs/_witnesses/") || strings.HasPrefix(path, "docs/research/inventory/") {
			continue
		}
		n := normalize(path)
		if allTerms(n, terms) {
			out = append(out, path)
		}
	}
	sort.Strings(out)
	return out
}

func issueMatchString(match issueMatch) string {
	return fmt.Sprintf("issue:#%d state=%s basis=%s score=%d record=%s", match.Record.Number, match.Record.State, match.Basis, match.Score, issueRecordDigest(match.Record))
}

func queryFor(cluster Cluster) string {
	return fmt.Sprintf("signal=%q field=%s mechanism=%s policy=exact-title-or-conservative-lexical", normalize(cluster.Signal), clusterField(cluster.Key), cluster.Mechanism)
}

func clusterField(key string) string {
	parts := strings.Split(key, ":")
	if len(parts) >= 2 {
		return parts[1]
	}
	return ""
}

func exactSignalEligible(terms []string) bool {
	if len(terms) >= 2 {
		return true
	}
	if len(terms) != 1 || len(terms[0]) < 6 {
		return false
	}
	_, generic := genericSingleTerms[terms[0]]
	return !generic
}

var genericSingleTerms = map[string]struct{}{
	"architecture": {}, "backend": {}, "compiler": {}, "cuda": {}, "deployment": {}, "distributed": {}, "documentation": {},
	"executor": {}, "frontend": {}, "gpu": {}, "hardware": {}, "installation": {}, "logging": {}, "metrics": {}, "observability": {},
	"profiling": {}, "reliability": {}, "scheduler": {}, "scheduling": {}, "security": {}, "startup": {}, "telemetry": {}, "tracing": {},
}

var stopTerms = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "api": {}, "for": {}, "in": {}, "of": {}, "on": {}, "the": {}, "to": {}, "with": {},
}

func significantTerms(s string) []string {
	parts := strings.Fields(normalize(s))
	out := make([]string, 0, len(parts))
	seen := map[string]bool{}
	for _, term := range parts {
		if len(term) < 2 {
			continue
		}
		if _, stop := stopTerms[term]; stop {
			continue
		}
		if !seen[term] {
			out = append(out, term)
			seen[term] = true
		}
	}
	return out
}

func normalize(s string) string {
	var b strings.Builder
	space := true
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			space = false
		} else if !space {
			b.WriteByte(' ')
			space = true
		}
	}
	return strings.TrimSpace(b.String())
}

func containsPhrase(text, phrase string) bool {
	if phrase == "" {
		return false
	}
	return strings.Contains(" "+text+" ", " "+phrase+" ")
}

func allTerms(text string, terms []string) bool {
	padded := " " + text + " "
	for _, term := range terms {
		if !strings.Contains(padded, " "+term+" ") {
			return false
		}
	}
	return len(terms) > 0
}

func referencedExistingPaths(body, repoRoot string) []string {
	seen := map[string]bool{}
	var out []string
	for _, raw := range pathPattern.FindAllString(body, -1) {
		path := strings.TrimRight(raw, ".,:;)]}`'\"")
		if i := strings.Index(path, ":"); i >= 0 {
			path = path[:i]
		}
		path = filepath.ToSlash(filepath.Clean(path))
		if path == "." || strings.Contains(path, "...") || seen[path] {
			continue
		}
		if _, err := os.Stat(filepath.Join(repoRoot, filepath.FromSlash(path))); err != nil {
			continue
		}
		seen[path] = true
		out = append(out, path)
	}
	sort.Strings(out)
	if len(out) > 8 {
		out = out[:8]
	}
	return out
}

func issueRecords(forge ForgeCorpus) ([]ForgeRecord, error) {
	if forge.Schema != "fak-studyforge-corpus/1" || forge.Receipt.Schema != "fak-studyforge-receipt/1" || forge.Receipt.Status != "complete" || forge.Receipt.Revision == "" || forge.Receipt.Cutoff == "" {
		return nil, invalidf("forge corpus is not a complete terminal capture")
	}
	for _, source := range forge.Receipt.Sources {
		if source.Status != "complete" {
			return nil, invalidf("forge source %s is %s", source.Name, source.Status)
		}
	}
	seen := map[int]bool{}
	var issues []ForgeRecord
	for _, record := range forge.Records {
		if record.Kind != "issue" {
			continue
		}
		if record.Number <= 0 || record.State == "" || record.Title == "" {
			return nil, invalidf("forge contains malformed issue record")
		}
		if seen[record.Number] {
			return nil, invalidf("forge contains duplicate issue #%d", record.Number)
		}
		seen[record.Number] = true
		issues = append(issues, record)
	}
	if len(issues) == 0 {
		return nil, invalidf("forge contains no issue records")
	}
	sort.Slice(issues, func(i, j int) bool { return issues[i].Number < issues[j].Number })
	return issues, nil
}

func readIndex(path string) (CompactIndex, []byte, error) {
	var index CompactIndex
	b, err := os.ReadFile(path)
	if err != nil {
		return index, nil, fmt.Errorf("studylink: read index: %w", err)
	}
	if err := json.Unmarshal(b, &index); err != nil {
		return index, nil, fmt.Errorf("studylink: decode index: %w", err)
	}
	if len(index.Clusters) == 0 {
		return index, nil, invalidf("index has no clusters")
	}
	seen := map[string]bool{}
	for _, c := range index.Clusters {
		if c.Key == "" || c.Signal == "" || seen[c.Key] {
			return index, nil, invalidf("index contains invalid or duplicate cluster %q", c.Key)
		}
		seen[c.Key] = true
	}
	return index, b, nil
}

func readForge(path string) (ForgeCorpus, []byte, error) {
	var forge ForgeCorpus
	b, err := os.ReadFile(path)
	if err != nil {
		return forge, nil, fmt.Errorf("studylink: read forge: %w", err)
	}
	if err := json.Unmarshal(b, &forge); err != nil {
		return forge, nil, fmt.Errorf("studylink: decode forge: %w", err)
	}
	if _, err := issueRecords(forge); err != nil {
		return forge, nil, err
	}
	return forge, b, nil
}

func readAdjacency(path string) (AdjacencyManifest, []byte, error) {
	var adjacency AdjacencyManifest
	b, err := os.ReadFile(path)
	if err != nil {
		return adjacency, nil, fmt.Errorf("studylink: read adjacency: %w", err)
	}
	if err := json.Unmarshal(b, &adjacency); err != nil {
		return adjacency, nil, fmt.Errorf("studylink: decode adjacency: %w", err)
	}
	if adjacency.Schema != "fak-study-adjacency/1" || adjacency.ID == "" || len(adjacency.Members) == 0 {
		return adjacency, nil, invalidf("invalid adjacency manifest")
	}
	return adjacency, b, nil
}

func cleanRepoRoot(root string) (string, error) {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		return "", fmt.Errorf("studylink: invalid repository root %q", root)
	}
	return abs, nil
}

func repositoryState(root string) (string, []string, error) {
	revCmd := exec.Command("git", "-C", root, "rev-parse", "HEAD")
	revBytes, err := revCmd.Output()
	if err != nil {
		return "", nil, err
	}
	listCmd := exec.Command("git", "-C", root, "ls-files", "-z")
	listBytes, err := listCmd.Output()
	if err != nil {
		return "", nil, err
	}
	parts := bytes.Split(listBytes, []byte{0})
	paths := make([]string, 0, len(parts))
	for _, part := range parts {
		if len(part) > 0 {
			paths = append(paths, filepath.ToSlash(string(part)))
		}
	}
	sort.Strings(paths)
	return strings.TrimSpace(string(revBytes)), paths, nil
}

func repositoryPathsAtRevision(root, revision string) ([]string, error) {
	cmd := exec.Command("git", "-C", root, "ls-tree", "-r", "--name-only", "-z", revision)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	parts := bytes.Split(out, []byte{0})
	paths := make([]string, 0, len(parts))
	for _, part := range parts {
		if len(part) > 0 {
			paths = append(paths, filepath.ToSlash(string(part)))
		}
	}
	sort.Strings(paths)
	return paths, nil
}

func pathTracked(path string, tracked []string) bool {
	i := sort.SearchStrings(tracked, path)
	if i < len(tracked) && tracked[i] == path {
		return true
	}
	prefix := strings.TrimSuffix(path, "/") + "/"
	i = sort.SearchStrings(tracked, prefix)
	return i < len(tracked) && strings.HasPrefix(tracked[i], prefix)
}

func sourcePath(path, repoRoot string) string {
	abs, err := filepath.Abs(path)
	if err == nil {
		if rel, relErr := filepath.Rel(repoRoot, abs); relErr == nil && !strings.HasPrefix(rel, "..") {
			return filepath.ToSlash(rel)
		}
	}
	return filepath.Base(path)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

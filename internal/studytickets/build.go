package studytickets

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
	"github.com/anthony-chaudhary/fak/internal/studyprio"
)

var issueMapping = map[string]int{
	"native-vllm-ir":          9377,
	"allocator-fragmentation": 9378,
}

var requiredSections = []string{
	"For", "Problem", "Today", "Better because", "Witness", "Centrality",
	"P1-P4", "Dependencies", "Close condition",
}

var sampleClusterIDs = []string{
	"apis_tool_calling_structured_output:body:guided-decoding",
	"apis_tool_calling_structured_output:title:structured-output",
	"apis_tool_calling_structured_output:body:json-schema",
	"model_backend_hardware:title:model-support",
	"explicit_non_" + "candidate:disposition:release-metadata-noncandidate",
}

var refreshObligations = []string{
	"Recapture the complete anthony-chaudhary/fak study-forge corpus before relying on issue open state after either mapped issue changes.",
	"Rebuild the study-link ledger when its classification index, FAK forge capture, repository revision, or adjacency manifest changes.",
	"Rebuild the study-priority ledger when the uncovered actionable set, hard gates, rubric, dependencies, or queue order changes.",
	"Run fak study-tickets build and validate whenever any recorded source checksum changes.",
	"Refresh this closure if #9377 or #9378 changes state, title, body, labels, horizon, source-cluster links, or dependencies.",
}

type joinLedger struct {
	Schema         string        `json:"schema"`
	Cutoff         string        `json:"cutoff"`
	SourceRevision string        `json:"source_revision"`
	Sources        joinSources   `json:"sources"`
	Joins          []joinCluster `json:"joins"`
}

type joinSources struct {
	AdjacencyPath    string `json:"adjacency_path"`
	AdjacencySHA256  string `json:"adjacency_sha256"`
	AdjacencyID      string `json:"adjacency_id"`
	AdjacencyMembers int    `json:"adjacency_members"`
}

type joinCluster struct {
	ClusterID     string         `json:"cluster_id"`
	Actionable    bool           `json:"actionable"`
	Disposition   string         `json:"disposition"`
	Artifacts     []joinArtifact `json:"artifacts"`
	Confidence    string         `json:"confidence"`
	Evidence      joinEvidence   `json:"evidence"`
	ManualReview  bool           `json:"manual_review"`
	ManualReason  string         `json:"manual_reason"`
	MembersSHA256 string         `json:"members_checksum"`
}

type joinArtifact struct {
	Kind string `json:"kind"`
	ID   string `json:"id"`
}

type joinEvidence struct {
	Digest string `json:"digest"`
}

type adjacencyManifest struct {
	Schema  string            `json:"schema"`
	ID      string            `json:"id"`
	Members []adjacencyMember `json:"members"`
}

type adjacencyMember struct {
	Repository          adjacencyRepository `json:"repository"`
	Processed           bool                `json:"processed"`
	SourceClassReceipts []adjacencyClassRaw `json:"source_class_receipts"`
}

type adjacencyRepository struct {
	Owner string `json:"owner"`
	Repo  string `json:"repo"`
}

type adjacencyClassRaw struct {
	Class           string `json:"class"`
	Status          string `json:"status"`
	TerminalReceipt string `json:"terminal_receipt"`
	Notes           string `json:"notes"`
}

type classificationIndex struct {
	Input struct {
		Revision      string `json:"revision"`
		Cutoff        string `json:"cutoff"`
		RecordCount   int    `json:"record_count"`
		IndexChecksum string `json:"index_checksum"`
	} `json:"input"`
}

func Build(opts BuildOptions) (Ledger, Report, error) {
	priority, priorityBytes, err := readJSON[studyprio.Ledger](opts.PriorityPath)
	if err != nil {
		return Ledger{}, Report{}, err
	}
	join, joinBytes, err := readJSON[joinLedger](opts.JoinPath)
	if err != nil {
		return Ledger{}, Report{}, err
	}
	forge, forgeBytes, err := readJSON[studyforge.Corpus](opts.ForgePath)
	if err != nil {
		return Ledger{}, Report{}, err
	}
	if err := studyforge.Validate(forge); err != nil {
		return Ledger{}, Report{}, invalidf("forge corpus: %v", err)
	}
	adjacency, adjacencyBytes, err := readJSON[adjacencyManifest](opts.AdjacencyPath)
	if err != nil {
		return Ledger{}, Report{}, err
	}
	classification, classificationBytes, err := readJSON[classificationIndex](opts.ClassificationPath)
	if err != nil {
		return Ledger{}, Report{}, err
	}
	ledger, err := buildLedger(opts, priority, priorityBytes, join, joinBytes, forge, forgeBytes, adjacency, adjacencyBytes, classification, classificationBytes)
	if err != nil {
		return Ledger{}, Report{}, err
	}
	if err := Validate(ledger); err != nil {
		return Ledger{}, Report{}, err
	}
	ledgerBytes, err := MarshalLedger(ledger)
	if err != nil {
		return Ledger{}, Report{}, err
	}
	return ledger, Report{Schema: ReportSchema, LedgerSHA256: digest(ledgerBytes), Detail: ledger}, nil
}

func buildLedger(opts BuildOptions, priority studyprio.Ledger, priorityBytes []byte, join joinLedger, joinBytes []byte, forge studyforge.Corpus, forgeBytes []byte, adjacency adjacencyManifest, adjacencyBytes []byte, classification classificationIndex, classificationBytes []byte) (Ledger, error) {
	if join.Schema != "fak.study-join-ledger/1" || len(join.Joins) != 193 {
		return Ledger{}, invalidf("join ledger must contain 193 complete dispositions")
	}
	if priority.Source.SHA256 != digest(joinBytes) {
		return Ledger{}, invalidf("priority source checksum does not match join ledger")
	}
	if join.Sources.AdjacencySHA256 != digest(adjacencyBytes) {
		return Ledger{}, invalidf("join adjacency checksum does not match adjacency manifest")
	}
	if adjacency.Schema != "fak-study-adjacency/1" || adjacency.ID != join.Sources.AdjacencyID || len(adjacency.Members) != join.Sources.AdjacencyMembers {
		return Ledger{}, invalidf("adjacency receipt does not match join ledger")
	}
	if classification.Input.RecordCount != 53848 || classification.Input.Revision == "" || classification.Input.Cutoff == "" || classification.Input.IndexChecksum == "" {
		return Ledger{}, invalidf("classification source identity mismatch")
	}

	coverage, uncovered, samples, err := buildCoverage(join)
	if err != nil {
		return Ledger{}, err
	}
	tickets, queue, err := buildTickets(priority, forge, uncovered)
	if err != nil {
		return Ledger{}, err
	}
	coverage.QueueSelections = len(priority.Candidates)
	coverage.MappedSourceClusters = len(uncovered)
	coverage.CreatedCount = len(tickets)
	coverage.ReusedCount = 0
	coverage.ConstructionDefinition = "created counts tickets constructed specifically for these two priority candidates before this offline closure build; reused counts pre-existing unrelated issues adopted for a candidate"
	coverage.ClosureLeftovers = coverage.SelectedUnmapped + coverage.Unclassified + coverage.UnmappedActionable

	receiptBytes, _ := json.Marshal(forge.Receipt)
	completeSources := make([]string, 0, len(forge.Receipt.Sources))
	for _, source := range forge.Receipt.Sources {
		if source.Status == studyforge.StatusComplete {
			completeSources = append(completeSources, source.Name)
		}
	}
	adjacencyReceipt := foldAdjacency(adjacency)
	relatedRecords, relatedComplete, err := foldRelatedForge(adjacency)
	if err != nil {
		return Ledger{}, err
	}
	return Ledger{
		Schema:      Schema,
		ParentIssue: ParentIssue,
		Sources: Sources{
			Priority:  SourceReceipt{Path: displayPath(opts.PriorityPath), SHA256: digest(priorityBytes), Schema: priority.Schema, Revision: priority.Source.SourceRevision, Cutoff: priority.Source.Cutoff, RecordCount: len(priority.Candidates)},
			Join:      SourceReceipt{Path: displayPath(opts.JoinPath), SHA256: digest(joinBytes), Schema: join.Schema, Revision: join.SourceRevision, Cutoff: join.Cutoff, RecordCount: len(join.Joins)},
			Forge:     SourceReceipt{Path: displayPath(opts.ForgePath), SHA256: digest(forgeBytes), Schema: forge.Schema, Revision: forge.Receipt.Revision, Cutoff: forge.Receipt.Cutoff, RecordCount: len(forge.Records), ReceiptSHA256: digest(receiptBytes), IndexChecksum: forge.Receipt.IndexChecksum, CaptureStatus: forge.Receipt.Status},
			Adjacency: SourceReceipt{Path: displayPath(opts.AdjacencyPath), SHA256: digest(adjacencyBytes), Schema: adjacency.Schema, RecordCount: len(adjacency.Members)},
		},
		Coverage:  coverage,
		Tickets:   tickets,
		Queue:     queue,
		Adjacency: adjacencyReceipt,
		Capture: CaptureReceipt{
			Repository: forge.Receipt.Repository, Revision: forge.Receipt.Revision, Cutoff: forge.Receipt.Cutoff,
			Status: forge.Receipt.Status, RecordCount: len(forge.Records), CompleteSources: completeSources,
			IndexChecksum: forge.Receipt.IndexChecksum, ReceiptSHA256: digest(receiptBytes),
		},
		CorpusReceipt: CorpusReceipt{
			VLLMRecords: classification.Input.RecordCount, VLLMRevision: classification.Input.Revision,
			VLLMCutoff: classification.Input.Cutoff, VLLMIndexChecksum: classification.Input.IndexChecksum,
			RelatedRepositories: len(adjacency.Members), RelatedForgeComplete: relatedComplete,
			RelatedRecords: relatedRecords, FAKRecords: len(forge.Records),
		},
		SamplingEvidence:   samples,
		RefreshObligations: append(append([]string(nil), refreshObligations...), "classification source checksum: "+digest(classificationBytes)),
	}, nil
}

func buildCoverage(join joinLedger) (Coverage, map[string]joinCluster, []SampleEvidence, error) {
	allowed := map[string]bool{"landed": true, "open_exact": true, "partial": true, "conflict": true, "obsolete": true, "uncovered": true}
	counts := map[string]*DispositionCount{}
	for _, name := range []string{"landed", "open_exact", "partial", "conflict", "obsolete", "uncovered"} {
		counts[name] = &DispositionCount{Disposition: name}
	}
	seen := map[string]bool{}
	uncovered := map[string]joinCluster{}
	sampleByID := map[string]joinCluster{}
	coverage := Coverage{JoinClusters: len(join.Joins)}
	for _, cluster := range join.Joins {
		if cluster.ClusterID == "" || seen[cluster.ClusterID] {
			return Coverage{}, nil, nil, invalidf("join cluster IDs must be non-empty and unique")
		}
		seen[cluster.ClusterID] = true
		if !allowed[cluster.Disposition] {
			coverage.Unclassified++
			continue
		}
		counts[cluster.Disposition].Count++
		if cluster.Actionable {
			coverage.ActionableClusters++
			counts[cluster.Disposition].Actionable++
		}
		if cluster.Actionable && cluster.Disposition == "uncovered" {
			if len(cluster.Artifacts) != 0 {
				return Coverage{}, nil, nil, invalidf("uncovered cluster %s has prior artifacts", cluster.ClusterID)
			}
			uncovered[cluster.ClusterID] = cluster
		}
		sampleByID[cluster.ClusterID] = cluster
	}
	coverage.UncoveredActionable = len(uncovered)
	for _, name := range []string{"landed", "open_exact", "partial", "conflict", "obsolete", "uncovered"} {
		coverage.DispositionCounts = append(coverage.DispositionCounts, *counts[name])
	}
	if len(uncovered) != 5 {
		return Coverage{}, nil, nil, invalidf("expected five uncovered actionable clusters, got %d", len(uncovered))
	}
	samples := make([]SampleEvidence, 0, len(sampleClusterIDs))
	for _, id := range sampleClusterIDs {
		cluster, ok := sampleByID[id]
		if !ok {
			return Coverage{}, nil, nil, invalidf("sampling cluster %s missing", id)
		}
		samples = append(samples, sampleEvidence(cluster))
	}
	return coverage, uncovered, samples, nil
}

func buildTickets(priority studyprio.Ledger, forge studyforge.Corpus, uncovered map[string]joinCluster) ([]Ticket, []QueueEntry, error) {
	if len(priority.Candidates) != 2 || len(priority.Queue) != 2 {
		return nil, nil, invalidf("bounded closure requires two candidates and two queue entries")
	}
	candidates := map[string]studyprio.Candidate{}
	for _, candidate := range priority.Candidates {
		if _, ok := candidates[candidate.ID]; ok {
			return nil, nil, invalidf("duplicate candidate %s", candidate.ID)
		}
		candidates[candidate.ID] = candidate
	}
	issues := map[int]studyforge.Record{}
	for _, record := range forge.Records {
		if record.Kind != "issue" {
			continue
		}
		if _, wanted := issueNumberWanted(record.Number); !wanted {
			continue
		}
		if _, duplicate := issues[record.Number]; duplicate {
			return nil, nil, invalidf("captured issue #%d appears more than once", record.Number)
		}
		issues[record.Number] = record
	}

	mappedClusters := map[string]string{}
	usedIssues := map[int]string{}
	ticketBySelection := map[string]Ticket{}
	for _, q := range priority.Queue {
		candidate, ok := candidates[q.CandidateID]
		if !ok {
			return nil, nil, invalidf("queue candidate %s missing", q.CandidateID)
		}
		number, ok := issueMapping[candidate.ID]
		if !ok {
			return nil, nil, invalidf("candidate %s has no bounded issue mapping", candidate.ID)
		}
		if prior := usedIssues[number]; prior != "" {
			return nil, nil, invalidf("issue #%d reused by %s and %s", number, prior, candidate.ID)
		}
		usedIssues[number] = candidate.ID
		issue, ok := issues[number]
		if !ok {
			return nil, nil, invalidf("mapped issue #%d missing from captured forge corpus", number)
		}
		if err := validateIssue(candidate, q, issue); err != nil {
			return nil, nil, err
		}
		for _, mapping := range candidate.SourceMappings {
			if _, ok := uncovered[mapping.ClusterID]; !ok {
				return nil, nil, invalidf("candidate %s maps non-uncovered cluster %s", candidate.ID, mapping.ClusterID)
			}
			if prior := mappedClusters[mapping.ClusterID]; prior != "" {
				return nil, nil, invalidf("cluster %s mapped by %s and %s", mapping.ClusterID, prior, candidate.ID)
			}
			mappedClusters[mapping.ClusterID] = candidate.ID
		}
		recordBytes, _ := json.Marshal(issue)
		ticketBySelection[candidate.ID] = Ticket{
			CandidateID: candidate.ID, WorkItemTitle: candidate.Title, Issue: number, URL: issue.URL,
			State: issue.State, Title: issue.Title, RecordSHA256: digest(recordBytes), CreatedAt: issue.CreatedAt,
			UpdatedAt: issue.UpdatedAt, Labels: append([]string(nil), issue.Labels...), Horizon: candidate.Horizon,
			QueueRank: q.Rank, Score: candidate.Score, Centrality: candidate.Centrality,
			Dependencies: append([]string(nil), candidate.Dependencies...), SourceClusters: append([]studyprio.SourceMapping(nil), candidate.SourceMappings...),
			PurposeBuilt: true, ReusedExistingWork: false,
			NativeConstraint: fmt.Sprintf("engine=%s model=%s fallback_allowed=%t", candidate.Execution.Engine, candidate.Execution.DefaultModel, candidate.Execution.FallbackAllowed),
		}
	}
	if len(mappedClusters) != len(uncovered) {
		return nil, nil, invalidf("%d uncovered actionable clusters remain unmapped", len(uncovered)-len(mappedClusters))
	}

	tickets := make([]Ticket, 0, len(priority.Queue))
	queue := make([]QueueEntry, 0, len(priority.Queue))
	for i, q := range priority.Queue {
		if q.Rank != i+1 {
			return nil, nil, invalidf("priority queue rank drift at %s", q.CandidateID)
		}
		ticket := ticketBySelection[q.CandidateID]
		tickets = append(tickets, ticket)
		queue = append(queue, QueueEntry{Rank: q.Rank, CandidateID: q.CandidateID, Issue: ticket.Issue, Horizon: q.Horizon, Dependencies: append([]string(nil), q.Dependencies...)})
	}
	if err := validateDependencyOrder(queue); err != nil {
		return nil, nil, err
	}
	return tickets, queue, nil
}

func validateIssue(candidate studyprio.Candidate, q studyprio.QueueEntry, issue studyforge.Record) error {
	if issue.State != "open" {
		return invalidf("mapped issue #%d is %s, not open", issue.Number, issue.State)
	}
	if strings.TrimSpace(issue.Title) == "" || strings.TrimSpace(issue.Body) == "" {
		return invalidf("mapped issue #%d has empty title or body", issue.Number)
	}
	combined := issue.Title + "\n" + issue.Body
	if !strings.Contains(combined, fmt.Sprintf("#%d", ParentIssue)) || !strings.Contains(combined, candidate.ID) {
		return invalidf("mapped issue #%d lacks parent #%d or candidate %s", issue.Number, ParentIssue, candidate.ID)
	}
	for _, mapping := range candidate.SourceMappings {
		if strings.Count(combined, mapping.ClusterID) != 1 {
			return invalidf("mapped issue #%d must contain source cluster %s exactly once", issue.Number, mapping.ClusterID)
		}
	}
	for _, section := range requiredSections {
		body, count := markdownSection(issue.Body, section)
		if count != 1 || strings.TrimSpace(body) == "" {
			return invalidf("mapped issue #%d lacks one non-empty %s section", issue.Number, section)
		}
	}
	if candidate.Execution.Engine == "fak-native" {
		if !strings.Contains(combined, "fak-native") || !strings.Contains(combined, candidate.Execution.DefaultModel) {
			return invalidf("mapped issue #%d lacks native engine/model constraint", issue.Number)
		}
	}
	requiredLabels := []string{"gen/" + candidate.Horizon, "performance", "priority/P1"}
	labelSet := map[string]bool{}
	for _, label := range issue.Labels {
		if label == "" || labelSet[label] {
			return invalidf("mapped issue #%d has empty or duplicate labels", issue.Number)
		}
		labelSet[label] = true
	}
	for _, label := range requiredLabels {
		if !labelSet[label] {
			return invalidf("mapped issue #%d lacks required label %s", issue.Number, label)
		}
	}
	if q.Horizon != candidate.Horizon || q.Score != candidate.Score || !equalStrings(q.Dependencies, candidate.Dependencies) {
		return invalidf("candidate %s queue metadata drift", candidate.ID)
	}
	return nil
}

func markdownSection(body, name string) (string, int) {
	marker := "## " + name
	lines := strings.Split(body, "\n")
	count := 0
	var captured []string
	active := false
	for _, line := range lines {
		if strings.HasPrefix(line, "## ") {
			active = line == marker
			if active {
				count++
				captured = captured[:0]
			}
			continue
		}
		if active {
			captured = append(captured, line)
		}
	}
	return strings.Join(captured, "\n"), count
}

func validateDependencyOrder(queue []QueueEntry) error {
	rank := map[string]int{}
	issues := map[int]bool{}
	for index, entry := range queue {
		if entry.CandidateID == "" || rank[entry.CandidateID] != 0 || entry.Issue <= 0 || issues[entry.Issue] {
			return invalidf("queue contains duplicate or empty candidate/issue")
		}
		if entry.Rank != index+1 {
			return invalidf("queue rank drift for %s: got %d want %d", entry.CandidateID, entry.Rank, index+1)
		}
		rank[entry.CandidateID] = entry.Rank
		issues[entry.Issue] = true
	}
	for _, entry := range queue {
		for _, dependency := range entry.Dependencies {
			if rank[dependency] == 0 || rank[dependency] >= entry.Rank {
				return invalidf("queue dependency %s does not precede %s", dependency, entry.CandidateID)
			}
		}
	}
	return nil
}

func foldAdjacency(adjacency adjacencyManifest) AdjacencyReceipt {
	out := AdjacencyReceipt{ID: adjacency.ID, MemberCount: len(adjacency.Members)}
	for _, member := range adjacency.Members {
		repository := member.Repository.Owner + "/" + member.Repository.Repo
		for _, receipt := range member.SourceClassReceipts {
			row := AdjacencyClass{Repository: repository, Class: receipt.Class, Status: receipt.Status, Notes: receipt.Notes}
			switch receipt.Status {
			case "complete":
				out.CompleteClassCount++
			case "partial":
				out.PartialClassCount++
				out.PartialClasses = append(out.PartialClasses, row)
			case "missing", "inaccessible":
				out.InaccessibleCount++
				out.InaccessibleClasses = append(out.InaccessibleClasses, row)
			}
		}
	}
	sort.Slice(out.PartialClasses, func(i, j int) bool {
		return out.PartialClasses[i].Repository+out.PartialClasses[i].Class < out.PartialClasses[j].Repository+out.PartialClasses[j].Class
	})
	sort.Slice(out.InaccessibleClasses, func(i, j int) bool {
		return out.InaccessibleClasses[i].Repository+out.InaccessibleClasses[i].Class < out.InaccessibleClasses[j].Repository+out.InaccessibleClasses[j].Class
	})
	return out
}

var receiptRecordsPattern = regexp.MustCompile(`(?:^|;)\s*records:(\d+)(?:;|$)`)

func foldRelatedForge(adjacency adjacencyManifest) (records, complete int, err error) {
	for _, member := range adjacency.Members {
		found := false
		for _, receipt := range member.SourceClassReceipts {
			if receipt.Class != "forge_history" {
				continue
			}
			found = true
			if receipt.Status != "complete" {
				return 0, 0, invalidf("related forge capture is not complete for %s/%s", member.Repository.Owner, member.Repository.Repo)
			}
			match := receiptRecordsPattern.FindStringSubmatch(receipt.TerminalReceipt)
			if len(match) != 2 {
				return 0, 0, invalidf("related forge receipt lacks record count for %s/%s", member.Repository.Owner, member.Repository.Repo)
			}
			count, parseErr := strconv.Atoi(match[1])
			if parseErr != nil {
				return 0, 0, invalidf("related forge record count invalid for %s/%s", member.Repository.Owner, member.Repository.Repo)
			}
			records += count
			complete++
		}
		if !found {
			return 0, 0, invalidf("related forge receipt missing for %s/%s", member.Repository.Owner, member.Repository.Repo)
		}
	}
	return records, complete, nil
}

func sampleEvidence(cluster joinCluster) SampleEvidence {
	return SampleEvidence{
		ClusterID: cluster.ClusterID, Disposition: cluster.Disposition, Actionable: cluster.Actionable,
		ArtifactCount: len(cluster.Artifacts), Confidence: cluster.Confidence, ManualReview: cluster.ManualReview,
		ManualReason: cluster.ManualReason, MembersSHA256: cluster.MembersSHA256, EvidenceSHA256: cluster.Evidence.Digest,
	}
}

func issueNumberWanted(number int) (string, bool) {
	for candidate, mapped := range issueMapping {
		if number == mapped {
			return candidate, true
		}
	}
	return "", false
}

func readJSON[T any](path string) (T, []byte, error) {
	var out T
	if path == "" {
		return out, nil, invalidf("source path empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return out, nil, fmt.Errorf("studytickets: read %s: %w", path, err)
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return out, nil, invalidf("decode %s: %v", path, err)
	}
	return out, data, nil
}

func displayPath(path string) string {
	path = filepath.ToSlash(filepath.Clean(path))
	if filepath.IsAbs(path) {
		return filepath.Base(path)
	}
	for strings.HasPrefix(path, "../") {
		path = strings.TrimPrefix(path, "../")
	}
	return strings.TrimPrefix(path, "./")
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func invalidf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, args...))
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

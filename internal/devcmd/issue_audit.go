package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/closureaudit"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

var issueAuditCommitRE = regexp.MustCompile(`^[0-9a-fA-F]{7,64}$`)

func runIssueAudit(stdout, stderr io.Writer, argv []string) int {
	return runIssueAuditWith(stdout, stderr, argv, nil, nil)
}

func runIssueAuditWith(stdout, stderr io.Writer, argv []string, injectedFetcher modelroute.IssueAuditFetcher, injectedReviewer modelroute.IssueAuditReviewer) int {
	fs := flag.NewFlagSet("issue audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issueNumber := fs.Int("issue", 0, "closed GitHub issue number to audit")
	authorManifestPath := fs.String("author-manifest", "", "path to a fak-crossaudit-author/v1 JSON manifest")
	auditorTarget := fs.String("auditor", "", "auditor identity as PROVIDER/FAMILY/MODEL")
	auditorWeights := fs.String("auditor-weights", "", "auditor weights revision when known")
	auditorReasoning := fs.String("auditor-reasoning", "", "auditor reasoning posture")
	auditorEndpoint := fs.String("auditor-endpoint", firstNonEmpty(os.Getenv("FAK_AUDIT_ENDPOINT"), os.Getenv("FAK_REVIEW_ENDPOINT"), "http://127.0.0.1:8080/v1"), "OpenAI-compatible endpoint for the auditor")
	auditorAPIKeyEnv := fs.String("auditor-api-key-env", os.Getenv("FAK_REVIEW_API_KEY_ENV"), "environment variable containing the auditor API key")
	auditorDriver := fs.String("auditor-driver", "http", "auditor transport: http, codex, or claude")
	auditorTimeout := fs.Duration("auditor-timeout", 10*time.Minute, "maximum auditor inference time")
	identityRosterPath := fs.String("identity-roster", "", "authoritative fak-audit-identity-roster/v1 JSON file")
	repo := fs.String("repo", "", "GitHub OWNER/REPO (default: current repository)")
	bundleOnly := fs.Bool("bundle-only", false, "fetch and emit the bounded credential-free IssueAuditBundle without calling an auditor")
	bundleCommit := fs.String("bundle-commit", "", "explicit resolving commit SHA for --bundle-only when GitHub has no closing commit event")
	asJSON := fs.Bool("json", false, "emit the full typed JSON receipt")
	ledgerPath := fs.String("ledger", "", "append the verified receipt to this hash-chained ledger")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue audit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *issueNumber <= 0 {
		fmt.Fprintln(stderr, "fak-dev issue audit: --issue is required")
		return 2
	}
	if *auditorTimeout <= 0 {
		fmt.Fprintln(stderr, "fak-dev issue audit: --auditor-timeout must be positive")
		return 2
	}
	if *bundleOnly {
		if injectedFetcher == nil {
			injectedFetcher = &githubIssueAuditFetcher{repo: strings.TrimSpace(*repo), commitRef: strings.TrimSpace(*bundleCommit), runner: defaultIssueAuditRunner}
		}
		ctx, cancel := context.WithTimeout(context.Background(), *auditorTimeout)
		defer cancel()
		evidence, err := injectedFetcher.FetchIssueAuditEvidence(ctx, *issueNumber)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue audit: fetch bundle evidence: %v\n", err)
			return 1
		}
		bundle, err := modelroute.BuildIssueAuditBundle(evidence, modelroute.IssueAuditBundleOptions{})
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if encodeErr := enc.Encode(bundle); encodeErr != nil {
			fmt.Fprintf(stderr, "fak-dev issue audit: encode bundle: %v\n", encodeErr)
			return 1
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue audit: %v\n", err)
			return 3
		}
		return 0
	}
	if strings.TrimSpace(*authorManifestPath) == "" || strings.TrimSpace(*auditorTarget) == "" || strings.TrimSpace(*identityRosterPath) == "" {
		fmt.Fprintln(stderr, "fak-dev issue audit: --issue, --author-manifest, --auditor PROVIDER/FAMILY/MODEL, and --identity-roster are required")
		return 2
	}

	manifest, err := loadCrossAuditAuthorManifest(*authorManifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue audit: %v\n", err)
		return 2
	}
	roster, err := loadCrossAuditIdentityRoster(*identityRosterPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue audit: %v\n", err)
		return 2
	}
	auditor, err := parseCrossAuditAuditorTarget(*auditorTarget)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue audit: %v\n", err)
		return 2
	}
	auditor.WeightsRevision = strings.TrimSpace(*auditorWeights)
	driver := strings.ToLower(strings.TrimSpace(*auditorDriver))
	httpAPIKey := ""
	if name := strings.TrimSpace(*auditorAPIKeyEnv); name != "" {
		httpAPIKey = strings.TrimSpace(os.Getenv(name))
	}
	switch driver {
	case "codex", "claude":
		effort := strings.ToLower(strings.TrimSpace(*auditorReasoning))
		if !validIssueAuditEffort(effort) {
			fmt.Fprintf(stderr, "fak-dev issue audit: --auditor-driver %s requires --auditor-reasoning low|medium|high|xhigh|max\n", driver)
			return 2
		}
		auditor.Harness = driver + "-cli"
		auditor.EndpointClass = "hosted-cli"
		auditor.AccountClass = "cli-auth"
		auditor.ReasoningPosture = effort
	case "http":
		if strings.TrimSpace(*auditorReasoning) != "" {
			fmt.Fprintln(stderr, "fak-dev issue audit: HTTP does not bind --auditor-reasoning; omit it (the request is stamped provider-default)")
			return 2
		}
		auditor.Harness = "openai-compatible-http"
		auditor.EndpointClass = issueAuditEndpointClass(*auditorEndpoint)
		auditor.ReasoningPosture = "provider-default"
		if httpAPIKey != "" {
			auditor.AccountClass = "api-key"
		} else if auditor.EndpointClass == "local-http" {
			auditor.AccountClass = "local-no-key"
		} else {
			auditor.AccountClass = "unauthenticated"
		}
	default:
		fmt.Fprintf(stderr, "fak-dev issue audit: --auditor-driver %q, want http, codex, or claude\n", driver)
		return 2
	}
	auditor.Driver = driver
	canonicalAuditor, err := modelroute.ValidateAuditDriverIdentity(driver, auditor, roster.Aliases)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue audit: %v\n", err)
		return 2
	}
	auditor = canonicalAuditor

	if injectedFetcher == nil {
		injectedFetcher = &githubIssueAuditFetcher{
			repo:      strings.TrimSpace(*repo),
			commitRef: strings.TrimSpace(manifest.CommitRange),
			runner:    defaultIssueAuditRunner,
		}
	}
	if injectedReviewer == nil {
		switch driver {
		case "http":
			injectedReviewer = newIssueAuditHTTPReviewer(auditor, strings.TrimSpace(*auditorEndpoint), httpAPIKey)
		case "codex", "claude":
			injectedReviewer = newIssueAuditCLIReviewer(driver, auditor)
		default:
			fmt.Fprintf(stderr, "fak-dev issue audit: --auditor-driver %q, want http, codex, or claude\n", *auditorDriver)
			return 2
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), *auditorTimeout)
	defer cancel()
	receipt, err := modelroute.AuditIssue(ctx, modelroute.IssueAuditRequest{
		IssueNumber: *issueNumber,
		Author:      manifest,
		Auditor:     auditor,
		IndependencePolicy: func() modelroute.AuditIndependencePolicy {
			policy := modelroute.DefaultAuditIndependencePolicy()
			policy.Aliases = append([]modelroute.AuditIdentityAlias(nil), roster.Aliases...)
			return policy
		}(),
		RequireObservedAuditorIdentity: driver == "http",
	}, injectedFetcher, injectedReviewer)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue audit: %v\n", err)
		if modelroute.IsIndependenceRefusal(err) {
			return 3
		}
		return 1
	}
	if err := receipt.Verify(); err != nil {
		fmt.Fprintf(stderr, "fak-dev issue audit: internal receipt verification failed: %v\n", err)
		return 1
	}
	var ledger *modelroute.AuditReceiptAppendResult
	if strings.TrimSpace(*ledgerPath) != "" {
		result, err := modelroute.AppendAuditReceiptLedger(*ledgerPath, receipt)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue audit: append receipt ledger: %v\n", err)
			return 1
		}
		ledger = &result
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		var payload any = receipt
		if ledger != nil {
			payload = issueAuditLedgerOutput{Receipt: receipt, Ledger: *ledger}
		}
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue audit: encode receipt: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "issue audit %s: issue #%d commit %s author %s/%s auditor %s/%s receipt %s\n",
			receipt.Verdict, receipt.Subject.IssueNumber, receipt.Subject.CommitSHA,
			receipt.Author.Provider, receipt.Author.Family, receipt.Auditor.Provider, receipt.Auditor.Family, receipt.ReceiptDigest)
		fmt.Fprintf(stdout, "reason: %s\n", receipt.Reason)
		if ledger != nil {
			fmt.Fprint(stdout, renderIssueAuditLedger(*ledger))
		}
	}
	if receipt.Verdict != modelroute.CrossAuditPass {
		return 1
	}
	return 0
}

type issueAuditLedgerOutput struct {
	Receipt modelroute.IssueAuditReceipt        `json:"receipt"`
	Ledger  modelroute.AuditReceiptAppendResult `json:"ledger"`
}

func renderIssueAuditLedger(result modelroute.AuditReceiptAppendResult) string {
	status := "appended"
	if result.Duplicate {
		status = "duplicate"
	}
	return fmt.Sprintf("ledger: %s rows=%d head_hash=%s\n", status, result.Cursor.Rows, result.Cursor.HeadHash)
}

func validIssueAuditEffort(effort string) bool {
	switch effort {
	case "low", "medium", "high", "xhigh", "max":
		return true
	default:
		return false
	}
}

func issueAuditEndpointClass(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return "invalid-http"
	}
	host := strings.TrimSpace(u.Hostname())
	if strings.EqualFold(host, "localhost") {
		return "local-http"
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return "local-http"
	}
	return "remote-http"
}

func loadCrossAuditAuthorManifest(path string) (modelroute.AuthorManifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return modelroute.AuthorManifest{}, fmt.Errorf("read --author-manifest: %w", err)
	}
	var manifest modelroute.AuthorManifest
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return modelroute.AuthorManifest{}, fmt.Errorf("parse --author-manifest: %w", err)
	}
	return manifest, nil
}

func loadCrossAuditIdentityRoster(path string) (modelroute.AuditIdentityRoster, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return modelroute.AuditIdentityRoster{}, fmt.Errorf("read --identity-roster: %w", err)
	}
	var roster modelroute.AuditIdentityRoster
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&roster); err != nil {
		return modelroute.AuditIdentityRoster{}, fmt.Errorf("parse --identity-roster: %w", err)
	}
	if err := roster.Validate(); err != nil {
		return modelroute.AuditIdentityRoster{}, err
	}
	return roster, nil
}

func parseCrossAuditAuditorTarget(raw string) (modelroute.ModelIdentity, error) {
	parts := strings.SplitN(strings.TrimSpace(raw), "/", 3)
	if len(parts) != 3 {
		return modelroute.ModelIdentity{}, fmt.Errorf("--auditor %q must be PROVIDER/FAMILY/MODEL", raw)
	}
	id := modelroute.ModelIdentity{
		Provider: strings.TrimSpace(parts[0]),
		Family:   strings.TrimSpace(parts[1]),
		Model:    strings.TrimSpace(parts[2]),
	}
	if id.Provider == "" || id.Family == "" || id.Model == "" {
		return modelroute.ModelIdentity{}, fmt.Errorf("--auditor %q must have non-empty provider, family, and model", raw)
	}
	return id, nil
}

type issueAuditCommandRunner func(context.Context, string, ...string) ([]byte, error)

func defaultIssueAuditRunner(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

type githubIssueAuditFetcher struct {
	repo      string
	commitRef string
	runner    issueAuditCommandRunner
}

func (f *githubIssueAuditFetcher) FetchIssueAuditEvidence(ctx context.Context, issue int) (modelroute.IssueAuditEvidence, error) {
	if f.runner == nil {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("issue audit command runner is nil")
	}
	out, err := f.runner(ctx, "gh", "repo", "view", "--json", "nameWithOwner")
	if err != nil {
		return modelroute.IssueAuditEvidence{}, err
	}
	var repoRow struct {
		NameWithOwner string `json:"nameWithOwner"`
	}
	if err := json.Unmarshal(out, &repoRow); err != nil {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("decode gh repo view: %w", err)
	}
	currentRepo := strings.TrimSpace(repoRow.NameWithOwner)
	repo := strings.TrimSpace(f.repo)
	if repo == "" {
		repo = currentRepo
	} else if !strings.EqualFold(repo, currentRepo) {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("--repo %q does not match the current git repository %q; cross-repo git evidence is refused", repo, currentRepo)
	}
	if !validCrossAuditRepo(repo) {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("invalid GitHub repository %q", repo)
	}

	args := []string{"issue", "view", strconv.Itoa(issue), "--repo", repo, "--json", "number,title,body,state,closedAt,url,comments,closedByPullRequestsReferences"}
	out, err = f.runner(ctx, "gh", args...)
	if err != nil {
		return modelroute.IssueAuditEvidence{}, err
	}
	var issueRow struct {
		Number   int    `json:"number"`
		Title    string `json:"title"`
		Body     string `json:"body"`
		State    string `json:"state"`
		ClosedAt string `json:"closedAt"`
		URL      string `json:"url"`
		Comments []struct {
			ID        string `json:"id"`
			URL       string `json:"url"`
			Body      string `json:"body"`
			CreatedAt string `json:"createdAt"`
			UpdatedAt string `json:"updatedAt"`
			Author    struct {
				Login string `json:"login"`
			} `json:"author"`
		} `json:"comments"`
		ClosedByPullRequests []struct {
			Number int `json:"number"`
		} `json:"closedByPullRequestsReferences"`
	}
	if err := json.Unmarshal(out, &issueRow); err != nil {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("decode gh issue view: %w", err)
	}
	if strings.ToUpper(strings.TrimSpace(issueRow.State)) != "CLOSED" {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("issue #%d is %s, want CLOSED", issue, issueRow.State)
	}

	commit, source, err := f.closingCommit(ctx, repo, issue, issueRow.ClosedByPullRequests)
	if err != nil {
		return modelroute.IssueAuditEvidence{}, err
	}
	closing, err := f.readClosingCommitEvidence(ctx, commit)
	if err != nil {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("read closing evidence %s: %w", commit, err)
	}
	commit = closing.SHA
	comments := make([]modelroute.IssueAuditComment, 0, len(issueRow.Comments))
	for i, comment := range issueRow.Comments {
		id := strings.TrimSpace(comment.ID)
		if id == "" {
			id = fmt.Sprintf("comment-%04d", i+1)
		}
		comments = append(comments, modelroute.IssueAuditComment{
			ID: id, URL: comment.URL, Author: comment.Author.Login, CreatedAt: comment.CreatedAt, UpdatedAt: comment.UpdatedAt, Body: comment.Body,
		})
	}
	evidence := modelroute.IssueAuditEvidence{
		IssueNumber:    issueRow.Number,
		IssueURL:       issueRow.URL,
		Title:          issueRow.Title,
		Body:           issueRow.Body,
		Comments:       comments,
		State:          issueRow.State,
		ClosedAt:       issueRow.ClosedAt,
		CommitSHA:      commit,
		Diff:           closing.Patch,
		ClosingCommits: []modelroute.IssueAuditClosingCommit{closing},
		Evidence:       []modelroute.EvidenceRef{{Kind: "github-closure", Ref: source}},
	}
	populateIssueAuditEvidenceRefs(&evidence, repo)
	return evidence, nil
}

func (f *githubIssueAuditFetcher) readClosingCommitEvidence(ctx context.Context, commit string) (modelroute.IssueAuditClosingCommit, error) {
	parents, err := f.runner(ctx, "git", "rev-list", "--parents", "-n", "1", commit)
	if err != nil {
		return modelroute.IssueAuditClosingCommit{}, err
	}
	fields := strings.Fields(string(parents))
	if len(fields) < 2 {
		return modelroute.IssueAuditClosingCommit{}, fmt.Errorf("commit %s has no parent to diff", commit)
	}
	parent := fields[1]
	patch, err := f.runner(ctx, "git", "diff", "--no-ext-diff", "--binary", "--no-renames", parent, fields[0])
	if err != nil {
		return modelroute.IssueAuditClosingCommit{}, err
	}
	tree, err := f.runner(ctx, "git", "rev-parse", fields[0]+"^{tree}")
	if err != nil {
		return modelroute.IssueAuditClosingCommit{}, err
	}
	parentTree, err := f.runner(ctx, "git", "rev-parse", parent+"^{tree}")
	if err != nil {
		return modelroute.IssueAuditClosingCommit{}, err
	}
	pathsRaw, err := f.runner(ctx, "git", "diff", "--name-only", "-z", "--no-renames", parent, fields[0])
	if err != nil {
		return modelroute.IssueAuditClosingCommit{}, err
	}
	return modelroute.IssueAuditClosingCommit{
		SHA: fields[0], FirstParentSHA: parent, TreeOID: strings.TrimSpace(string(tree)), FirstParentTreeOID: strings.TrimSpace(string(parentTree)),
		Patch: string(patch), PatchSHA256: modelroute.IssueAuditContentDigest(string(patch)), ChangedPaths: splitIssueAuditNUL(pathsRaw),
	}, nil
}

func splitIssueAuditNUL(raw []byte) []string {
	parts := bytes.Split(raw, []byte{0})
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if path := strings.TrimSpace(string(part)); path != "" {
			out = append(out, path)
		}
	}
	return out
}

func populateIssueAuditEvidenceRefs(evidence *modelroute.IssueAuditEvidence, repo string) {
	if evidence == nil || len(evidence.ClosingCommits) == 0 {
		return
	}
	commit := evidence.ClosingCommits[0]
	evidence.CI = append(evidence.CI, modelroute.EvidenceRef{Kind: "github-check-runs", Ref: repo + "@" + commit.SHA})
	evidence.DOS = append(evidence.DOS, modelroute.EvidenceRef{Kind: "dos-commit-audit", Ref: "commit:" + commit.SHA})
	for _, path := range commit.ChangedPaths {
		normalized := strings.ReplaceAll(path, "\\", "/")
		if strings.HasSuffix(normalized, "_test.go") || strings.Contains(normalized, "/testdata/") {
			evidence.Tests = append(evidence.Tests, modelroute.EvidenceRef{Kind: "test-path", Ref: normalized})
		}
		if strings.HasPrefix(normalized, ".github/workflows/") {
			evidence.CI = append(evidence.CI, modelroute.EvidenceRef{Kind: "workflow-path", Ref: normalized})
		}
		if strings.Contains(normalized, "/testdata/") || strings.HasPrefix(normalized, "experiments/") || strings.HasSuffix(normalized, ".json") || strings.HasSuffix(normalized, ".jsonl") {
			evidence.Artifacts = append(evidence.Artifacts, modelroute.EvidenceRef{Kind: "artifact-path", Ref: normalized})
		}
	}
	for _, comment := range evidence.Comments {
		body := strings.ToLower(comment.Body)
		if strings.Contains(body, "audit finding") || strings.Contains(body, "[finding]") || strings.Contains(body, "verdict: refute") {
			ref := comment.URL
			if strings.TrimSpace(ref) == "" {
				ref = "comment:" + comment.ID
			}
			evidence.PriorFindings = append(evidence.PriorFindings, modelroute.EvidenceRef{Kind: "issue-comment-finding", Ref: ref})
		}
	}
}

func (f *githubIssueAuditFetcher) closingCommit(ctx context.Context, repo string, issue int, prs []struct {
	Number int `json:"number"`
}) (string, string, error) {
	if explicit := strings.TrimSpace(f.commitRef); explicit != "" {
		if !issueAuditCommitRE.MatchString(explicit) {
			return "", "", fmt.Errorf("author manifest commit_range %q must be one commit SHA in the first cross-audit spine", explicit)
		}
		if err := f.verifyResolvingCommit(ctx, issue, explicit); err != nil {
			return "", "", err
		}
		return explicit, "author-manifest:" + explicit, nil
	}
	out, err := f.runner(ctx, "gh", "api", fmt.Sprintf("repos/%s/issues/%d/events?per_page=100", repo, issue))
	if err != nil {
		return "", "", err
	}
	var events []struct {
		Event     string `json:"event"`
		CommitID  string `json:"commit_id"`
		CreatedAt string `json:"created_at"`
	}
	if err := json.Unmarshal(out, &events); err != nil {
		return "", "", fmt.Errorf("decode issue events: %w", err)
	}
	closedCommits := map[string]string{}
	for _, event := range events {
		commit := strings.TrimSpace(event.CommitID)
		if event.Event != "closed" || commit == "" {
			continue
		}
		closedCommits[commit] = fmt.Sprintf("github-event:closed:%s@%s", commit, event.CreatedAt)
	}
	if len(closedCommits) > 1 {
		return "", "", fmt.Errorf("issue #%d has %d distinct closed-event commits; closing diff is ambiguous", issue, len(closedCommits))
	}
	for commit, source := range closedCommits {
		if !issueAuditCommitRE.MatchString(commit) {
			return "", "", fmt.Errorf("issue #%d closed-event commit %q is invalid", issue, commit)
		}
		return commit, source, nil
	}
	if len(prs) > 1 {
		return "", "", fmt.Errorf("issue #%d has %d closing pull requests; closing diff is ambiguous", issue, len(prs))
	}
	if len(prs) == 1 {
		pr := prs[0].Number
		out, err := f.runner(ctx, "gh", "pr", "view", strconv.Itoa(pr), "--repo", repo, "--json", "mergeCommit")
		if err != nil {
			return "", "", err
		}
		var row struct {
			MergeCommit struct {
				OID string `json:"oid"`
			} `json:"mergeCommit"`
		}
		if err := json.Unmarshal(out, &row); err != nil {
			return "", "", fmt.Errorf("decode closing PR #%d: %w", pr, err)
		}
		commit := strings.TrimSpace(row.MergeCommit.OID)
		if !issueAuditCommitRE.MatchString(commit) {
			return "", "", fmt.Errorf("issue #%d closing PR #%d has no usable merge commit", issue, pr)
		}
		return commit, fmt.Sprintf("closing-pr:%d:%s", pr, commit), nil
	}
	return "", "", fmt.Errorf("issue #%d has no unambiguous closed-event/PR commit; set author manifest commit_range to one resolving commit SHA", issue)
}

func (f *githubIssueAuditFetcher) verifyResolvingCommit(ctx context.Context, issue int, commit string) error {
	out, err := f.runner(ctx, "git", "show", "-s", "--format=%s%x1f%b", commit)
	if err != nil {
		return fmt.Errorf("read author-manifest commit %s: %w", commit, err)
	}
	parts := strings.SplitN(string(out), "\x1f", 2)
	subject, body := strings.TrimSpace(parts[0]), ""
	if len(parts) == 2 {
		body = strings.TrimSpace(parts[1])
	}
	if closureaudit.ClassifyRefs(subject, body)[issue] != closureaudit.Resolving {
		return fmt.Errorf("author-manifest commit %s does not resolve issue #%d", commit, issue)
	}
	return nil
}

func validCrossAuditRepo(repo string) bool {
	parts := strings.Split(repo, "/")
	if len(parts) != 2 {
		return false
	}
	for _, part := range parts {
		if part == "" || strings.ContainsAny(part, "\\?#") {
			return false
		}
	}
	return true
}

type issueAuditHTTPReviewer struct {
	identity modelroute.ModelIdentity
	endpoint string
	apiKey   string
}

func newIssueAuditHTTPReviewer(identity modelroute.ModelIdentity, endpoint, apiKey string) modelroute.IssueAuditReviewer {
	return &issueAuditHTTPReviewer{identity: identity, endpoint: endpoint, apiKey: apiKey}
}

func (r *issueAuditHTTPReviewer) ReviewIssue(ctx context.Context, req modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
	if err := req.Verify(); err != nil {
		return modelroute.IssueAuditReviewResult{}, err
	}
	client := agent.NewHTTPPlanner(r.endpoint, r.identity.Model, r.apiKey)
	client.MaxTokens = 512
	client.Temperature = 0
	temp := 0.0
	completion, err := client.Complete(ctx, []agent.Message{
		{Role: agent.RoleSystem, Content: req.TrustedInstruction.Content},
		{Role: agent.RoleUser, Content: req.UntrustedEvidence.Content},
	}, nil, agent.WithMaxTokens(512), agent.WithTemperature(&temp))
	if err != nil {
		return modelroute.IssueAuditReviewResult{}, err
	}
	if completion == nil {
		return modelroute.IssueAuditReviewResult{}, fmt.Errorf("auditor returned nil completion")
	}
	var result modelroute.IssueAuditReviewResult
	if err := json.Unmarshal([]byte(stripJSONFence(completion.Message.Content)), &result); err != nil {
		return modelroute.IssueAuditReviewResult{}, fmt.Errorf("decode auditor result: %w", err)
	}
	result.ObservedAuditor = &modelroute.AuditIdentity{Model: strings.TrimSpace(completion.Model)}
	return result, nil
}

type issueAuditCLIReviewer struct {
	driver   string
	identity modelroute.ModelIdentity
	runner   issueAuditProcessRunner
}

func newIssueAuditCLIReviewer(driver string, identity modelroute.ModelIdentity) modelroute.IssueAuditReviewer {
	return &issueAuditCLIReviewer{driver: driver, identity: identity, runner: runIssueAuditProcess}
}

const issueAuditResultSchema = `{"type":"object","additionalProperties":false,"properties":{"verdict":{"type":"string","enum":["PASS","REFUTE","INCONCLUSIVE"]},"reason":{"type":"string"},"evidence_refs":{"type":"array","items":{"type":"string"}}},"required":["verdict","reason","evidence_refs"]}`

func (r *issueAuditCLIReviewer) ReviewIssue(ctx context.Context, req modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
	if err := req.Verify(); err != nil {
		return modelroute.IssueAuditReviewResult{}, err
	}
	runner := r.runner
	if runner == nil {
		runner = runIssueAuditProcess
	}
	switch r.driver {
	case "claude":
		out, err := runner(ctx, req.Prompt, "claude",
			"-p", "--model", r.identity.Model, "--output-format", "json",
			"--json-schema", issueAuditResultSchema, "--effort", r.identity.ReasoningPosture, "--tools", "")
		if err != nil {
			return modelroute.IssueAuditReviewResult{}, err
		}
		return parseIssueAuditReviewerOutput(out)
	case "codex":
		dir, err := os.MkdirTemp("", "fak-issue-audit-")
		if err != nil {
			return modelroute.IssueAuditReviewResult{}, fmt.Errorf("create codex audit scratch: %w", err)
		}
		defer os.RemoveAll(dir)
		schemaPath := dir + string(os.PathSeparator) + "schema.json"
		outputPath := dir + string(os.PathSeparator) + "result.json"
		if err := os.WriteFile(schemaPath, []byte(issueAuditResultSchema), 0o600); err != nil {
			return modelroute.IssueAuditReviewResult{}, fmt.Errorf("write codex audit schema: %w", err)
		}
		args := []string{
			"exec", "--ephemeral", "--ignore-rules", "--skip-git-repo-check",
			"-s", "read-only", "-C", dir, "-m", r.identity.Model,
			"--output-schema", schemaPath, "-o", outputPath,
		}
		if reasoning := strings.TrimSpace(r.identity.ReasoningPosture); reasoning != "" {
			args = append(args, "-c", fmt.Sprintf("model_reasoning_effort=%q", reasoning))
		}
		args = append(args, "-")
		if _, err := runner(ctx, req.Prompt, "codex", args...); err != nil {
			return modelroute.IssueAuditReviewResult{}, err
		}
		out, err := os.ReadFile(outputPath)
		if err != nil {
			return modelroute.IssueAuditReviewResult{}, fmt.Errorf("read codex auditor result: %w", err)
		}
		return parseIssueAuditReviewerOutput(out)
	default:
		return modelroute.IssueAuditReviewResult{}, fmt.Errorf("unsupported issue audit CLI driver %q", r.driver)
	}
}

type issueAuditProcessRunner func(context.Context, string, string, ...string) ([]byte, error)

func runIssueAuditProcess(ctx context.Context, stdin, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdin = strings.NewReader(stdin)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%s auditor: %w: %s", name, err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func parseIssueAuditReviewerOutput(out []byte) (modelroute.IssueAuditReviewResult, error) {
	trimmed := []byte(strings.TrimSpace(stripJSONFence(string(out))))
	var direct modelroute.IssueAuditReviewResult
	if err := json.Unmarshal(trimmed, &direct); err == nil && strings.TrimSpace(string(direct.Verdict)) != "" {
		return direct, nil
	}
	var envelope struct {
		StructuredOutput json.RawMessage `json:"structured_output"`
		Result           json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(trimmed, &envelope); err != nil {
		return modelroute.IssueAuditReviewResult{}, fmt.Errorf("decode CLI auditor envelope: %w", err)
	}
	for _, raw := range []json.RawMessage{envelope.StructuredOutput, envelope.Result} {
		if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
			continue
		}
		candidate := raw
		if len(candidate) > 0 && candidate[0] == '"' {
			var text string
			if err := json.Unmarshal(candidate, &text); err != nil {
				continue
			}
			candidate = []byte(strings.TrimSpace(stripJSONFence(text)))
		}
		if err := json.Unmarshal(candidate, &direct); err == nil && strings.TrimSpace(string(direct.Verdict)) != "" {
			return direct, nil
		}
	}
	return modelroute.IssueAuditReviewResult{}, fmt.Errorf("CLI auditor returned no structured verdict")
}

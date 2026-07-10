package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
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
	auditorHarness := fs.String("auditor-harness", "fak issue audit", "auditor harness provenance")
	auditorWeights := fs.String("auditor-weights", "", "auditor weights revision when known")
	auditorAccountClass := fs.String("auditor-account-class", "", "auditor account class")
	auditorEndpointClass := fs.String("auditor-endpoint-class", "openai-compatible", "auditor endpoint class")
	auditorReasoning := fs.String("auditor-reasoning", "", "auditor reasoning posture")
	auditorEndpoint := fs.String("auditor-endpoint", firstNonEmpty(os.Getenv("FAK_AUDIT_ENDPOINT"), os.Getenv("FAK_REVIEW_ENDPOINT"), "http://127.0.0.1:8080/v1"), "OpenAI-compatible endpoint for the auditor")
	auditorAPIKeyEnv := fs.String("auditor-api-key-env", os.Getenv("FAK_REVIEW_API_KEY_ENV"), "environment variable containing the auditor API key")
	auditorDriver := fs.String("auditor-driver", "http", "auditor transport: http, codex, or claude")
	auditorTimeout := fs.Duration("auditor-timeout", 10*time.Minute, "maximum auditor inference time")
	repo := fs.String("repo", "", "GitHub OWNER/REPO (default: current repository)")
	asJSON := fs.Bool("json", false, "emit the full typed JSON receipt")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak issue audit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *issueNumber <= 0 || strings.TrimSpace(*authorManifestPath) == "" || strings.TrimSpace(*auditorTarget) == "" {
		fmt.Fprintln(stderr, "fak issue audit: --issue, --author-manifest, and --auditor PROVIDER/FAMILY/MODEL are required")
		return 2
	}

	manifest, err := loadCrossAuditAuthorManifest(*authorManifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue audit: %v\n", err)
		return 2
	}
	auditor, err := parseCrossAuditAuditorTarget(*auditorTarget)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue audit: %v\n", err)
		return 2
	}
	auditor.Harness = strings.TrimSpace(*auditorHarness)
	auditor.WeightsRevision = strings.TrimSpace(*auditorWeights)
	auditor.AccountClass = strings.TrimSpace(*auditorAccountClass)
	auditor.EndpointClass = strings.TrimSpace(*auditorEndpointClass)
	auditor.ReasoningPosture = strings.TrimSpace(*auditorReasoning)
	driver := strings.ToLower(strings.TrimSpace(*auditorDriver))
	auditor.Driver = driver
	if err := validateIssueAuditDriverIdentity(driver, auditor); err != nil {
		fmt.Fprintf(stderr, "fak issue audit: %v\n", err)
		return 2
	}

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
			apiKey := ""
			if name := strings.TrimSpace(*auditorAPIKeyEnv); name != "" {
				apiKey = strings.TrimSpace(os.Getenv(name))
			}
			injectedReviewer = newIssueAuditHTTPReviewer(auditor, strings.TrimSpace(*auditorEndpoint), apiKey)
		case "codex", "claude":
			injectedReviewer = newIssueAuditCLIReviewer(driver, auditor)
		default:
			fmt.Fprintf(stderr, "fak issue audit: --auditor-driver %q, want http, codex, or claude\n", *auditorDriver)
			return 2
		}
	}

	if *auditorTimeout <= 0 {
		fmt.Fprintln(stderr, "fak issue audit: --auditor-timeout must be positive")
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), *auditorTimeout)
	defer cancel()
	receipt, err := modelroute.AuditIssue(ctx, modelroute.IssueAuditRequest{
		IssueNumber: *issueNumber,
		Author:      manifest,
		Auditor:     auditor,
	}, injectedFetcher, injectedReviewer)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue audit: %v\n", err)
		if modelroute.IsIndependenceRefusal(err) {
			return 3
		}
		return 1
	}
	if err := receipt.Verify(); err != nil {
		fmt.Fprintf(stderr, "fak issue audit: internal receipt verification failed: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(receipt); err != nil {
			fmt.Fprintf(stderr, "fak issue audit: encode receipt: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "issue audit %s: issue #%d commit %s author %s/%s auditor %s/%s receipt %s\n",
			receipt.Verdict, receipt.Subject.IssueNumber, receipt.Subject.CommitSHA,
			receipt.Author.Provider, receipt.Author.Family, receipt.Auditor.Provider, receipt.Auditor.Family, receipt.ReceiptDigest)
		fmt.Fprintf(stdout, "reason: %s\n", receipt.Reason)
	}
	if receipt.Verdict != modelroute.CrossAuditPass {
		return 1
	}
	return 0
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

func validateIssueAuditDriverIdentity(driver string, id modelroute.ModelIdentity) error {
	provider := strings.ToLower(strings.TrimSpace(id.Provider))
	family := strings.ToLower(strings.TrimSpace(id.Family))
	model := strings.ToLower(strings.TrimSpace(id.Model))
	switch driver {
	case "http":
		return nil
	case "claude":
		if provider != "anthropic" || !strings.Contains(family, "claude") || !strings.Contains(model, "claude") {
			return fmt.Errorf("--auditor-driver claude requires a declared anthropic/claude/claude-* auditor identity")
		}
		return nil
	case "codex":
		modelIsOpenAI := strings.HasPrefix(model, "gpt-") || regexp.MustCompile(`^o[0-9]`).MatchString(model)
		familyIsOpenAI := strings.HasPrefix(family, "gpt") || family == "openai" || regexp.MustCompile(`^o[0-9]`).MatchString(family)
		if provider != "openai" || !familyIsOpenAI || !modelIsOpenAI {
			return fmt.Errorf("--auditor-driver codex requires a declared openai/gpt-or-o/gpt-or-o auditor identity")
		}
		return nil
	default:
		return fmt.Errorf("--auditor-driver %q, want http, codex, or claude", driver)
	}
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

	args := []string{"issue", "view", strconv.Itoa(issue), "--repo", repo, "--json", "number,title,body,state,closedAt,url,closedByPullRequestsReferences"}
	out, err = f.runner(ctx, "gh", args...)
	if err != nil {
		return modelroute.IssueAuditEvidence{}, err
	}
	var issueRow struct {
		Number               int    `json:"number"`
		Title                string `json:"title"`
		Body                 string `json:"body"`
		State                string `json:"state"`
		ClosedAt             string `json:"closedAt"`
		URL                  string `json:"url"`
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
	diff, err := f.readClosingDiff(ctx, commit)
	if err != nil {
		return modelroute.IssueAuditEvidence{}, fmt.Errorf("read closing diff %s: %w", commit, err)
	}
	return modelroute.IssueAuditEvidence{
		IssueNumber: issueRow.Number,
		IssueURL:    issueRow.URL,
		Title:       issueRow.Title,
		Body:        issueRow.Body,
		State:       issueRow.State,
		ClosedAt:    issueRow.ClosedAt,
		CommitSHA:   commit,
		Diff:        string(diff),
		Evidence:    []modelroute.EvidenceRef{{Kind: "github-closure", Ref: source}},
	}, nil
}

func (f *githubIssueAuditFetcher) readClosingDiff(ctx context.Context, commit string) ([]byte, error) {
	parents, err := f.runner(ctx, "git", "rev-list", "--parents", "-n", "1", commit)
	if err != nil {
		return nil, err
	}
	fields := strings.Fields(string(parents))
	if len(fields) < 2 {
		return nil, fmt.Errorf("commit %s has no parent to diff", commit)
	}
	if len(fields) > 2 {
		// A normal `git show <merge>` is a combined diff and can be empty even
		// when the PR changed hundreds of lines. The PR's actual closing change
		// is the merge result relative to its first parent.
		return f.runner(ctx, "git", "diff", "--no-ext-diff", "--binary", "--no-renames", fields[1], fields[0])
	}
	return f.runner(ctx, "git", "show", "--format=", "--no-ext-diff", "--binary", "--no-renames", fields[0])
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
	client := agent.NewHTTPPlanner(r.endpoint, r.identity.Model, r.apiKey)
	client.MaxTokens = 512
	client.Temperature = 0
	temp := 0.0
	completion, err := client.Complete(ctx, []agent.Message{
		{Role: agent.RoleUser, Content: req.Prompt},
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
	runner := r.runner
	if runner == nil {
		runner = runIssueAuditProcess
	}
	switch r.driver {
	case "claude":
		out, err := runner(ctx, req.Prompt, "claude",
			"-p", "--model", r.identity.Model, "--output-format", "json",
			"--json-schema", issueAuditResultSchema, "--tools", "")
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

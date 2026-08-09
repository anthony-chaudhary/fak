package agenticbench

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/genlock"
)

const (
	ExternalHarnessQueueSchema = "fak.agentic-benchmark-external-harness-queue.v1"
	externalHarnessQueueLock   = "internal/agenticbench/external-harness-queue.lock.json"
)

type ExternalHarnessQueue struct {
	Schema             string                     `json:"schema"`
	GeneratedAt        string                     `json:"generated_at"`
	Epic               int                        `json:"epic"`
	Status             string                     `json:"status"`
	ResultClaimAllowed bool                       `json:"result_claim_allowed"`
	Summary            ExternalHarnessSummary     `json:"summary"`
	Items              []ExternalHarnessQueueItem `json:"items"`
	ClaimBoundary      string                     `json:"claim_boundary"`
}

type ExternalHarnessSummary struct {
	ItemsTotal              int   `json:"items_total"`
	ReadyItems              int   `json:"ready_items"`
	BlockedItems            int   `json:"blocked_items"`
	RequiredReturnArtifacts int   `json:"required_return_artifacts"`
	PendingIssues           []int `json:"pending_issues"`
}

type ExternalHarnessQueueItem struct {
	Issue               int              `json:"issue"`
	Packet              string           `json:"packet"`
	Title               string           `json:"title"`
	SourceArtifact      string           `json:"source_artifact"`
	Gate                string           `json:"gate"`
	Status              string           `json:"status"`
	ExternalState       string           `json:"external_state"`
	ClaimBoundary       string           `json:"claim_boundary,omitempty"`
	Commands            []HarnessCommand `json:"commands,omitempty"`
	Arms                []HarnessArm     `json:"arms,omitempty"`
	RequiredBeforeClaim []string         `json:"required_before_claim"`
	CompareMetrics      []string         `json:"compare_metrics,omitempty"`
	ResultClaimAllowed  bool             `json:"result_claim_allowed"`
}

type HarnessCommand struct {
	ID             string   `json:"id"`
	Description    string   `json:"description,omitempty"`
	Command        string   `json:"command"`
	Artifacts      []string `json:"artifacts,omitempty"`
	Required       bool     `json:"required"`
	ArtifactStatus string   `json:"artifact_status,omitempty"`
	ArtifactDetail string   `json:"artifact_detail,omitempty"`
}

type HarnessArm struct {
	Name              string   `json:"name"`
	Harness           string   `json:"harness,omitempty"`
	Command           string   `json:"command"`
	OutputDir         string   `json:"output_dir,omitempty"`
	RequiredArtifacts []string `json:"required_artifacts,omitempty"`
}

func BuildExternalHarnessQueue(root string, now time.Time) (*ExternalHarnessQueue, error) {
	report, err := Build(root, now)
	if err != nil {
		return nil, err
	}
	queue := &ExternalHarnessQueue{
		Schema:             ExternalHarnessQueueSchema,
		GeneratedAt:        now.UTC().Format(time.RFC3339),
		Epic:               report.Epic,
		Status:             report.Status,
		ResultClaimAllowed: false,
		ClaimBoundary:      "External harness queue only: folds pending child contracts into runnable work items and refuses a #868 result claim until benchmark-native raw/fak grader artifacts are checked in.",
	}
	for _, child := range report.Children {
		if child.Gate != "PENDING_EXTERNAL_HARNESS" {
			continue
		}
		doc, err := readJSON(filepath.Join(root, filepath.FromSlash(child.Artifact)))
		if err != nil {
			return nil, fmt.Errorf("read queue source #%d: %w", child.Issue, err)
		}
		var item ExternalHarnessQueueItem
		if child.Issue == 870 {
			item = queueGLM(child, doc)
		} else {
			item = queueContract(child, doc)
		}
		queue.Items = append(queue.Items, item)
		queue.Summary.PendingIssues = append(queue.Summary.PendingIssues, child.Issue)
		queue.Summary.RequiredReturnArtifacts += len(item.RequiredBeforeClaim)
		if strings.HasPrefix(item.ExternalState, "BLOCKED") {
			queue.Summary.BlockedItems++
		} else {
			queue.Summary.ReadyItems++
		}
	}
	queue.Summary.ItemsTotal = len(queue.Items)
	return queue, nil
}

func queueGLM(child ChildStatus, doc map[string]any) ExternalHarnessQueueItem {
	item := ExternalHarnessQueueItem{
		Issue:               child.Issue,
		Packet:              child.Packet,
		Title:               child.Title,
		SourceArtifact:      child.Artifact,
		Gate:                child.Gate,
		Status:              child.Status,
		ExternalState:       "READY_FOR_EXTERNAL_RUN",
		ClaimBoundary:       nestedString(doc, "summary", "honesty"),
		ResultClaimAllowed:  false,
		RequiredBeforeClaim: glmRequiredBeforeClaim(doc),
		CompareMetrics: []string{
			"swebench_resolve_rate",
			"safe_completion",
			"latency",
			"token_or_cost_proxy",
			"policy_blocks",
			"evidence_completeness",
		},
	}
	if containsString(nestedStringSlice(doc, "summary", "required_failed"), "preflight") {
		item.ExternalState = "BLOCKED_ON_H200_VLLM_READY"
	}
	for _, step := range anySlice(doc["steps"]) {
		m, ok := step.(map[string]any)
		if !ok {
			continue
		}
		id := str(m, "id")
		kind := str(m, "kind")
		if id != "preflight" && !strings.HasPrefix(kind, "live-") && kind != "manual-live-serving" {
			continue
		}
		cmd := nestedString(m, "command", "shell")
		if cmd == "" {
			continue
		}
		item.Commands = append(item.Commands, HarnessCommand{
			ID:             id,
			Description:    str(m, "description"),
			Command:        cmd,
			Artifacts:      stringSlice(m, "artifacts"),
			Required:       boolv(m, "required"),
			ArtifactStatus: str(m, "artifact_status"),
			ArtifactDetail: str(m, "artifact_detail"),
		})
	}
	return item
}

func queueContract(child ChildStatus, doc map[string]any) ExternalHarnessQueueItem {
	item := ExternalHarnessQueueItem{
		Issue:               child.Issue,
		Packet:              child.Packet,
		Title:               child.Title,
		SourceArtifact:      child.Artifact,
		Gate:                child.Gate,
		Status:              child.Status,
		ExternalState:       child.Status,
		ClaimBoundary:       str(doc, "claim_boundary"),
		RequiredBeforeClaim: stringSlice(doc, "required_before_claim"),
		CompareMetrics:      stringSlice(doc, "compare_metrics"),
		ResultClaimAllowed:  boolv(doc, "result_claim_allowed"),
	}
	if strings.Contains(item.ExternalState, "READY") {
		item.ExternalState = "READY_FOR_EXTERNAL_HARNESS"
	}
	for _, arm := range anySlice(doc["arms"]) {
		m, ok := arm.(map[string]any)
		if !ok {
			continue
		}
		item.Arms = append(item.Arms, HarnessArm{
			Name:              str(m, "name"),
			Harness:           str(m, "harness"),
			Command:           str(m, "command"),
			OutputDir:         str(m, "output_dir"),
			RequiredArtifacts: stringSlice(m, "required_artifacts"),
		})
	}
	return item
}

func glmRequiredBeforeClaim(doc map[string]any) []string {
	var out []string
	for _, id := range nestedStringSlice(doc, "summary", "required_failed") {
		out = append(out, fmt.Sprintf("%s must pass on the target GLM-5.2/vLLM node", id))
	}
	for _, id := range nestedStringSlice(doc, "summary", "required_missing") {
		out = append(out, fmt.Sprintf("%s artifact must be generated and pass the final-check witness", id))
	}
	return out
}

func RenderExternalHarnessQueueMarkdown(q *ExternalHarnessQueue) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Agentic Benchmark Epic #868 External Harness Queue\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", q.GeneratedAt)
	fmt.Fprintf(&b, "- Status: `%s`\n", q.Status)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", q.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Items: `%d`\n", q.Summary.ItemsTotal)
	fmt.Fprintf(&b, "- Ready items: `%d`\n", q.Summary.ReadyItems)
	fmt.Fprintf(&b, "- Blocked items: `%d`\n", q.Summary.BlockedItems)
	fmt.Fprintf(&b, "- Required returned artifacts: `%d`\n", q.Summary.RequiredReturnArtifacts)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", q.ClaimBoundary)

	fmt.Fprintf(&b, "## Queue\n\n")
	fmt.Fprintf(&b, "| Issue | Packet | Source | External state | Required returns |\n")
	fmt.Fprintf(&b, "|---:|---|---|---|---:|\n")
	for _, item := range q.Items {
		fmt.Fprintf(&b, "| #%d | `%s` | `%s` | `%s` | %d |\n",
			item.Issue, item.Packet, item.SourceArtifact, item.ExternalState, len(item.RequiredBeforeClaim))
	}

	for _, item := range q.Items {
		fmt.Fprintf(&b, "\n## #%d Packet %s - %s\n\n", item.Issue, item.Packet, item.Title)
		fmt.Fprintf(&b, "- Source: `%s`\n", item.SourceArtifact)
		fmt.Fprintf(&b, "- Status: `%s`\n", item.Status)
		fmt.Fprintf(&b, "- External state: `%s`\n", item.ExternalState)
		if item.ClaimBoundary != "" {
			fmt.Fprintf(&b, "- Boundary: %s\n", item.ClaimBoundary)
		}
		if len(item.Commands) > 0 {
			fmt.Fprintf(&b, "\n### Commands\n\n")
			for _, cmd := range item.Commands {
				fmt.Fprintf(&b, "#### %s\n\n", cmd.ID)
				if cmd.Description != "" {
					fmt.Fprintf(&b, "%s\n\n", cmd.Description)
				}
				fmt.Fprintf(&b, "```bash\n%s\n```\n\n", cmd.Command)
				if len(cmd.Artifacts) > 0 {
					fmt.Fprintf(&b, "Artifacts: `%s`\n\n", strings.Join(cmd.Artifacts, "`, `"))
				}
				if cmd.ArtifactStatus != "" {
					fmt.Fprintf(&b, "Current artifact status: `%s` - %s\n\n", cmd.ArtifactStatus, cmd.ArtifactDetail)
				}
			}
		}
		if len(item.Arms) > 0 {
			fmt.Fprintf(&b, "\n### Arms\n\n")
			for _, arm := range item.Arms {
				fmt.Fprintf(&b, "#### %s\n\n", arm.Name)
				if arm.Harness != "" {
					fmt.Fprintf(&b, "- Harness: `%s`\n", arm.Harness)
				}
				if arm.OutputDir != "" {
					fmt.Fprintf(&b, "- Output: `%s`\n", arm.OutputDir)
				}
				if len(arm.RequiredArtifacts) > 0 {
					fmt.Fprintf(&b, "- Required artifacts: `%s`\n", strings.Join(arm.RequiredArtifacts, "`, `"))
				}
				fmt.Fprintf(&b, "\n```powershell\n%s\n```\n\n", arm.Command)
			}
		}
		if len(item.RequiredBeforeClaim) > 0 {
			fmt.Fprintf(&b, "\n### Required Before Claim\n\n")
			for _, req := range item.RequiredBeforeClaim {
				fmt.Fprintf(&b, "- %s\n", req)
			}
		}
		if len(item.CompareMetrics) > 0 {
			fmt.Fprintf(&b, "\n### Compare Metrics\n\n")
			for _, metric := range item.CompareMetrics {
				fmt.Fprintf(&b, "- `%s`\n", metric)
			}
		}
	}
	return b.String()
}

func WriteExternalHarnessQueue(root, jsonPath, markdownPath string, now time.Time) (*ExternalHarnessQueue, error) {
	// The queue embeds GeneratedAt, so two renders of the same committed inputs are
	// intentionally not byte-reproducible. Key freshness on the source tree instead:
	// an unchanged canonical input skips both artifact writes completely.
	input, err := externalHarnessQueueInput(root)
	if err != nil {
		return nil, err
	}
	// Build state lives beside this generator, under the named agenticbench lane.
	// Putting it under docs would misclassify state as prose; a loose root path
	// would fall through to the exclusive global catch-all.
	lock := genlock.OpenAt(root, externalHarnessQueueLock)
	if outputsCurrent(root, lock, input, jsonPath, markdownPath) {
		return BuildExternalHarnessQueue(root, now)
	}

	queue, err := BuildExternalHarnessQueue(root, now)
	if err != nil {
		return nil, err
	}
	if jsonPath != "" {
		b, err := json.MarshalIndent(queue, "", "  ")
		if err != nil {
			return nil, err
		}
		b = append(b, '\n')
		if err := writeQueueFile(jsonPath, b); err != nil {
			return nil, err
		}
		if err := lock.Record(repoPath(root, jsonPath), input); err != nil {
			return nil, err
		}
		if err := lock.Save(); err != nil {
			return nil, err
		}
	}
	if markdownPath != "" {
		if err := writeQueueFile(markdownPath, []byte(RenderExternalHarnessQueueMarkdown(queue))); err != nil {
			return nil, err
		}
		if err := lock.Record(repoPath(root, markdownPath), input); err != nil {
			return nil, err
		}
		if err := lock.Save(); err != nil {
			return nil, err
		}
	}
	return queue, nil
}

func outputsCurrent(root string, lock *genlock.Lock, input []byte, paths ...string) bool {
	seen := false
	for _, path := range paths {
		if path == "" {
			continue
		}
		seen = true
		if !lock.Current(repoPath(root, path), input) {
			return false
		}
	}
	return seen
}

func repoPath(root, path string) string {
	if rel, err := filepath.Rel(root, path); err == nil && rel != "." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(path)
}

func externalHarnessQueueInput(root string) ([]byte, error) {
	// The queue reads these declared child artifacts. Keep the freshness input
	// explicit: scanning the output directory would make the generated queue hash
	// itself and could never become current.
	files := []string{
		"BENCHMARK-AUTHORITY.md",
		"experiments/agent-live/agentdojo-fak-fullstack-20260625.json",
		"experiments/vllm/glm52-agentic-battery/final-check.json",
		"experiments/agent-live/swebench-opus-smoke-contract-20260626.json",
		"experiments/agent-live/deepswe-raw-fak-contract-20260626.json",
		"experiments/agent-live/toolsandbox-official-run-contract-20260626.json",
		"experiments/agent-live/terminalbench-official-run-contract-20260626.json",
		"experiments/agent-live/browseraction-official-run-contract-20260626.json",
	}
	var parts [][]byte
	for _, rel := range files {
		body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				body = []byte("<missing>")
			} else {
				return nil, err
			}
		}
		parts = append(parts, []byte(rel), body)
	}
	return genlock.Canonical(parts...), nil
}

func writeQueueFile(path string, b []byte) error {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, b, 0o644)
}

func nestedStringSlice(m map[string]any, keys ...string) []string {
	return stringsFromAny(nestedAny(m, keys...))
}

func stringsFromAny(raw any) []string {
	items, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func anySlice(raw any) []any {
	items, _ := raw.([]any)
	return items
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

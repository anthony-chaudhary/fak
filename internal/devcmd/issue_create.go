package devcmd

import (
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/categorybaseline"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// issueCreateRunner is the injectable gh-argv executor behind runIssueCreateWith — same
// shape as taskHandoffRunner (taskmgr.go:174) so a nil runner can default straight to the
// existing runTaskHandoffGH (taskmgr.go:343) instead of a second exec.Command wrapper.
type issueCreateRunner func(args []string) (stdout, stderr string, ok bool)

var issueCreateHeadingRE = regexp.MustCompile(`^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$`)

// issueCreateShiftLeftScope makes the two scope decisions explicit before an issue
// reaches GitHub. Legacy headings remain accepted, but newly filed contracts use the
// names that state the decision an author must make rather than generic containers.
func issueCreateShiftLeftScope(body string) (string, error) {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	type decision struct {
		line  int
		level string
		value string
	}
	var core, boundary []decision
	for i, raw := range lines {
		m := issueCreateHeadingRE.FindStringSubmatch(raw)
		if m == nil {
			continue
		}
		heading := strings.ToLower(strings.Join(strings.Fields(strings.Trim(m[2], "`*_:# ")), " "))
		if heading != "core through-line" && heading != "in scope" && heading != "gold-plating boundary" && heading != "out of scope" {
			continue
		}
		end := len(lines)
		for j := i + 1; j < len(lines); j++ {
			if issueCreateHeadingRE.MatchString(lines[j]) {
				end = j
				break
			}
		}
		valueLines := make([]string, 0, end-i-1)
		for _, line := range lines[i+1 : end] {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || (strings.HasPrefix(trimmed, "<!--") && strings.HasSuffix(trimmed, "-->")) {
				continue
			}
			valueLines = append(valueLines, trimmed)
		}
		d := decision{line: i, level: m[1], value: strings.TrimSpace(strings.Join(valueLines, "\n"))}
		if heading == "core through-line" || heading == "in scope" {
			core = append(core, d)
		} else {
			boundary = append(boundary, d)
		}
	}
	if len(core) != 1 || len(boundary) != 1 {
		return "", fmt.Errorf("default issue contracts require exactly one ## Core through-line and one ## Gold-plating boundary (legacy In scope/Out of scope are accepted; use --raw-body only for a deliberate non-contract issue)")
	}
	if core[0].value == "" {
		return "", fmt.Errorf("Core through-line is empty: name the shortest change -> real seam -> observable outcome -> witness path")
	}
	if boundary[0].value == "" {
		return "", fmt.Errorf("Gold-plating boundary is empty: name tempting work the outcome and witness still work without, or state why none exists")
	}
	lines[core[0].line] = core[0].level + " Core through-line"
	lines[boundary[0].line] = boundary[0].level + " Gold-plating boundary"
	return strings.Join(lines, "\n"), nil
}

// issueCreateResult is the --json shape: the rendered gh argv is always included (even on
// a dry run) so a caller can see exactly what would run or did run.
type issueCreateResult struct {
	OK     bool     `json:"ok"`
	DryRun bool     `json:"dry_run"`
	Title  string   `json:"title"`
	Repo   string   `json:"repo,omitempty"`
	Labels []string `json:"labels,omitempty"`
	Args   []string `json:"args"`
	URL    string   `json:"url,omitempty"`
	Error  string   `json:"error,omitempty"`
}

func runIssueCreate(stdout, stderr io.Writer, argv []string) int {
	return runIssueCreateWith(stdout, stderr, argv, nil)
}

func issueCreateClassifyBody(body, category, layer string, registry categorybaseline.Registry) (string, error) {
	category = strings.ToLower(strings.TrimSpace(category))
	layer = strings.ToLower(strings.TrimSpace(layer))
	hasMetadata := regexp.MustCompile(`(?im)^##\s+(Category|Layer)\s*$`).MatchString(body)
	if category == "" && layer == "" && !hasMetadata {
		return body, nil
	}
	if category == "" && layer == "" {
		return "", fmt.Errorf("body already contains Category/Layer metadata; pass it through --category and --layer so it can be validated")
	}
	if category == "" {
		return "", fmt.Errorf("--layer requires --category")
	}
	var declaration *categorybaseline.Category
	for i := range registry.Categories {
		if registry.Categories[i].Name == category {
			declaration = &registry.Categories[i]
			break
		}
	}
	if declaration != nil {
		if layer == "" {
			return "", fmt.Errorf("category %q is baseline-governed; --layer is required", category)
		}
		known := false
		for _, candidate := range declaration.Layers {
			known = known || candidate == layer
		}
		if !known {
			return "", fmt.Errorf("layer %q is not declared for category %q (known: %s)", layer, category, strings.Join(declaration.Layers, ","))
		}
	}
	if layer == "" {
		return "", fmt.Errorf("--category requires --layer")
	}
	if hasMetadata {
		return "", fmt.Errorf("body already contains Category/Layer metadata; pass it only through --category and --layer")
	}
	return strings.TrimRight(body, "\r\n") + "\n\n## Category\n\n" + category + "\n\n## Layer\n\n" + layer + "\n", nil
}

// runIssueCreateWith is fak's smooth, structurally-ungated path for filing one GitHub
// issue: it shells to `gh issue create` from the trusted fak binary (runner defaults to
// runTaskHandoffGH, taskmgr.go:343-351) rather than the model proposing raw `gh issue
// create` via Bash — which is exactly the invocation internal/adjudicator/reversibility.go
// deliberately escalates (REQUIRE_WITNESS/ESCALATE, by design, not a false positive). A
// compiled verb shelling out internally was never in scope for that classifier, the same
// way `fak commit`/`fak sync push` already sidestep the `git push` family. The runner
// param exists purely so tests can inject a fake and assert on the built gh argv without
// ever invoking real gh.
func runIssueCreateWith(stdout, stderr io.Writer, argv []string, runner issueCreateRunner) int {
	fs := flag.NewFlagSet("issue create", flag.ContinueOnError)
	fs.SetOutput(stderr)
	title := fs.String("title", "", "issue title (required)")
	body := fs.String("body", "", "issue body text (exactly one of --body/--body-file)")
	bodyFile := fs.String("body-file", "", "path to a file containing the issue body")
	labels := fs.String("labels", "", "comma-separated labels")
	estimatePoints := fs.Float64("estimate-points", 0, "project-work estimate points (required when body lacks Work estimate)")
	parentBaselinePoints := fs.Float64("parent-baseline-points", 0, "parent production-scope baseline points (required when body lacks Overall completion contribution)")
	contributionPoints := fs.Float64("contribution-points", 0, "parent contribution points (default: estimate-points)")
	completionStandard := fs.String("completion-standard", "production", "production|integrated|staging|development|demo|prototype|experiment|research")
	targetEnvelope := fs.String("target-envelope", "", "production target envelope entries (required unless body declares them)")
	witnessedEnvelope := fs.String("witnessed-envelope", "", "directly witnessed envelope entries (required unless body declares them)")
	rawBody := fs.Bool("raw-body", false, "do not append/review project-work metadata (only for non-dispatchable administrative issues)")
	category := fs.String("category", "", "explicit category for baseline-aware dispatch")
	layer := fs.String("layer", "", "ordered layer within --category")
	repo := fs.String("repo", "", "owner/name override (default: gh infers from cwd)")
	dryRun := fs.Bool("dry-run", false, "render the issue + gh argv without calling gh")
	asJSON := fs.Bool("json", false, "emit the machine-readable result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue create: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*title) == "" {
		fmt.Fprintln(stderr, "fak-dev issue create: --title is required")
		return 2
	}
	*title = hooks.ScrubHardwareNames(*title)
	if strings.TrimSpace(*body) != "" && strings.TrimSpace(*bodyFile) != "" {
		fmt.Fprintln(stderr, "fak-dev issue create: pass exactly one of --body or --body-file")
		return 2
	}
	resolvedBody := *body
	if strings.TrimSpace(*bodyFile) != "" {
		b, err := os.ReadFile(*bodyFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue create: read --body-file: %v\n", err)
			return 2
		}
		resolvedBody = string(b)
	}
	resolvedBody = hooks.ScrubHardwareNames(resolvedBody)
	if strings.TrimSpace(resolvedBody) == "" {
		fmt.Fprintln(stderr, "fak-dev issue create: --body or --body-file is required")
		return 2
	}
	if !*rawBody {
		var err error
		resolvedBody, err = issueCreateShiftLeftScope(resolvedBody)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue create: %v\n", err)
			return 2
		}
		frame := issuepolicy.AssessProblemFrame(issuepolicy.IssueDraft{Title: strings.TrimSpace(*title), Body: resolvedBody})
		if !frame.Enforced {
			fmt.Fprintln(stderr, "fak-dev issue create: problem frame is incomplete: problem_frame_unclassified; repair: declare Centrality and P1-P4 with concrete evidence")
			return 2
		}
		if !frame.Ready {
			fmt.Fprintf(stderr, "fak-dev issue create: problem frame is incomplete: %s; repair: %s\n", strings.Join(frame.Reasons, ","), strings.Join(frame.RepairActions, "; "))
			return 2
		}
		resolvedBody, err = issuepolicy.AppendProjectWorkDefaults(resolvedBody, issuepolicy.ProjectWorkAuthoring{
			EstimatePoints: *estimatePoints, ParentBaseline: *parentBaselinePoints,
			ContributionPoints: *contributionPoints, CompletionStandard: *completionStandard,
			TargetEnvelope: *targetEnvelope, WitnessedEnvelope: *witnessedEnvelope,
		})
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue create: %v\n", err)
			return 2
		}
		review := issuepolicy.ReviewIssueDraft(issuepolicy.IssueDraft{Title: strings.TrimSpace(*title), Body: resolvedBody}, issuepolicy.Options{StrictProjectWork: true})
		if review.ProjectWork.Status != issuepolicy.ProjectWorkValid {
			fmt.Fprintf(stderr, "fak-dev issue create: generated project-work contract is %s: %s\n", review.ProjectWork.Status, strings.Join(review.ProjectWork.Invalid, "; "))
			return 2
		}
		if review.OperatingEnvelope.Required && review.OperatingEnvelope.Status != issuepolicy.EnvelopeMet {
			fmt.Fprintf(stderr, "fak-dev issue create: generated production operating envelope is %s; gaps=%d invalid=%s\n", review.OperatingEnvelope.Status, len(review.OperatingEnvelope.Gaps), strings.Join(review.OperatingEnvelope.Invalid, "; "))
			return 2
		}
	}
	classifiedBody, err := issueCreateClassifyBody(resolvedBody, *category, *layer, categorybaseline.Load("."))
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue create: %v\n", err)
		return 2
	}
	resolvedBody = classifiedBody

	labelList := issueFanoutSplit(*labels)
	if !*rawBody {
		labelList = issueCreateShiftLeftLabels(*title, resolvedBody, labelList)
	}
	args := []string{"issue", "create", "--title", *title, "--body", resolvedBody}
	for _, l := range labelList {
		args = append(args, "--label", l)
	}
	if strings.TrimSpace(*repo) != "" {
		args = append(args, "--repo", *repo)
	}

	result := issueCreateResult{DryRun: *dryRun, Title: *title, Repo: *repo, Labels: labelList, Args: args}

	if *dryRun {
		result.OK = true
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue create")
		}
		fmt.Fprintf(stdout, "fak-dev issue create --dry-run: would run `gh %s`\n", strings.Join(args, " "))
		return 0
	}

	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}
	out, errOut, ok := run(args)
	result.OK = ok
	result.URL = strings.TrimSpace(out)
	if !ok {
		result.Error = strings.TrimSpace(errOut)
		if *asJSON {
			encodeJSONOrFail(stdout, stderr, result, "fak-dev issue create")
		} else {
			fmt.Fprintf(stderr, "fak-dev issue create: gh failed: %s\n", result.Error)
		}
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue create")
	}
	fmt.Fprintln(stdout, result.URL)
	return 0
}

// issueCreateShiftLeftLabels derives and defaults canonical fleet labels from title and body
// when --labels is omitted or incomplete, shifting label derivation left to creation time.
func issueCreateShiftLeftLabels(title, body string, explicit []string) []string {
	var labels []string
	seen := make(map[string]bool)
	hasKind := false
	hasClass := false
	hasPriority := false
	hasGen := false

	add := func(l string) {
		l = strings.TrimSpace(l)
		if l == "" || seen[l] {
			return
		}
		seen[l] = true
		labels = append(labels, l)
		switch {
		case l == "enhancement" || l == "bug" || l == "documentation" || l == "performance" || l == "testing" || l == "build" || l == "research":
			hasKind = true
		case strings.HasPrefix(l, "class:"):
			hasClass = true
		case strings.HasPrefix(l, "priority/"):
			hasPriority = true
		case strings.HasPrefix(l, "gen/"):
			hasGen = true
		}
	}

	for _, l := range explicit {
		add(l)
	}

	lowerTitle := strings.ToLower(title)
	lowerBody := strings.ToLower(body)

	// 1. Kind from conventional commit prefix if not explicitly set
	if !hasKind {
		switch {
		case strings.HasPrefix(lowerTitle, "feat"):
			add("enhancement")
		case strings.HasPrefix(lowerTitle, "fix"):
			add("bug")
		case strings.HasPrefix(lowerTitle, "docs"):
			add("documentation")
		case strings.HasPrefix(lowerTitle, "perf"):
			add("performance")
		case strings.HasPrefix(lowerTitle, "test"):
			add("testing")
		case strings.HasPrefix(lowerTitle, "build") || strings.HasPrefix(lowerTitle, "ci"):
			add("build")
		case strings.HasPrefix(lowerTitle, "research"):
			add("research")
		default:
			add("enhancement")
		}
	}

	// 2. Class: default to class:dev if no class:* label set
	if !hasClass {
		add("class:dev")
	}

	// 3. Priority: default to priority/P1 if not explicitly set
	if !hasPriority {
		add("priority/P1")
	}

	// 4. Generation: fix/docs are gen/now, otherwise gen/next
	if !hasGen {
		if strings.HasPrefix(lowerTitle, "fix") || strings.HasPrefix(lowerTitle, "docs") {
			add("gen/now")
		} else {
			add("gen/next")
		}
	}

	// 5. Area from conventional commit scope in title e.g. type(scope):
	if start := strings.Index(lowerTitle, "("); start != -1 {
		if end := strings.Index(lowerTitle[start:], ")"); end != -1 {
			scope := lowerTitle[start+1 : start+end]
			switch scope {
			case "compute", "cuda", "gpu":
				add("compute")
			case "model", "moe", "loader":
				add("model")
			case "cachemeta", "kv", "radixkv":
				add("compute")
			case "gateway", "serve", "proxy":
				add("agentic-serving")
			case "cmd", "dispatch", "taskmgr":
				add("dispatch")
			case "bench":
				add("benchmark")
			case "security", "guard", "adjudicator":
				add("security")
			}
		}
	}

	// 6. Keywords from title/body for domain tags if explicit was empty
	if len(explicit) == 0 {
		text := lowerTitle + " " + lowerBody
		if strings.Contains(text, "cuda") {
			add("cuda")
		}
		if strings.Contains(text, "moe") || strings.Contains(text, "expert") {
			add("moe")
		}
		if strings.Contains(text, "quant") {
			add("quantization")
		}
		if strings.Contains(text, "pagedattention") || strings.Contains(text, "kv cache") || strings.Contains(text, "kv-cache") {
			add("pagedattention")
		}
	}

	return labels
}

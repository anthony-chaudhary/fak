package issuepolicy

import (
	"path/filepath"
	"strings"
)

const (
	// ReasonHorizontalFragment identifies a feature draft that proposes a
	// horizontal code fragment instead of an executable vertical slice.
	ReasonHorizontalFragment = "ISSUE_HORIZONTAL_FRAGMENT"
	// ReasonFragmentationSpike identifies a feature draft that declares more
	// than three children before it declares a working tracer bullet.
	ReasonFragmentationSpike = "ISSUE_FRAGMENTATION_SPIKE"

	tracerStageInput   = "input_trigger"
	tracerStageEngine  = "engine_execution"
	tracerStageReceipt = "external_witness_or_receipt"
)

// TracerBulletReadout is the deterministic feature-intake result. The check is
// deliberately narrow: it applies only to type:feat drafts and reads only the
// sections where authors declare the execution spine.
type TracerBulletReadout struct {
	Enforced      bool     `json:"enforced"`
	Ready         bool     `json:"ready"`
	MissingStages []string `json:"missing_stages,omitempty"`
	TypeOnlyPaths bool     `json:"type_only_paths,omitempty"`
	FragmentCount int      `json:"fragment_count,omitempty"`
	RepairActions []string `json:"repair_actions,omitempty"`
}

// AssessTracerBullet requires a feature draft to name an entry trigger, real
// execution, and an externally observable witness or receipt. It does not try
// to infer architecture from arbitrary prose and never applies to bug or docs
// issues.
func AssessTracerBullet(d IssueDraft) TracerBulletReadout {
	if !isFeatureIssueDraft(d) {
		return TracerBulletReadout{Ready: true}
	}

	candidate := CandidateFromIssueDraft(d)
	if isTracerBulletExemptWorkUnit(candidate.WorkUnit) {
		return TracerBulletReadout{Ready: true}
	}

	sections := markdownSections(d.Body)
	spine := strings.ToLower(strings.Join([]string{
		sections["working spine"],
		sections["core through-line"],
	}, "\n"))

	readout := TracerBulletReadout{Enforced: true}
	if !containsAffirmativeTracerTerm(spine, "input", "entrypoint", "entry point", "trigger", "cli", "command", "invocation", "request", "handler") {
		readout.MissingStages = append(readout.MissingStages, tracerStageInput)
	}
	if !containsAffirmativeTracerTerm(spine, "engine", "execute", "execution", "algorithm", "runtime", "kernel", "processor") {
		readout.MissingStages = append(readout.MissingStages, tracerStageEngine)
	}
	if !containsAffirmativeTracerTerm(spine, "external witness", "verifiable witness", "durable receipt", "receipt", "observable result", "response", "output") {
		readout.MissingStages = append(readout.MissingStages, tracerStageReceipt)
	}
	readout.TypeOnlyPaths = onlyTypeDefinitionPaths(candidate.Paths)
	readout.FragmentCount = declaredFragmentCount(d.Body)
	readout.Ready = len(readout.MissingStages) == 0 && !readout.TypeOnlyPaths && readout.FragmentCount <= 3
	if readout.Ready {
		return readout
	}

	if len(readout.MissingStages) > 0 {
		readout.RepairActions = append(readout.RepairActions,
			"declare one unbroken Working spine or Core through-line: input/trigger -> engine execution -> external witness or durable receipt")
	}
	if readout.TypeOnlyPaths {
		readout.RepairActions = append(readout.RepairActions,
			"include the production caller or entrypoint, executing engine path, and receipt producer; type/interface-only paths are not a tracer bullet")
	}
	if readout.FragmentCount > 3 {
		readout.RepairActions = append(readout.RepairActions,
			"combine the declared child issues into a working tracer bullet before expanding into more than three horizontal follow-ons")
	}
	return readout
}

func applyTracerBulletReview(review *Review, draft IssueDraft) {
	readout := AssessTracerBullet(draft)
	if !readout.Enforced || readout.Ready {
		return
	}
	review.OK = false
	review.Verdict = "refused"
	review.Dispatchability = Refused
	if len(readout.MissingStages) > 0 || readout.TypeOnlyPaths {
		addReviewReason(review, ReasonHorizontalFragment)
	}
	if readout.FragmentCount > 3 {
		addReviewReason(review, ReasonFragmentationSpike)
	}
	for _, stage := range readout.MissingStages {
		review.MissingFields = appendUnique(review.MissingFields, "tracer_bullet."+stage)
	}
	review.Coordination = appendUnique(review.Coordination, readout.RepairActions...)
}

func hasIssueLabel(labels []IssueLabel, want string) bool {
	for _, label := range labels {
		if strings.EqualFold(strings.TrimSpace(label.Name), want) {
			return true
		}
	}
	return false
}

func isFeatureIssueDraft(d IssueDraft) bool {
	// Explicit bug/docs metadata wins even if a malformed issue also carries a
	// feature label. Those issue classes are outside this policy.
	for _, exempt := range []string{"type:bug", "type:docs", "type:doc"} {
		if hasIssueLabel(d.Labels, exempt) {
			return false
		}
	}
	if hasIssueLabel(d.Labels, "type:feat") {
		return true
	}
	title := strings.ToLower(strings.TrimSpace(d.Title))
	if !strings.HasPrefix(title, "feat:") && !strings.HasPrefix(title, "feat(") {
		return false
	}
	// The CLI's --body/--body-file path has no labels. In that path, the
	// conventional feature title plus an explicit tracer declaration heading is
	// the intake metadata. Requiring the heading avoids retroactively applying
	// this rule to historical conventional-title issues in older body formats.
	sections := markdownSections(d.Body)
	_, hasSpine := sections["working spine"]
	_, hasThroughLine := sections["core through-line"]
	if !hasSpine && !hasThroughLine {
		return false
	}
	if d.Number == 0 {
		return true
	}
	// Filed rows without type labels predate this policy. Apply title fallback
	// only when their declared spine uses the concrete stage vocabulary this
	// validator owns; generic legacy "change -> seam -> witness" prose remains
	// descriptive rather than being reclassified retroactively.
	spine := strings.ToLower(sections["working spine"] + "\n" + sections["core through-line"])
	return containsTracerTermEvenNegated(spine,
		"input", "entrypoint", "entry point", "trigger", "cli", "command", "invocation", "request", "handler",
		"engine", "execute", "execution", "algorithm", "runtime", "kernel", "processor")
}

func isTracerBulletExemptWorkUnit(workUnit string) bool {
	switch strings.ToLower(strings.TrimSpace(workUnit)) {
	case "chore", "doc", "docs", "documentation":
		return true
	default:
		return false
	}
}

func containsAffirmativeTracerTerm(text string, terms ...string) bool {
	for _, term := range terms {
		from := 0
		for {
			rel := strings.Index(text[from:], term)
			if rel < 0 {
				break
			}
			at := from + rel
			from = at + len(term)
			if !tracerTermBoundary(text, at, len(term)) || tracerTermNegated(text, at) {
				continue
			}
			return true
		}
	}
	return false
}

func containsTracerTermEvenNegated(text string, terms ...string) bool {
	for _, term := range terms {
		from := 0
		for {
			rel := strings.Index(text[from:], term)
			if rel < 0 {
				break
			}
			at := from + rel
			from = at + len(term)
			if tracerTermBoundary(text, at, len(term)) {
				return true
			}
		}
	}
	return false
}

func tracerTermBoundary(text string, at, size int) bool {
	return (at == 0 || !isTracerWordByte(text[at-1])) &&
		(at+size == len(text) || !isTracerWordByte(text[at+size]))
}

func isTracerWordByte(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= '0' && b <= '9' || b == '_'
}

func tracerTermNegated(text string, at int) bool {
	start := at
	for start > 0 && at-start < 64 {
		if strings.ContainsRune("\n.;", rune(text[start-1])) {
			break
		}
		if start >= 2 && (text[start-2:start] == "->" || text[start-2:start] == "=>") {
			break
		}
		start--
	}
	prefix := strings.TrimSpace(text[start:at])
	// A comma alone can join a negated list ("no input, engine, or receipt"),
	// so it must not end negation. Explicit transition clauses do: the stage in
	// "do not add mocks, but execute the engine" is affirmative.
	transition := -1
	for _, marker := range []string{", then ", ", but "} {
		if i := strings.LastIndex(prefix, marker); i > transition {
			transition = i + len(marker)
		}
	}
	if transition >= 0 {
		prefix = strings.TrimSpace(prefix[transition:])
	}
	for _, negation := range []string{
		"no ", "not ", "never ", "without ", "do not ", "does not ", "will not ", "won't ", "doesn't ",
		"later ", "future ",
	} {
		if strings.Contains(prefix, negation) {
			return true
		}
	}
	return false
}

func onlyTypeDefinitionPaths(paths []string) bool {
	if len(paths) == 0 {
		return false
	}
	for _, path := range paths {
		norm := strings.ToLower(filepath.ToSlash(path))
		name := filepath.Base(norm)
		if strings.HasSuffix(name, "_test.go") || strings.HasSuffix(name, ".md") || strings.HasPrefix(norm, "docs/") || strings.Contains(norm, "/testdata/") ||
			name == "doc.go" || name == "generate.go" || name == "schema.go" ||
			strings.HasSuffix(name, "_generator.go") || strings.HasSuffix(name, "_generate.go") {
			continue
		}
		if name != "types.go" && name != "type.go" && name != "interface.go" && name != "interfaces.go" &&
			!strings.HasSuffix(name, "_types.go") && !strings.HasSuffix(name, "_interface.go") && !strings.HasSuffix(name, "_interfaces.go") {
			return false
		}
	}
	// Auxiliary docs/tests cannot turn a type-only proposal into a production
	// slice; an all-auxiliary feature is equally non-executable.
	return true
}

func declaredFragmentCount(body string) int {
	total := 0
	active := false
	bullets := 0
	var section strings.Builder
	flush := func() {
		if !active {
			return
		}
		refs := len(extractIssueRefs(section.String()))
		if refs > bullets {
			bullets = refs
		}
		total += bullets
		bullets = 0
		section.Reset()
	}
	for _, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line := strings.TrimSpace(raw)
		if match := markdownHeadingRE.FindStringSubmatch(line); match != nil {
			flush()
			heading := normalizeHeading(match[1])
			active = isFragmentHeading(heading)
			continue
		}
		if !active {
			continue
		}
		section.WriteString(raw)
		section.WriteByte('\n')
		if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") || isNumberedListItem(line) {
			bullets++
		}
	}
	flush()
	return total
}

func isNumberedListItem(line string) bool {
	delim := strings.IndexAny(line, ".)")
	if delim <= 0 || delim+1 >= len(line) || line[delim+1] != ' ' {
		return false
	}
	for _, r := range line[:delim] {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func isFragmentHeading(heading string) bool {
	for _, marker := range []string{
		"child issue", "sub-ticket", "subticket", "planned ticket", "planned-ticket",
		"follow-up issue", "follow up issue", "followup issue", "follow-up-issue",
	} {
		if strings.Contains(heading, marker) {
			return true
		}
	}
	return false
}

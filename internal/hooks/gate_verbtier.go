package hooks

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// gate_verbtier.go — the push-scoped hygiene gate (VERB_UNTIERED) that verifies every
// CLI dispatch verb in cmd/fak/main.go and cmd/fak-dev/main.go has a declared tier in
// internal/devindex/tiers.go before push.
//
// TestVerbTierCoverageIsTotal in internal/devindex/tiers_test.go already asserts that
// every dispatched CLI verb has a tier assigned in tiers.go. When an author adds a CLI
// verb without declaring its tier, git push previously succeeded and CI turned red minutes
// later. This gate surfaces missing verb tier rows at pre-push time rather than post-push CI,
// scoped to changes touching CLI dispatch or tier tables (#12172, epic #12126).

const (
	mainCmdFile   = "cmd/fak/main.go"
	devCmdFile    = "cmd/fak-dev/main.go"
	verbTiersFile = "internal/devindex/tiers.go"
)

var (
	verbTierRowRE = regexp.MustCompile(`"([A-Za-z0-9_\-]+)"\s*:\s*(?:devindex\.)?(?:Tier)?([A-Za-z0-9_"]+)`)
	mainCaseRE    = regexp.MustCompile(`"([^"]+)"`)
)

// parseDeclaredVerbTiers extracts the lower-cased verb names and tiers from the
// `var verbTiers = map[string]VerbTier{...}` map literal in internal/devindex/tiers.go.
// ok is false when the table cannot be parsed, causing the gate to fail open.
func parseDeclaredVerbTiers(body []byte) (map[string]string, bool) {
	declared := map[string]string{}
	inTable := false
	for _, line := range strings.Split(string(body), "\n") {
		trimmed := strings.TrimSpace(line)
		if !inTable {
			if strings.Contains(line, "var verbTiers = ") && strings.Contains(line, "map[string]") {
				inTable = true
			}
			continue
		}
		if trimmed == "}" {
			break
		}
		for _, m := range verbTierRowRE.FindAllStringSubmatch(line, -1) {
			val := strings.Trim(m[2], `"`)
			val = strings.TrimPrefix(val, "Tier")
			declared[strings.ToLower(m[1])] = val
		}
	}
	if len(declared) == 0 {
		return nil, false
	}
	return declared, true
}

// dispatchVerbs parses the quoted verb tokens of top-level dispatch switches in main.go bytes.
// It tracks brace depth so nested helper bodies do not end the switch scan prematurely.
func dispatchVerbs(b []byte, isOpener func(string) bool) []string {
	seen := map[string]bool{}
	var out []string
	inSwitch := false
	depth := 0
	for _, raw := range strings.Split(string(b), "\n") {
		t := strings.TrimSpace(raw)
		if !inSwitch {
			if isOpener(t) {
				inSwitch = true
				depth = 1
			}
			continue
		}
		if depth == 1 {
			if strings.HasPrefix(t, "default:") {
				inSwitch = false
				depth = 0
				continue
			}
			if strings.HasPrefix(t, "case ") && strings.HasSuffix(t, ":") {
				for _, m := range mainCaseRE.FindAllStringSubmatch(t, -1) {
					verb := strings.ToLower(m[1])
					if verb != "" && !seen[verb] {
						seen[verb] = true
						out = append(out, verb)
					}
				}
			}
		}
		depth += strings.Count(t, "{") - strings.Count(t, "}")
		if depth <= 0 {
			inSwitch = false
		}
	}
	sort.Strings(out)
	return out
}

func mainDispatchVerbs(b []byte) []string {
	return dispatchVerbs(b, func(t string) bool {
		return strings.HasPrefix(t, "switch os.Args[1]") || strings.HasPrefix(t, "switch name")
	})
}

func devDispatchVerbs(b []byte) []string {
	return dispatchVerbs(b, func(t string) bool {
		return strings.HasPrefix(t, "switch argv[0]")
	})
}

// caseLineAliases scans case lines with multiple quoted tokens (e.g. `case "version", "-v", "--version":`)
// and maps each alias token to the canonical declared verb on the same case arm.
func caseLineAliases(b []byte, isOpener func(string) bool, declared map[string]string) map[string]string {
	aliases := map[string]string{}
	inSwitch := false
	depth := 0
	for _, raw := range strings.Split(string(b), "\n") {
		t := strings.TrimSpace(raw)
		if !inSwitch {
			if isOpener(t) {
				inSwitch = true
				depth = 1
			}
			continue
		}
		if depth == 1 {
			if strings.HasPrefix(t, "default:") {
				inSwitch = false
				depth = 0
				continue
			}
			if strings.HasPrefix(t, "case ") && strings.HasSuffix(t, ":") {
				var lineVerbs []string
				canonical := ""
				for _, m := range mainCaseRE.FindAllStringSubmatch(t, -1) {
					v := strings.ToLower(m[1])
					if v != "" {
						lineVerbs = append(lineVerbs, v)
						if declared[v] != "" {
							canonical = v
						} else if _, ok := devindex.TierOf(v); ok {
							canonical = v
						}
					}
				}
				if canonical != "" {
					for _, v := range lineVerbs {
						aliases[v] = canonical
					}
				}
			}
		}
		depth += strings.Count(t, "{") - strings.Count(t, "}")
		if depth <= 0 {
			inSwitch = false
		}
	}
	return aliases
}

// hasVerbTier determines if a verb token is classified in declared tiers, is an alias
// of a declared verb on the same case line, or resolves through devindex.TierOf.
func hasVerbTier(verb string, declared map[string]string, aliases map[string]string) bool {
	v := strings.ToLower(strings.TrimSpace(verb))
	if v == "" {
		return true
	}
	if _, ok := declared[v]; ok {
		return true
	}
	if canon, ok := aliases[v]; ok && declared[canon] != "" {
		return true
	}
	if _, ok := devindex.TierOf(v); ok {
		return true
	}
	return false
}

// gateVerbTierTree audits CLI dispatch verbs in cmd/fak/main.go and cmd/fak-dev/main.go
// against internal/devindex/tiers.go declarations. Emits VERB_UNTIERED findings for any
// dispatched verb that has no tier classification.
func gateVerbTierTree(t *TrackedTree) ([]Finding, error) {
	tiersBody, exists := t.FileBytes(verbTiersFile)
	if !exists {
		return nil, ErrCouldNotRun
	}
	declared, ok := parseDeclaredVerbTiers(tiersBody)
	if !ok {
		return nil, ErrCouldNotRun
	}

	mainBody, exists := t.FileBytes(mainCmdFile)
	if !exists {
		return nil, ErrCouldNotRun
	}
	mainVerbs := mainDispatchVerbs(mainBody)
	mainAliases := caseLineAliases(mainBody, func(t string) bool {
		return strings.HasPrefix(t, "switch os.Args[1]") || strings.HasPrefix(t, "switch name")
	}, declared)

	var devVerbs []string
	var devAliases map[string]string
	if devBody, devExists := t.FileBytes(devCmdFile); devExists {
		devVerbs = devDispatchVerbs(devBody)
		devAliases = caseLineAliases(devBody, func(t string) bool {
			return strings.HasPrefix(t, "switch argv[0]")
		}, declared)
	}

	var findings []Finding
	for _, verb := range mainVerbs {
		if !hasVerbTier(verb, declared, mainAliases) {
			findings = append(findings, Finding{
				Gate: "VERB_UNTIERED",
				File: mainCmdFile,
				Detail: fmt.Sprintf("dispatched verb %q has no tier assigned in %s; classify it in one tier block (TierFrontdoor, TierDev, or TierHidden)", verb, verbTiersFile),
			})
		}
	}
	for _, verb := range devVerbs {
		if !hasVerbTier(verb, declared, devAliases) {
			findings = append(findings, Finding{
				Gate: "VERB_UNTIERED",
				File: devCmdFile,
				Detail: fmt.Sprintf("dispatched verb %q has no tier assigned in %s; classify it in one tier block (TierFrontdoor, TierDev, or TierHidden)", verb, verbTiersFile),
			})
		}
	}

	sort.Slice(findings, func(i, j int) bool {
		if findings[i].File != findings[j].File {
			return findings[i].File < findings[j].File
		}
		return findings[i].Detail < findings[j].Detail
	})
	return findings, nil
}

// ScopeVerbTierFindings applies the committed-push ownership boundary to
// whole-tree VERB_UNTIERED findings. When scoped is false (no remote/trunk read-back),
// it keeps every finding blocking. Only blocks if cmd/fak/main.go, cmd/fak-dev/main.go,
// or internal/devindex/tiers.go is in the push delta; otherwise findings are demoted
// to advisory (#12172).
func ScopeVerbTierFindings(findings []Finding, changedPaths []string, scoped bool) []Finding {
	if !scoped {
		return findings
	}
	touched := false
	for _, path := range changedPaths {
		slash := strings.TrimPrefix(filepath.ToSlash(strings.TrimSpace(path)), "./")
		if slash == mainCmdFile || slash == devCmdFile || slash == verbTiersFile {
			touched = true
			break
		}
	}
	out := append([]Finding(nil), findings...)
	for i := range out {
		if out[i].Gate != "VERB_UNTIERED" {
			continue
		}
		if !touched && !out[i].Advisory {
			out[i].Advisory = true
			out[i].Detail += " This push does not touch CLI dispatch or tier tables; its author must classify it."
		}
	}
	return out
}

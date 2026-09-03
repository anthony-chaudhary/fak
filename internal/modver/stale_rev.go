package modver

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// ModuleRevClaim represents a recalled reference to a module at a specific revision.
type ModuleRevClaim struct {
	Module string `json:"module"`
	Rev    int    `json:"rev"`
	Commit string `json:"commit,omitempty"`
}

// ModuleRevStatus names the freshness verdict for one module revision claim.
type ModuleRevStatus string

const (
	ModuleRevFresh        ModuleRevStatus = "fresh"
	ModuleRevStale        ModuleRevStatus = "stale"
	ModuleRevUnverifiable ModuleRevStatus = "unverifiable"
)

// ModuleRevFinding describes the evaluation of one module revision claim against live state.
type ModuleRevFinding struct {
	Claim       ModuleRevClaim  `json:"claim"`
	Status      ModuleRevStatus `json:"status"`
	CurrentRev  int             `json:"current_rev,omitempty"`
	ReasonClass string          `json:"reason_class,omitempty"`
	Detail      string          `json:"detail,omitempty"`
}

// ModuleRecallAdvisory represents the advisory-first recall gate output for module claims (#2483).
type ModuleRecallAdvisory struct {
	Advisory    bool               `json:"advisory"` // true: advisory-first, does not block by default
	StaleCount  int                `json:"stale_count"`
	Findings    []ModuleRevFinding `json:"findings"`
	ReasonClass string             `json:"reason_class,omitempty"` // ReasonModuleRevStale when stale claims exist
	Message     string             `json:"message,omitempty"`
}

var moduleRevRegex = regexp.MustCompile(`\b([a-zA-Z0-9_\-\./]+)@r(\d+)(?:\+g([0-9a-fA-F]+))?\b`)

// ExtractModuleRevClaims parses text for module version citations (e.g. "internal/gateway@r652+g1f75c56d").
func ExtractModuleRevClaims(text string) []ModuleRevClaim {
	matches := moduleRevRegex.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}
	var claims []ModuleRevClaim
	seen := make(map[string]bool)
	for _, m := range matches {
		mod := strings.TrimSpace(m[1])
		rev, err := strconv.Atoi(m[2])
		if err != nil || rev < 0 {
			continue
		}
		commit := ""
		if len(m) > 3 {
			commit = m[3]
		}
		key := fmt.Sprintf("%s@r%d", mod, rev)
		if !seen[key] {
			seen[key] = true
			claims = append(claims, ModuleRevClaim{
				Module: mod,
				Rev:    rev,
				Commit: commit,
			})
		}
	}
	return claims
}

// CheckModuleRevClaims evaluates claims against an active Report.
func CheckModuleRevClaims(claims []ModuleRevClaim, live *Report) ModuleRecallAdvisory {
	if len(claims) == 0 {
		return ModuleRecallAdvisory{Advisory: true}
	}
	liveMap := make(map[string]Module)
	if live != nil {
		for _, m := range live.Modules {
			liveMap[m.Name] = m
		}
	}

	adv := ModuleRecallAdvisory{
		Advisory: true,
	}

	for _, c := range claims {
		liveMod, ok := liveMap[c.Module]
		if !ok {
			adv.Findings = append(adv.Findings, ModuleRevFinding{
				Claim:  c,
				Status: ModuleRevUnverifiable,
				Detail: fmt.Sprintf("module %q not found in current modules index", c.Module),
			})
			continue
		}

		if c.Rev < liveMod.Rev {
			adv.StaleCount++
			adv.Findings = append(adv.Findings, ModuleRevFinding{
				Claim:       c,
				Status:      ModuleRevStale,
				CurrentRev:  liveMod.Rev,
				ReasonClass: ReasonModuleRevStale,
				Detail:      fmt.Sprintf("recalled module %s@r%d is stale; live trunk holds r%d", c.Module, c.Rev, liveMod.Rev),
			})
		} else {
			adv.Findings = append(adv.Findings, ModuleRevFinding{
				Claim:      c,
				Status:     ModuleRevFresh,
				CurrentRev: liveMod.Rev,
				Detail:     fmt.Sprintf("module %s@r%d is current (live r%d)", c.Module, c.Rev, liveMod.Rev),
			})
		}
	}

	if adv.StaleCount > 0 {
		adv.ReasonClass = ReasonModuleRevStale
		adv.Message = fmt.Sprintf("advisory: %d module revision claim(s) stale against trunk; refresh with fak version modules", adv.StaleCount)
	}

	return adv
}

// CheckRecallText parses text for module citations and checks them against the live report.
func CheckRecallText(text string, live *Report) ModuleRecallAdvisory {
	claims := ExtractModuleRevClaims(text)
	return CheckModuleRevClaims(claims, live)
}

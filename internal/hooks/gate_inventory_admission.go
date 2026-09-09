package hooks

import (
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// gate_inventory_admission.go — the CLEAR_INVENTORY_ADMISSION commit-boundary gate.
//
// Detects newly added mock substitutions and stub panics in staged Go files under internal/,
// cmd/, or pkg/. Hermes' rule: "mocks hide integration bugs". Default to clear inventory gating
// concepts and real shift-left execution rather than stubbing in mocks.
//
// ADVISORY by default (DefaultMode "warn"): it prints which mock/stub was detected and how to
// satisfy the rule (register the concept, attach a runtime proof, or stage a shift-left/e2e
// witness). Set FLEET_INVENTORY_GUARD=block to enforce, or ALLOW_UNINVENTORIED_STUB=1 to skip once.

const (
	gateClearInventoryAdmissionName = "CLEAR_INVENTORY_ADMISSION"
	inventoryModeEnv                = "FLEET_INVENTORY_GUARD"
	inventoryEscapeEnv              = "ALLOW_UNINVENTORIED_STUB"
)

var (
	mockStructRE = regexp.MustCompile(`\btype\s+(?:(?:Mock|Fake)[A-Z]\w*|Mock|Fake)(?:\[[^\]]+\])?\s+struct\b`)
	mockFuncRE   = regexp.MustCompile(`\bfunc\s+(?:\([^)]+\)\s+)?(?:NewMock|NewFake)[A-Za-z0-9_]*\b`)
	stubPanicRE  = regexp.MustCompile(`\bpanic\(\s*"(?i:not implemented|unimplemented|todo)(?:[:\s][^"]*)?"\s*\)`)

	nolintInventoryRE = regexp.MustCompile(`(?i)//\s*nolint:(?:inventory_mock|stub)\b`)

	shiftLeftWitnesses = []string{
		"shift-left-verified:",
		"e2e-verified:",
		"inventory-gated:",
		"concept-verified:",
		"smoke-verified:",
		"smoke-test:",
		"real-world-verified:",
	}
)

func inventoryAdmissionDetail(file string) string {
	return `uninventoried mock substitution or stub detected in "` + file + `" without clear inventory gating concept — ` +
		`Hermes' rule: "mocks hide integration bugs". Default to clear inventory gating concepts and real shift-left execution rather than stubbing in mocks. ` +
		`Register the concept in tools/concept_disambiguation_scorecard.data/, attach a runtime proof in internal/maturity/runtime-proofs.json, ` +
		`or stage an "E2E-verified:" / "Shift-left-verified:" trailer citing the real run. ` +
		`(advisory; FLEET_INVENTORY_GUARD=block enforces, ALLOW_UNINVENTORIED_STUB=1 skips once)`
}

func isInventoryAdmissionScope(file string) bool {
	norm := filepath.ToSlash(file)
	norm = strings.TrimPrefix(norm, "./")
	if !strings.HasSuffix(norm, ".go") {
		return false
	}
	return strings.HasPrefix(norm, "internal/") ||
		strings.HasPrefix(norm, "cmd/") ||
		strings.HasPrefix(norm, "pkg/")
}

func isMockImport(line string) bool {
	return strings.Contains(line, "github.com/golang/mock") ||
		strings.Contains(line, "github.com/stretchr/testify/mock")
}

func fileCarriesNolint(d *StagedDiff, file string, lines []AddedLine) bool {
	for _, al := range lines {
		trimmed := strings.TrimSpace(al.Text)
		if strings.HasPrefix(trimmed, "//") && nolintInventoryRE.MatchString(trimmed) {
			return true
		}
	}
	if b, ok := d.FileBytes(file); ok {
		if nolintInventoryRE.Match(b) {
			return true
		}
	}
	return false
}

func gateClearInventoryAdmission(d *StagedDiff) ([]Finding, error) {
	if d == nil {
		return nil, nil
	}

	hasWitness := false
	for _, al := range d.AddedLines() {
		lower := strings.ToLower(al.Text)
		for _, w := range shiftLeftWitnesses {
			if strings.Contains(lower, w) {
				hasWitness = true
				break
			}
		}
		if hasWitness {
			break
		}
	}

	candidateCount := 0
	var findings []Finding

	for _, file := range d.sortedFiles() {
		norm := filepath.ToSlash(file)
		norm = strings.TrimPrefix(norm, "./")
		if !isInventoryAdmissionScope(norm) {
			continue
		}

		lines := d.AddedByFile[file]
		fileSuppressed := fileCarriesNolint(d, norm, lines)
		isTest := strings.HasSuffix(norm, "_test.go")

		for _, al := range lines {
			trimmed := strings.TrimSpace(al.Text)
			if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "/*") || strings.HasPrefix(trimmed, "*") {
				continue
			}

			isCandidate := false
			if mockStructRE.MatchString(al.Text) {
				isCandidate = true
			} else if mockFuncRE.MatchString(al.Text) {
				isCandidate = true
			} else if isMockImport(al.Text) {
				isCandidate = true
			} else if !isTest && stubPanicRE.MatchString(al.Text) {
				isCandidate = true
			}

			if !isCandidate {
				continue
			}

			candidateCount++

			if !hasWitness && !fileSuppressed && !nolintInventoryRE.MatchString(al.Text) {
				findings = append(findings, Finding{
					Gate:     gateClearInventoryAdmissionName,
					File:     norm,
					Line:     al.New,
					Detail:   inventoryAdmissionDetail(norm),
					Advisory: true,
				})
			}
		}
	}

	d.NoteCandidates(gateClearInventoryAdmissionName, candidateCount, "staged mock/stub candidate(s)")

	if hasWitness || len(findings) == 0 {
		return nil, nil
	}

	sort.Slice(findings, func(i, j int) bool {
		if findings[i].File != findings[j].File {
			return findings[i].File < findings[j].File
		}
		return findings[i].Line < findings[j].Line
	})

	return findings, nil
}

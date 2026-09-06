package readmenext

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var (
	reBoldSpan         = regexp.MustCompile(`\*\*([^*]+)\*\*`)
	reNaive            = regexp.MustCompile(`(?i)\bnaive\b`)
	reTunedOrSOTA      = regexp.MustCompile(`(?i)\b(tuned|sota|already-shipped|reference)\b`)
	reVsNaive          = regexp.MustCompile(`(?i)\b(vs\.?|versus)\s+(a\s+)?naive\b`)
	rePerformanceClaim = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?(?:×|x)\s+vs|\btok/s\b|\bspeedup\b|\bfaster\b|\bless work\b)`)
)

// ValidateFragment audits a candidate fragment against schema, repository, and policy constraints.
func ValidateFragment(f *CandidateFragment, repoRoot string) error {
	if f == nil {
		return errors.New("candidate fragment is nil")
	}

	// 1. Schema name validation
	if f.Schema != SchemaCandidate {
		return fmt.Errorf("invalid schema %q: expected %q", f.Schema, SchemaCandidate)
	}

	// 2. Issue number validation
	if f.Issue <= 0 {
		return fmt.Errorf("invalid issue %d: must be positive integer", f.Issue)
	}

	// 3. Topic slug validation
	if strings.TrimSpace(f.Topic) == "" {
		return errors.New("topic cannot be empty")
	}

	// 4. Candidate content validation
	if strings.TrimSpace(f.CandidateContent) == "" {
		return errors.New("candidate_content cannot be empty")
	}

	// 5. TargetSection validation
	switch f.TargetSection {
	case TargetHardwareTable, TargetHeroHeadline, TargetMemoryOverflow, TargetWhyFak, TargetDefaultPriorities, TargetCustom:
		// Valid section
	default:
		return fmt.Errorf("unsupported target_section %q: must be one of hardware_table, hero_headline, memory_overflow, why_fak, default_priorities, custom", f.TargetSection)
	}

	// 6. Date validation
	if f.Date != "" {
		if _, err := time.Parse("2006-01-02", f.Date); err != nil {
			return fmt.Errorf("invalid date format %q: expected YYYY-MM-DD", f.Date)
		}
	}

	// 7. RetireTarget validation
	action := f.RetireTarget.Action
	if action == "" {
		action = RetireActionNone
	}
	switch action {
	case RetireActionReplaceRow, RetireActionAppendToLegacy:
		if strings.TrimSpace(f.RetireTarget.TargetText) == "" {
			return fmt.Errorf("retire action %q requires non-empty target_text", action)
		}
	case RetireActionNone:
		// Valid non-retiring action
	default:
		return fmt.Errorf("unsupported retire action %q: must be replace_row, append_to_legacy, or none", action)
	}

	if doc := strings.TrimSpace(f.RetireTarget.LegacyArchiveDoc); doc != "" {
		cleaned := filepath.Clean(doc)
		if filepath.IsAbs(doc) || strings.HasPrefix(cleaned, "..") {
			return fmt.Errorf("legacy_archive_doc %q must be a relative path within the repository", doc)
		}
	}

	// Bounded sections requirement: hardware_table and hero_headline cannot grow unbounded.
	// They require an explicit replacement or retirement target.
	if f.TargetSection == TargetHardwareTable || f.TargetSection == TargetHeroHeadline {
		if action == RetireActionNone || strings.TrimSpace(f.RetireTarget.TargetText) == "" {
			return fmt.Errorf("bounded section %q requires a valid RetireTarget (action replace_row or append_to_legacy with non-empty target_text)", f.TargetSection)
		}
	}

	// 8. Witness validations against repository artifacts
	if repoRoot != "" {
		if err := validateWitness(f, repoRoot); err != nil {
			return err
		}
	} else {
		// If repoRoot is empty but witness paths are specified, require repoRoot.
		if strings.TrimSpace(f.Witness.ReceiptPath) != "" ||
			strings.TrimSpace(f.Witness.AuthorityEntry) != "" ||
			strings.TrimSpace(f.Witness.HardwareJSONRow) != "" {
			return errors.New("repoRoot cannot be empty when verifying witness paths")
		}
	}

	// 9. SOTA-vs-us law and naive baseline validation
	if err := validateSOTALaw(f); err != nil {
		return err
	}

	return nil
}

func validateWitness(f *CandidateFragment, repoRoot string) error {
	// Verify ReceiptPath if provided
	if receipt := strings.TrimSpace(f.Witness.ReceiptPath); receipt != "" {
		cleaned := filepath.Clean(receipt)
		if filepath.IsAbs(receipt) || strings.HasPrefix(cleaned, "..") {
			return fmt.Errorf("witness receipt_path %q must be a relative path within the repository", receipt)
		}
		fullPath := filepath.Join(repoRoot, cleaned)
		if _, err := os.Stat(fullPath); err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("witness receipt_path %q does not exist relative to repoRoot: %s", receipt, fullPath)
			}
			return fmt.Errorf("failed to access receipt_path %q: %w", receipt, err)
		}
	}

	// Verify AuthorityEntry if provided
	if authEntry := strings.TrimSpace(f.Witness.AuthorityEntry); authEntry != "" {
		authPath := filepath.Join(repoRoot, DefaultBenchmarkAuthorityPath)
		data, err := os.ReadFile(authPath)
		if err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("benchmark authority doc not found at %q: %w", authPath, err)
			}
			return fmt.Errorf("failed to read benchmark authority doc %q: %w", authPath, err)
		}

		// Extract key from anchor if formatted as "BENCHMARK-AUTHORITY.md#key" or "#key"
		key := authEntry
		if idx := strings.Index(authEntry, "#"); idx != -1 {
			key = authEntry[idx+1:]
		}
		key = strings.TrimSpace(key)
		if key != "" && !strings.Contains(string(data), key) {
			return fmt.Errorf("authority_entry %q (key %q) not found in %s", authEntry, key, DefaultBenchmarkAuthorityPath)
		}
	}

	// Verify HardwareJSONRow if provided
	if hwRow := strings.TrimSpace(f.Witness.HardwareJSONRow); hwRow != "" {
		hwPath := filepath.Join(repoRoot, DefaultHardwareJSONPath)
		data, err := os.ReadFile(hwPath)
		if err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("hardware latest manifest not found at %q: %w", hwPath, err)
			}
			return fmt.Errorf("failed to access hardware latest manifest %q: %w", hwPath, err)
		}
		var manifest HardwareLatestManifest
		if err := json.Unmarshal(data, &manifest); err != nil {
			return fmt.Errorf("malformed hardware latest manifest: %w", err)
		}
		if _, ok := manifest.Platforms[hwRow]; !ok {
			return fmt.Errorf("hardware platform %q not found in %s", hwRow, DefaultHardwareJSONPath)
		}
	}

	return nil
}

func validateSOTALaw(f *CandidateFragment) error {
	content := f.CandidateContent

	// Check bolded headlines for naive baseline leads
	matches := reBoldSpan.FindAllStringSubmatch(content, -1)
	for _, m := range matches {
		body := m[1]
		if reNaive.MatchString(body) && !reTunedOrSOTA.MatchString(body) {
			return fmt.Errorf("candidate content violates SOTA-vs-us law: bold headline leads with a naive baseline: %q", body)
		}
	}

	// Check content-level naive comparisons (e.g. "vs naive" without tuned or sota counterpart)
	if reVsNaive.MatchString(content) && !reTunedOrSOTA.MatchString(content) {
		return fmt.Errorf("candidate content violates SOTA-vs-us law: comparative claim compares against naive without tuned/SOTA baseline: %q", content)
	}

	// Check comparative performance claims against SOTAComparison checklist
	if rePerformanceClaim.MatchString(content) && !f.LawsChecklist.SOTAComparison {
		return errors.New("candidate content makes a comparative performance claim but LawsChecklist.SOTAComparison is false")
	}

	return nil
}

package readmenext

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var (
	reHardwareHeader = regexp.MustCompile(`(## Latest hardware results\s*—\s*)\d{4}-\d{2}-\d{2}`)
	reVerifiedMarker = regexp.MustCompile(`(<!--\s*readme-verified:\s*)\d{4}-\d{2}-\d{2}`)
)

// PreviewNext renders the simulated README with fragments applied and returns
// the transformed content alongside a list of changes and retired items.
func PreviewNext(readmeContent string, fragments []*CandidateFragment) (string, []string, error) {
	result := readmeContent
	var changes []string

	for _, f := range fragments {
		if f == nil {
			continue
		}

		action := f.RetireTarget.Action
		if action == "" {
			action = RetireActionNone
		}
		targetText := f.RetireTarget.TargetText

		switch action {
		case RetireActionReplaceRow:
			if !strings.Contains(result, targetText) {
				return "", nil, fmt.Errorf("fragment %q (issue #%d): target_text %q not found in README", f.Topic, f.Issue, targetText)
			}
			result = strings.Replace(result, targetText, f.CandidateContent, 1)
			changes = append(changes, fmt.Sprintf("[%s] Replaced row in %s: %q -> %q", f.Topic, f.TargetSection, targetText, f.CandidateContent))
			changes = append(changes, fmt.Sprintf("[retired] %s", targetText))

		case RetireActionAppendToLegacy:
			if !strings.Contains(result, targetText) {
				return "", nil, fmt.Errorf("fragment %q (issue #%d): target_text %q not found in README", f.Topic, f.Issue, targetText)
			}
			if strings.TrimSpace(f.CandidateContent) != "" {
				result = strings.Replace(result, targetText, f.CandidateContent, 1)
				changes = append(changes, fmt.Sprintf("[%s] Replaced content in %s and queued old for legacy: %q -> %q", f.Topic, f.TargetSection, targetText, f.CandidateContent))
			} else {
				result = strings.Replace(result, targetText, "", 1)
				changes = append(changes, fmt.Sprintf("[%s] Removed content from %s and queued for legacy: %q", f.Topic, f.TargetSection, targetText))
			}
			changes = append(changes, fmt.Sprintf("[retired_to_legacy] %s", targetText))

		case RetireActionNone:
			if targetText != "" {
				if !strings.Contains(result, targetText) {
					return "", nil, fmt.Errorf("fragment %q (issue #%d): target_text %q not found in README", f.Topic, f.Issue, targetText)
				}
				result = strings.Replace(result, targetText, f.CandidateContent, 1)
				changes = append(changes, fmt.Sprintf("[%s] Replaced %q in %s", f.Topic, targetText, f.TargetSection))
			} else {
				var err error
				result, err = applySectionFragment(result, f, &changes)
				if err != nil {
					return "", nil, err
				}
			}
		default:
			return "", nil, fmt.Errorf("fragment %q: unsupported retire action %q", f.Topic, action)
		}

		// Update hardware section date if fragment is hardware_table and specifies date
		if f.TargetSection == TargetHardwareTable && f.Date != "" {
			if reHardwareHeader.MatchString(result) {
				result = reHardwareHeader.ReplaceAllString(result, "${1}"+f.Date)
				changes = append(changes, fmt.Sprintf("[%s] Updated hardware results header date to %s", f.Topic, f.Date))
			}
			if reVerifiedMarker.MatchString(result) {
				result = reVerifiedMarker.ReplaceAllString(result, "${1}"+f.Date)
			}
		}
	}

	return result, changes, nil
}

func applySectionFragment(content string, f *CandidateFragment, changes *[]string) (string, error) {
	switch f.TargetSection {
	case TargetHardwareTable:
		return applyHardwareRow(content, f, changes)
	case TargetHeroHeadline:
		return applyHeroHeadline(content, f, changes)
	case TargetWhyFak:
		res, err := appendToMarkdownSection(content, "## Why run coding agents on fak", f.CandidateContent)
		if err != nil {
			return "", err
		}
		*changes = append(*changes, fmt.Sprintf("[%s] Appended bullet/item to why_fak", f.Topic))
		return res, nil
	case TargetMemoryOverflow:
		res, err := appendToMarkdownSection(content, "## Open-source memory overflow landscape", f.CandidateContent)
		if err != nil {
			return "", err
		}
		*changes = append(*changes, fmt.Sprintf("[%s] Appended item to memory_overflow", f.Topic))
		return res, nil
	case TargetDefaultPriorities:
		res, err := appendToMarkdownSection(content, "## Default priorities & operating modes", f.CandidateContent)
		if err != nil {
			return "", err
		}
		*changes = append(*changes, fmt.Sprintf("[%s] Appended item to default_priorities", f.Topic))
		return res, nil
	case TargetCustom:
		res := strings.TrimRight(content, "\n") + "\n\n" + f.CandidateContent + "\n"
		*changes = append(*changes, fmt.Sprintf("[%s] Appended custom candidate content", f.Topic))
		return res, nil
	default:
		return "", fmt.Errorf("unsupported section %q for auto-placement without target_text", f.TargetSection)
	}
}

func applyHardwareRow(content string, f *CandidateFragment, changes *[]string) (string, error) {
	candidate := strings.TrimSpace(f.CandidateContent)
	if !strings.HasPrefix(candidate, "|") {
		return "", fmt.Errorf("hardware_table candidate content must be a markdown table row starting with '|': %q", candidate)
	}

	cols := strings.Split(candidate, "|")
	if len(cols) < 3 {
		return "", fmt.Errorf("hardware_table candidate row does not contain valid columns: %q", candidate)
	}
	platform := strings.TrimSpace(cols[1])

	// Try finding existing platform row in the hardware results table
	lines := strings.Split(content, "\n")
	foundPlatform := false
	inHwSection := false

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "## Latest hardware results") {
			inHwSection = true
			continue
		}
		if inHwSection && strings.HasPrefix(trimmed, "## ") {
			// Left hardware section
			break
		}
		if inHwSection && strings.HasPrefix(trimmed, "|") {
			rowCols := strings.Split(trimmed, "|")
			if len(rowCols) > 1 && strings.TrimSpace(rowCols[1]) == platform {
				oldRow := line
				lines[i] = candidate
				foundPlatform = true
				*changes = append(*changes, fmt.Sprintf("[%s] Replaced platform row for %s: %q -> %q", f.Topic, platform, oldRow, candidate))
				*changes = append(*changes, fmt.Sprintf("[retired] %s", oldRow))
				break
			}
		}
	}

	if foundPlatform {
		return strings.Join(lines, "\n"), nil
	}

	// If platform not found, append to table in hardware results section
	res, err := appendRowToTableInSection(content, "## Latest hardware results", candidate)
	if err != nil {
		return "", err
	}
	*changes = append(*changes, fmt.Sprintf("[%s] Appended new hardware row for %s", f.Topic, platform))
	return res, nil
}

func applyHeroHeadline(content string, f *CandidateFragment, changes *[]string) (string, error) {
	candidate := strings.TrimSpace(f.CandidateContent)
	lines := strings.Split(content, "\n")
	foundH1 := false

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "# ") {
			foundH1 = true
			continue
		}
		if foundH1 && strings.HasPrefix(trimmed, "**") && strings.HasSuffix(trimmed, "**") {
			oldHeadline := line
			lines[i] = candidate
			*changes = append(*changes, fmt.Sprintf("[%s] Replaced hero headline: %q -> %q", f.Topic, oldHeadline, candidate))
			*changes = append(*changes, fmt.Sprintf("[retired] %s", oldHeadline))
			return strings.Join(lines, "\n"), nil
		}
	}

	// Fallback: replace any bold lead paragraph after H1
	return "", fmt.Errorf("unable to locate existing hero headline paragraph to replace")
}

func appendToMarkdownSection(content string, sectionHeader string, item string) (string, error) {
	idx := strings.Index(content, sectionHeader)
	if idx == -1 {
		return "", fmt.Errorf("section %q not found in content", sectionHeader)
	}

	// Find the end of this section (start of next ## or end of content)
	afterHeader := content[idx+len(sectionHeader):]
	nextSectionIdx := strings.Index(afterHeader, "\n## ")

	itemFormatted := "\n" + strings.TrimSpace(item) + "\n"

	if nextSectionIdx == -1 {
		// Append at the end
		return strings.TrimRight(content, "\n") + "\n" + itemFormatted, nil
	}

	insertPoint := idx + len(sectionHeader) + nextSectionIdx
	return content[:insertPoint] + itemFormatted + content[insertPoint:], nil
}

func appendRowToTableInSection(content string, sectionHeader string, row string) (string, error) {
	idx := strings.Index(content, sectionHeader)
	if idx == -1 {
		return "", fmt.Errorf("section %q not found in content", sectionHeader)
	}

	lines := strings.Split(content, "\n")
	inSection := false
	lastTableRowIdx := -1

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, sectionHeader) {
			inSection = true
			continue
		}
		if inSection && strings.HasPrefix(trimmed, "## ") {
			break
		}
		if inSection && strings.HasPrefix(trimmed, "|") {
			lastTableRowIdx = i
		}
	}

	if lastTableRowIdx == -1 {
		return "", fmt.Errorf("no table found in section %q", sectionHeader)
	}

	newLines := make([]string, 0, len(lines)+1)
	newLines = append(newLines, lines[:lastTableRowIdx+1]...)
	newLines = append(newLines, row)
	newLines = append(newLines, lines[lastTableRowIdx+1:]...)

	return strings.Join(newLines, "\n"), nil
}

// Publish coordinates the validation, preview, and atomic file-level publication
// of candidate fragments to README.md, docs/README-legacy.md, and hardware-latest.json.
func Publish(repoRoot string, fragments []*CandidateFragment, dryRun bool) (*PublishResult, error) {
	readmePath := filepath.Join(repoRoot, DefaultReadmePath)
	data, err := os.ReadFile(readmePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read README at %q: %w", readmePath, err)
	}
	readmeContent := string(data)

	// Validate all fragments first
	for _, f := range fragments {
		if err := ValidateFragment(f, repoRoot); err != nil {
			return nil, fmt.Errorf("validation failed for fragment %q (issue #%d): %w", f.Topic, f.Issue, err)
		}
	}

	// Generate preview
	simulatedReadme, changes, err := PreviewNext(readmeContent, fragments)
	if err != nil {
		return nil, fmt.Errorf("preview synthesis failed: %w", err)
	}

	// Collect retired items and legacy appends
	var retiredItems []string
	legacyAppendsByDoc := make(map[string][]string)

	for _, f := range fragments {
		action := f.RetireTarget.Action
		targetText := strings.TrimSpace(f.RetireTarget.TargetText)
		if (action == RetireActionAppendToLegacy || action == RetireActionReplaceRow) && targetText != "" {
			retiredItems = append(retiredItems, targetText)
			if action == RetireActionAppendToLegacy {
				doc := f.RetireTarget.LegacyArchiveDoc
				if doc == "" {
					doc = DefaultLegacyArchiveDoc
				}
				dateStr := f.Date
				if dateStr == "" {
					dateStr = time.Now().Format("2006-01-02")
				}
				entry := fmt.Sprintf("### Retired from README.md (%s, Issue #%d: %s)\n\n%s\n", dateStr, f.Issue, f.Topic, targetText)
				legacyAppendsByDoc[doc] = append(legacyAppendsByDoc[doc], entry)
			}
		}
	}

	// Check hardware-latest manifest updates
	hwPath := filepath.Join(repoRoot, DefaultHardwareJSONPath)
	var hwManifest HardwareLatestManifest
	var hardwareJSONUpdated bool

	if hwData, err := os.ReadFile(hwPath); err == nil {
		if err := json.Unmarshal(hwData, &hwManifest); err == nil {
			if hwManifest.Platforms == nil {
				hwManifest.Platforms = make(map[string]*HardwarePlatformEntry)
			}
		}
	}
	if hwManifest.Schema == "" {
		hwManifest.Schema = "fak-hardware-latest/1"
		hwManifest.Platforms = make(map[string]*HardwarePlatformEntry)
	}

	for _, f := range fragments {
		if f.Witness.HardwareJSONRow != "" || f.TargetSection == TargetHardwareTable {
			platform := strings.TrimSpace(f.Witness.HardwareJSONRow)
			if platform == "" {
				cols := strings.Split(f.CandidateContent, "|")
				if len(cols) > 2 {
					platform = strings.TrimSpace(cols[1])
				}
			}

			if platform != "" {
				observed := f.Date
				if observed == "" {
					observed = time.Now().Format("2006-01-02")
				}
				detail := f.Witness.ReceiptPath
				if detail == "" && hwManifest.Platforms[platform] != nil {
					detail = hwManifest.Platforms[platform].Detail
				}

				hwManifest.Platforms[platform] = &HardwarePlatformEntry{
					Observed: observed,
					Detail:   detail,
					Row:      strings.TrimSpace(f.CandidateContent),
				}
				hwManifest.AsOf = observed
				hardwareJSONUpdated = true
			}
		}
	}

	primaryLegacyPath := ""
	if len(legacyAppendsByDoc) > 0 {
		for docPath := range legacyAppendsByDoc {
			primaryLegacyPath = filepath.Join(repoRoot, docPath)
			break
		}
	}

	result := &PublishResult{
		Success:             true,
		DryRun:              dryRun,
		ReadmePath:          readmePath,
		LegacyPath:          primaryLegacyPath,
		HardwareJSONPath:    hwPath,
		HardwareJSONUpdated: hardwareJSONUpdated,
		AppliedFragments:    len(fragments),
		Changes:             changes,
		RetiredItems:        retiredItems,
	}

	if dryRun {
		return result, nil
	}

	// Non-dry-run: commit mutations to disk with atomic rollback on failure
	var rollbackErr error
	defer func() {
		if rollbackErr != nil {
			// Restore original README content
			_ = os.WriteFile(readmePath, data, 0644)
		}
	}()

	if err := os.WriteFile(readmePath, []byte(simulatedReadme), 0644); err != nil {
		return nil, fmt.Errorf("failed to write updated README at %q: %w", readmePath, err)
	}

	// Write legacy archives
	for docRelPath, entries := range legacyAppendsByDoc {
		targetFile := filepath.Join(repoRoot, docRelPath)
		if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
			rollbackErr = err
			return nil, fmt.Errorf("failed to create directory for legacy doc %q: %w", targetFile, err)
		}

		f, err := os.OpenFile(targetFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			rollbackErr = err
			return nil, fmt.Errorf("failed to open legacy doc %q: %w", targetFile, err)
		}
		for _, entry := range entries {
			if _, err := f.WriteString("\n" + entry + "\n"); err != nil {
				f.Close()
				rollbackErr = err
				return nil, fmt.Errorf("failed to write to legacy doc %q: %w", targetFile, err)
			}
		}
		f.Close()
	}

	// Write hardware manifest
	if hardwareJSONUpdated {
		if err := os.MkdirAll(filepath.Dir(hwPath), 0755); err != nil {
			rollbackErr = err
			return nil, fmt.Errorf("failed to create directory for hardware manifest: %w", err)
		}
		formattedJSON, err := json.MarshalIndent(&hwManifest, "", "  ")
		if err != nil {
			rollbackErr = err
			return nil, fmt.Errorf("failed to marshal hardware manifest: %w", err)
		}
		formattedJSON = append(formattedJSON, '\n')
		if err := os.WriteFile(hwPath, formattedJSON, 0644); err != nil {
			rollbackErr = err
			return nil, fmt.Errorf("failed to write hardware manifest at %q: %w", hwPath, err)
		}
	}

	return result, nil
}

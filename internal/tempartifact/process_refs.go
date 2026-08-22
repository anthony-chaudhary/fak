package tempartifact

import (
	"path/filepath"
	"strings"
)

type processRecord struct {
	ExecutablePath string
	CommandLine    string
}

func referencesFromProcessRecords(records []processRecord, candidates []string) map[string]bool {
	references := map[string]bool{}
	for _, candidate := range candidates {
		for _, record := range records {
			if windowsPathEqual(record.ExecutablePath, candidate) {
				references[pathKey(candidate)] = true
				break
			}
			for _, arg := range splitWindowsCommandLine(record.CommandLine) {
				value := arg
				if index := strings.IndexByte(value, '='); index >= 0 {
					value = value[index+1:]
				}
				if windowsPathEqual(value, candidate) {
					references[pathKey(candidate)] = true
					break
				}
			}
			if references[pathKey(candidate)] {
				break
			}
		}
	}
	return references
}

func windowsPathEqual(left, right string) bool {
	left = strings.TrimSpace(left)
	right = strings.TrimSpace(right)
	if left == "" || right == "" {
		return false
	}
	left = strings.ReplaceAll(filepath.Clean(left), "/", "\\")
	right = strings.ReplaceAll(filepath.Clean(right), "/", "\\")
	return strings.EqualFold(left, right)
}

// splitWindowsCommandLine implements the CommandLineToArgvW quote/backslash
// rules without loading shell32, which keeps the matching logic testable on
// every host.
func splitWindowsCommandLine(line string) []string {
	var args []string
	for index := 0; index < len(line); {
		for index < len(line) && (line[index] == ' ' || line[index] == '\t') {
			index++
		}
		if index == len(line) {
			break
		}
		var arg strings.Builder
		quoted := false
		for index < len(line) {
			if !quoted && (line[index] == ' ' || line[index] == '\t') {
				break
			}
			backslashes := 0
			for index < len(line) && line[index] == '\\' {
				backslashes++
				index++
			}
			if index < len(line) && line[index] == '"' {
				arg.WriteString(strings.Repeat("\\", backslashes/2))
				if backslashes%2 == 0 {
					quoted = !quoted
				} else {
					arg.WriteByte('"')
				}
				index++
				continue
			}
			arg.WriteString(strings.Repeat("\\", backslashes))
			if index < len(line) {
				arg.WriteByte(line[index])
				index++
			}
		}
		args = append(args, arg.String())
	}
	return args
}

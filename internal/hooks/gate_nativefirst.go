package hooks

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/nativefirst"
)

var nativeFirstExtensions = map[string]bool{".go": true, ".md": true, ".txt": true, ".toml": true, ".json": true, ".yaml": true, ".yml": true}

func checkNativeFirst(d *StagedDiff) ([]Finding, error) {
	var out []Finding
	candidates := 0
	for path, lines := range d.AddedByFile {
		if !nativeFirstExtensions[strings.ToLower(filepath.Ext(path))] {
			continue
		}
		candidates += len(lines)
		for _, line := range lines {
			finding := nativefirst.ScanLine(line.Text)
			if finding == nil {
				continue
			}
			out = append(out, Finding{
				File:   path,
				Line:   line.New,
				Detail: fmt.Sprintf("%s; phrase %q", finding.Reason, finding.Phrase),
			})
		}
	}
	d.NoteCandidates("NATIVE_FIRST", candidates, "added line(s) in native-first-scannable files")
	return out, nil
}

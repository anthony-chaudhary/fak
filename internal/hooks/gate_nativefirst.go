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
	for path, lines := range d.AddedByFile {
		if !nativeFirstExtensions[strings.ToLower(filepath.Ext(path))] {
			continue
		}
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
	return out, nil
}

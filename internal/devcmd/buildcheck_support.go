package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

type pathList []string

func (p *pathList) String() string { return strings.Join(*p, ",") }
func (p *pathList) Set(v string) error {
	v = strings.TrimSpace(v)
	if v == "" {
		return fmt.Errorf("empty --path")
	}
	*p = append(*p, v)
	return nil
}
func writeIndentedJSONNoEscape(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(v)
}
func repoSlash(path string) string { return strings.ReplaceAll(filepath.ToSlash(path), "\\", "/") }

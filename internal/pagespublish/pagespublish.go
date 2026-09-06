package pagespublish

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Report is the machine-readable witness emitted before and after a Pages build.
type Report struct {
	Mode  string `json:"mode"`
	Root  string `json:"root"`
	Files int    `json:"files"`
	Pages int    `json:"pages"`
	Bytes int64  `json:"bytes"`
}

type Manifest struct {
	Schema string         `json:"schema"`
	Files  []ManifestFile `json:"files"`
}

type ManifestFile struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

func AuditSource(root string) (Report, error) {
	report := Report{Mode: "source", Root: filepath.ToSlash(root)}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		report.Files++
		info, err := d.Info()
		if err != nil {
			return err
		}
		report.Bytes += info.Size()
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".md" && ext != ".html" && ext != ".yml" && ext != ".yaml" && ext != ".xml" && ext != ".txt" {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if !utf8.Valid(b) {
			return fmt.Errorf("%s: source is not valid UTF-8", filepath.ToSlash(path))
		}
		if ext == ".md" || ext == ".html" {
			report.Pages++
			if err := checkLiquidSyntax(path, b); err != nil {
				return err
			}
		}
		return nil
	})
	return report, err
}

func AuditArtifact(root, baseURL string, minimumPages int, required []string, writeManifest bool) (Report, error) {
	report := Report{Mode: "artifact", Root: filepath.ToSlash(root)}
	manifest := Manifest{Schema: "fak-pages-manifest/1"}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if rel == ".pages-manifest.json" {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		report.Files++
		report.Bytes += int64(len(b))
		if strings.HasSuffix(strings.ToLower(rel), ".html") {
			report.Pages++
		}
		sum := sha256.Sum256(b)
		manifest.Files = append(manifest.Files, ManifestFile{Path: rel, Bytes: int64(len(b)), SHA256: hex.EncodeToString(sum[:])})
		return nil
	})
	if err != nil {
		return report, err
	}
	if report.Pages < minimumPages {
		return report, fmt.Errorf("artifact has %d HTML pages; require at least %d", report.Pages, minimumPages)
	}
	for _, rel := range required {
		path := filepath.Join(root, filepath.FromSlash(rel))
		b, err := os.ReadFile(path)
		if err != nil {
			return report, fmt.Errorf("required page %s: %w", rel, err)
		}
		if strings.HasSuffix(strings.ToLower(rel), ".html") {
			text := string(b)
			for _, needle := range []string{"<title>", `name="description"`, `rel="canonical"`} {
				if !strings.Contains(text, needle) {
					return report, fmt.Errorf("required page %s lacks %s", rel, needle)
				}
			}
		}
	}
	sitemap, err := os.ReadFile(filepath.Join(root, "sitemap.xml"))
	if err != nil {
		return report, fmt.Errorf("sitemap.xml: %w", err)
	}
	if baseURL != "" && !strings.Contains(string(sitemap), baseURL) {
		return report, fmt.Errorf("sitemap.xml does not contain base URL %q", baseURL)
	}
	if writeManifest {
		sort.Slice(manifest.Files, func(i, j int) bool { return manifest.Files[i].Path < manifest.Files[j].Path })
		b, err := json.MarshalIndent(manifest, "", "  ")
		if err != nil {
			return report, err
		}
		b = append(b, '\n')
		if err := os.WriteFile(filepath.Join(root, ".pages-manifest.json"), b, 0o644); err != nil {
			return report, err
		}
	}
	return report, nil
}

func WriteJSON(report Report) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	return enc.Encode(report)
}

func CountSitemapURLs(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 0
	for s.Scan() {
		count += strings.Count(s.Text(), "<url>")
	}
	if err := s.Err(); err != nil {
		return 0, err
	}
	if count == 0 {
		return 0, errors.New("sitemap contains no URLs")
	}
	return count, nil
}

func checkLiquidSyntax(path string, b []byte) error {
	s := bufio.NewScanner(bytes.NewReader(b))
	lineNum := 0
	inRaw := false
	for s.Scan() {
		lineNum++
		line := s.Text()
		if strings.Contains(line, "{% raw %}") {
			inRaw = true
		}
		if strings.Contains(line, "{% endraw %}") {
			inRaw = false
		}
		if inRaw {
			continue
		}
		if strings.Contains(line, "{{") && !strings.Contains(line, "}}") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "|") {
				return fmt.Errorf("%s:%d: unterminated Liquid tag {{ in markdown table row", filepath.ToSlash(path), lineNum)
			}
		}
	}
	return s.Err()
}

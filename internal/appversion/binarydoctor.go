package appversion

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	SeverityOK   = "ok"
	SeverityWarn = "warn"
)

type BinaryRecommendation struct {
	Check     string `json:"check"`
	Severity  string `json:"severity"`
	Finding   string `json:"finding"`
	Recommend string `json:"recommend,omitempty"`
}

type BinaryReport struct {
	Executable      string                 `json:"executable"`
	Images          []BinaryImage          `json:"images"`
	Recommendations []BinaryRecommendation `json:"recommendations"`
	Findings        int                    `json:"findings"`
}

type BinaryImage struct {
	Path        string `json:"path"`
	Exists      bool   `json:"exists"`
	Current     bool   `json:"current,omitempty"`
	Size        int64  `json:"size,omitempty"`
	ModTime     string `json:"mod_time,omitempty"`
	SHA256      string `json:"sha256,omitempty"`
	ReadError   string `json:"read_error,omitempty"`
	SameCurrent bool   `json:"same_as_current,omitempty"`
	Newer       bool   `json:"newer_than_current,omitempty"`
}

func DefaultBinaryDoctorCandidates(exe string) []string {
	dir := filepath.Dir(exe)
	return uniqueStrings([]string{
		exe,
		filepath.Join(dir, "fak"),
		filepath.Join(dir, "fak.exe"),
	})
}

func DiagnoseBinary(exe string, candidates []string) BinaryReport {
	exe, _ = filepath.Abs(exe)
	rep := BinaryReport{Executable: exe}
	current := readBinaryImage(exe, exe)
	rep.Images = append(rep.Images, current)
	for _, c := range uniqueStrings(candidates) {
		abs, _ := filepath.Abs(c)
		if samePath(abs, exe) {
			continue
		}
		rep.Images = append(rep.Images, readBinaryImage(abs, exe))
	}
	sort.SliceStable(rep.Images, func(i, j int) bool {
		if rep.Images[i].Current != rep.Images[j].Current {
			return rep.Images[i].Current
		}
		return rep.Images[i].Path < rep.Images[j].Path
	})

	byPath := map[string]BinaryImage{}
	for _, img := range rep.Images {
		byPath[cleanPathKey(img.Path)] = img
	}
	fak := byPath[cleanPathKey(filepath.Join(filepath.Dir(exe), "fak"))]
	fakExe := byPath[cleanPathKey(filepath.Join(filepath.Dir(exe), "fak.exe"))]
	if fak.Exists && fakExe.Exists && fak.SHA256 != "" && fakExe.SHA256 != "" && fak.SHA256 != fakExe.SHA256 {
		newer := "fak"
		if !fak.Newer && fakExe.Newer {
			newer = "fak.exe"
		}
		rep.Recommendations = append(rep.Recommendations, BinaryRecommendation{
			Check:    "binary-shadow",
			Severity: SeverityWarn,
			Finding:  fmt.Sprintf("sibling fak and fak.exe differ; %s is newer", newer),
			Recommend: "rebuild or replace the stale sibling after live holders exit. On Windows, " +
				"PowerShell resolves `fak`/`.\\fak` to fak.exe, so a stale fak.exe can make " +
				"operators run old sweep/commit logic while the extensionless fak is current.",
		})
	}
	for _, img := range rep.Images {
		if img.Current || !img.Exists || img.SHA256 == "" || current.SHA256 == "" {
			continue
		}
		if img.SHA256 != current.SHA256 && img.Newer {
			rep.Recommendations = append(rep.Recommendations, BinaryRecommendation{
				Check:    "binary-current",
				Severity: SeverityWarn,
				Finding:  fmt.Sprintf("running %s is older/different than %s", filepath.Base(exe), filepath.Base(img.Path)),
				Recommend: "rerun with the newer sibling or replace the stale executable once no live " +
					"process holds it. This prevents diagnostics such as `fak sweep` from reporting " +
					"old lane/junk classifications.",
			})
			break
		}
	}
	if len(rep.Recommendations) == 0 {
		rep.Recommendations = append(rep.Recommendations, BinaryRecommendation{
			Check:    "binary-shadow",
			Severity: SeverityOK,
			Finding:  "no newer differing fak/fak.exe sibling detected",
		})
	}
	for _, r := range rep.Recommendations {
		if r.Severity == SeverityWarn {
			rep.Findings++
		}
	}
	return rep
}

func readBinaryImage(path, exe string) BinaryImage {
	img := BinaryImage{Path: path, Current: samePath(path, exe)}
	st, err := os.Stat(path)
	if err != nil {
		return img
	}
	img.Exists = true
	img.Size = st.Size()
	img.ModTime = st.ModTime().UTC().Format(time.RFC3339)
	if !img.Current {
		if cur, err := os.Stat(exe); err == nil {
			img.Newer = st.ModTime().After(cur.ModTime())
		}
	}
	b, err := os.ReadFile(path)
	if err != nil {
		img.ReadError = err.Error()
		return img
	}
	sum := sha256.Sum256(b)
	img.SHA256 = fmt.Sprintf("%x", sum[:])
	if !img.Current {
		if cur, err := os.ReadFile(exe); err == nil {
			curSum := sha256.Sum256(cur)
			img.SameCurrent = img.SHA256 == fmt.Sprintf("%x", curSum[:])
		}
	}
	return img
}

func uniqueStrings(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		k := cleanPathKey(s)
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, s)
	}
	return out
}

func samePath(a, b string) bool {
	return cleanPathKey(a) == cleanPathKey(b)
}

func cleanPathKey(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		abs = p
	}
	return strings.ToLower(filepath.Clean(abs))
}

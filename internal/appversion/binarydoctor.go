package appversion

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
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
	Executable       string                 `json:"executable"`
	Images           []BinaryImage          `json:"images"`
	Processes        []BinaryProcess        `json:"processes,omitempty"`
	ProcessScanError string                 `json:"process_scan_error,omitempty"`
	Recommendations  []BinaryRecommendation `json:"recommendations"`
	Findings         int                    `json:"findings"`
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

type BinaryProcess struct {
	PID         int    `json:"pid"`
	Path        string `json:"path"`
	Command     string `json:"command,omitempty"`
	Current     bool   `json:"current,omitempty"`
	SHA256      string `json:"sha256,omitempty"`
	SameCurrent bool   `json:"same_as_current,omitempty"`
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
	return DiagnoseBinaryWithProcesses(exe, candidates, nil, "")
}

func DiagnoseBinaryWithProcesses(exe string, candidates []string, processes []BinaryProcess, processScanError string) BinaryReport {
	exe, _ = filepath.Abs(exe)
	rep := BinaryReport{Executable: exe, ProcessScanError: strings.TrimSpace(processScanError)}
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
	rep.Processes = annotateBinaryProcesses(exe, current.SHA256, rep.Images, processes)
	staleLive := 0
	for _, p := range rep.Processes {
		if p.Current || p.SameCurrent || p.SHA256 == "" || current.SHA256 == "" {
			continue
		}
		if p.SHA256 != current.SHA256 {
			staleLive++
		}
	}
	if staleLive > 0 {
		rep.Recommendations = append(rep.Recommendations, BinaryRecommendation{
			Check:    "binary-live-process",
			Severity: SeverityWarn,
			Finding:  fmt.Sprintf("%d live fak process(es) are running a different sibling image", staleLive),
			Recommend: "let those processes exit or stop them deliberately before replacing the stale image. " +
				"On Windows a live process can keep fak.exe locked, leaving operators on old commit/sweep logic.",
		})
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

func CollectBinaryProcesses(candidates []string) ([]BinaryProcess, string) {
	candidates = uniqueStrings(candidates)
	if runtime.GOOS == "windows" {
		return collectBinaryProcessesWindows(candidates)
	}
	if runtime.GOOS == "linux" {
		return collectBinaryProcessesLinux(candidates)
	}
	return nil, "process inventory unavailable on " + runtime.GOOS
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

func annotateBinaryProcesses(exe, currentSHA string, images []BinaryImage, processes []BinaryProcess) []BinaryProcess {
	if len(processes) == 0 {
		return nil
	}
	byPath := map[string]BinaryImage{}
	for _, img := range images {
		byPath[cleanPathKey(img.Path)] = img
	}
	out := make([]BinaryProcess, 0, len(processes))
	for _, p := range processes {
		if p.PID <= 0 || strings.TrimSpace(p.Path) == "" {
			continue
		}
		abs, _ := filepath.Abs(p.Path)
		p.Path = abs
		p.Current = samePath(abs, exe)
		if img, ok := byPath[cleanPathKey(abs)]; ok {
			p.SHA256 = img.SHA256
		}
		if p.SHA256 == "" {
			p.SHA256 = hashFile(abs)
		}
		p.SameCurrent = p.Current || (p.SHA256 != "" && currentSHA != "" && p.SHA256 == currentSHA)
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].PID < out[j].PID
	})
	return out
}

type winProcRow struct {
	PID     int    `json:"pid"`
	Path    string `json:"path"`
	Command string `json:"command"`
}

func collectBinaryProcessesWindows(candidates []string) ([]BinaryProcess, string) {
	if len(candidates) == 0 {
		return nil, ""
	}
	quoted := make([]string, 0, len(candidates))
	for _, c := range candidates {
		abs, _ := filepath.Abs(c)
		quoted = append(quoted, "'"+strings.ReplaceAll(abs, "'", "''")+"'")
	}
	script := "$paths=@(" + strings.Join(quoted, ",") + "); " +
		"Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | " +
		"Where-Object { $paths -contains $_.ExecutablePath } | " +
		"ForEach-Object { [pscustomobject]@{ pid=$_.ProcessId; path=$_.ExecutablePath; command=$_.CommandLine } } | " +
		"ConvertTo-Json -Compress"
	out, err := runBinaryDoctorTool(30*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	if err != "" {
		return nil, err
	}
	rows := parseBinaryRows[winProcRow](out)
	procs := make([]BinaryProcess, 0, len(rows))
	for _, r := range rows {
		procs = append(procs, BinaryProcess{PID: r.PID, Path: r.Path, Command: r.Command})
	}
	return procs, ""
}

func collectBinaryProcessesLinux(candidates []string) ([]BinaryProcess, string) {
	wanted := map[string]bool{}
	for _, c := range candidates {
		abs, err := filepath.Abs(c)
		if err != nil {
			continue
		}
		if real, err := filepath.EvalSymlinks(abs); err == nil {
			abs = real
		}
		wanted[cleanPathKey(abs)] = true
	}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, "read /proc: " + err.Error()
	}
	var out []BinaryProcess
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		exePath, err := os.Readlink(filepath.Join("/proc", e.Name(), "exe"))
		if err != nil {
			continue
		}
		cleanExe := strings.TrimSuffix(exePath, " (deleted)")
		if real, err := filepath.EvalSymlinks(cleanExe); err == nil {
			cleanExe = real
		}
		if !wanted[cleanPathKey(cleanExe)] {
			continue
		}
		out = append(out, BinaryProcess{PID: pid, Path: cleanExe, Command: readProcCmdline(pid)})
	}
	return out, ""
}

func readProcCmdline(pid int) string {
	b, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "cmdline"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(strings.ReplaceAll(string(b), "\x00", " "))
}

func runBinaryDoctorTool(timeout time.Duration, name string, args ...string) (string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return "", err.Error()
	}
	return string(out), ""
}

func parseBinaryRows[T any](text string) []T {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	var rows []T
	if json.Unmarshal([]byte(text), &rows) == nil {
		return rows
	}
	var one T
	if json.Unmarshal([]byte(text), &one) == nil {
		return []T{one}
	}
	return nil
}

func hashFile(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%x", sum[:])
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

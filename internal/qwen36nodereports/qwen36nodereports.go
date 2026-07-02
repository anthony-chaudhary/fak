package qwen36nodereports

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf16"
)

const Schema = "fak.qwen36-node-reports.v1"

type Receive struct {
	Ran      bool     `json:"ran"`
	Reason   string   `json:"reason,omitempty"`
	Command  []string `json:"command,omitempty"`
	ExitCode int      `json:"exit_code,omitempty"`
	Stdout   string   `json:"stdout,omitempty"`
	Stderr   string   `json:"stderr,omitempty"`
	Inbox    string   `json:"inbox"`
}

type ImportArgs struct {
	Inbox        string
	OutDir       string
	Archive      string
	Wait         bool
	SkipTaildrop bool
	Replace      bool
	LogTailLines int
}

func FindTailscale() string {
	for _, name := range []string{"tailscale", "tailscale.exe"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	candidate := `C:\Program Files\Tailscale\tailscale.exe`
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return ""
}

func ReceiveTaildrop(inbox string, wait bool) Receive {
	_ = os.MkdirAll(inbox, 0o755)
	exe := FindTailscale()
	if exe == "" {
		return Receive{Ran: false, Reason: "tailscale CLI not found", Inbox: inbox}
	}
	cmd := []string{exe, "file", "get", "--conflict=rename"}
	if wait {
		cmd = append(cmd, "--wait")
	}
	cmd = append(cmd, inbox)
	proc := exec.Command(cmd[0], cmd[1:]...)
	out, err := proc.Output()
	stderr := ""
	code := 0
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = string(ee.Stderr)
			code = ee.ExitCode()
		} else {
			stderr = err.Error()
			code = 127
		}
	}
	return Receive{Ran: true, Command: cmd, ExitCode: code, Stdout: tail(string(out), 2000), Stderr: tail(stderr, 2000), Inbox: inbox}
}

func ArchiveCandidates(inbox string) []string {
	var paths []string
	for _, pattern := range []string{"qwen36-node-reports-*.zip", "*qwen36-node-reports-*.zip"} {
		hits, _ := filepath.Glob(filepath.Join(inbox, pattern))
		paths = append(paths, hits...)
	}
	uniq := map[string]bool{}
	var out []string
	for _, p := range paths {
		if !uniq[p] {
			uniq[p] = true
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		ai, _ := os.Stat(out[i])
		aj, _ := os.Stat(out[j])
		if ai == nil || aj == nil {
			return out[i] < out[j]
		}
		return ai.ModTime().After(aj.ModTime())
	})
	return out
}

func SafeMemberName(name string) ([]string, error) {
	normalized := strings.ReplaceAll(name, "\\", "/")
	if strings.HasPrefix(normalized, "/") || strings.TrimSpace(normalized) == "" {
		return nil, fmt.Errorf("unsafe zip member path: %s", name)
	}
	parts := strings.Split(normalized, "/")
	var clean []string
	for _, part := range parts {
		if part == "" || part == "." {
			continue
		}
		if part == ".." || strings.Contains(part, ":") {
			return nil, fmt.Errorf("unsafe zip member path: %s", name)
		}
		clean = append(clean, part)
	}
	if len(clean) == 0 {
		return nil, fmt.Errorf("unsafe zip member path: %s", name)
	}
	return clean, nil
}

func ValidateZip(archive string) ([]*zip.File, error) {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	return validateZipFiles(zr.File, archive)
}

func ExtractArchive(archive, outDir string, replace bool) (string, error) {
	abs, _ := filepath.Abs(archive)
	dest := filepath.Join(outDir, strings.TrimSuffix(filepath.Base(archive), filepath.Ext(archive)))
	if _, err := os.Stat(dest); err == nil && !replace {
		return "", fmt.Errorf("%s already exists; rerun with --replace", dest)
	}
	if err := os.RemoveAll(dest); err != nil {
		return "", err
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return "", err
	}
	zr, err := zip.OpenReader(abs)
	if err != nil {
		return "", err
	}
	defer zr.Close()
	members, err := validateZipFiles(zr.File, abs)
	if err != nil {
		return "", err
	}
	destAbs, _ := filepath.Abs(dest)
	for _, info := range members {
		parts, err := SafeMemberName(info.Name)
		if err != nil {
			return "", err
		}
		target := filepath.Join(append([]string{dest}, parts...)...)
		targetAbs, _ := filepath.Abs(target)
		if targetAbs != destAbs && !strings.HasPrefix(targetAbs, destAbs+string(os.PathSeparator)) {
			return "", fmt.Errorf("zip member escapes destination: %s", info.Name)
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return "", err
		}
		src, err := info.Open()
		if err != nil {
			return "", err
		}
		dst, err := os.Create(target)
		if err != nil {
			_ = src.Close()
			return "", err
		}
		_, copyErr := io.Copy(dst, src)
		closeErr := errors.Join(src.Close(), dst.Close())
		if copyErr != nil {
			return "", copyErr
		}
		if closeErr != nil {
			return "", closeErr
		}
	}
	return dest, nil
}

func validateZipFiles(files []*zip.File, archive string) ([]*zip.File, error) {
	var members []*zip.File
	for _, f := range files {
		if _, err := SafeMemberName(f.Name); err != nil {
			return nil, err
		}
		if f.FileInfo().IsDir() {
			continue
		}
		members = append(members, f)
	}
	if len(members) == 0 {
		return nil, fmt.Errorf("archive contains no files: %s", archive)
	}
	return members, nil
}

func ReadTextAuto(path string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if bytes.HasPrefix(raw, []byte{0xff, 0xfe}) {
		return decodeUTF16(raw[2:], binary.LittleEndian), nil
	}
	if bytes.HasPrefix(raw, []byte{0xfe, 0xff}) {
		return decodeUTF16(raw[2:], binary.BigEndian), nil
	}
	if bytes.HasPrefix(raw, []byte{0xef, 0xbb, 0xbf}) {
		raw = raw[3:]
	}
	if json.Valid(raw) || utf8ish(raw) {
		return string(raw), nil
	}
	return decodeUTF16(raw, binary.LittleEndian), nil
}

func TailText(path string, maxLines int) string {
	text, err := ReadTextAuto(path)
	if err != nil {
		return fmt.Sprintf("<failed to read %s: %v>", path, err)
	}
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if maxLines > 0 && len(lines) > maxLines {
		lines = lines[len(lines)-maxLines:]
	}
	return strings.Join(lines, "\n")
}

func LoadPreflight(path string) map[string]any {
	text, err := ReadTextAuto(path)
	if err != nil {
		return map[string]any{"path": path, "parsed": false, "error": err.Error()}
	}
	var data map[string]any
	if err := json.Unmarshal([]byte(text), &data); err != nil {
		return map[string]any{"path": path, "parsed": false, "error": err.Error()}
	}
	checks, _ := data["checks"].([]any)
	var failed []string
	var nvidia any
	for _, item := range checks {
		check, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if check["ok"] == false {
			if name, ok := check["name"].(string); ok {
				failed = append(failed, name)
			}
		}
		if check["name"] == "nvidia_smi" {
			nvidia = check
		}
	}
	failures, ok := data["failures"].([]any)
	if !ok {
		failures = []any{}
	}
	return map[string]any{
		"path":               path,
		"parsed":             true,
		"ok":                 data["ok"],
		"profile":            data["profile"],
		"base_url":           data["base_url"],
		"llama_server_found": data["llama_server_found"],
		"failures":           failures,
		"failed_checks":      failed,
		"nvidia_smi":         nvidia,
	}
}

func SummarizeDir(reportDir string, logTailLines int) map[string]any {
	preflightPaths, _ := filepath.Glob(filepath.Join(reportDir, "**", "preflight-*.json"))
	serverLogs, _ := filepath.Glob(filepath.Join(reportDir, "**", "server-*.log"))
	if len(preflightPaths) == 0 {
		preflightPaths = walkGlob(reportDir, "preflight-*.json")
	}
	if len(serverLogs) == 0 {
		serverLogs = walkGlob(reportDir, "server-*.log")
	}
	sortByMod(preflightPaths)
	sortByMod(serverLogs)
	preflights := make([]map[string]any, 0, len(preflightPaths))
	for _, p := range preflightPaths {
		preflights = append(preflights, LoadPreflight(p))
	}
	var latestPreflight any
	if len(preflights) > 0 {
		latestPreflight = preflights[len(preflights)-1]
	}
	latestLog := ""
	if len(serverLogs) > 0 {
		latestLog = serverLogs[len(serverLogs)-1]
	}
	summary := map[string]any{
		"schema":                 Schema,
		"report_dir":             reportDir,
		"preflight_count":        len(preflightPaths),
		"server_log_count":       len(serverLogs),
		"latest_preflight":       latestPreflight,
		"latest_server_log":      latestLog,
		"latest_server_log_tail": "",
	}
	if latestLog != "" {
		summary["latest_server_log_tail"] = TailText(latestLog, logTailLines)
	}
	if latestPreflight == nil {
		summary["status"] = "NO_PREFLIGHT"
	} else if lp, ok := latestPreflight.(map[string]any); ok && lp["ok"] == true {
		summary["status"] = "PREFLIGHT_OK"
	} else {
		summary["status"] = "PREFLIGHT_FAILED"
	}
	return summary
}

func ImportReportBundle(args ImportArgs) map[string]any {
	var receive any
	if !args.SkipTaildrop {
		receive = ReceiveTaildrop(args.Inbox, args.Wait)
	}
	archive := args.Archive
	if archive == "" {
		candidates := ArchiveCandidates(args.Inbox)
		if len(candidates) == 0 {
			return map[string]any{"schema": Schema, "imported": false, "receive": receive, "error": fmt.Sprintf("no qwen36-node-reports-*.zip bundle found under %s", args.Inbox)}
		}
		archive = candidates[0]
	}
	dest, err := ExtractArchive(archive, args.OutDir, args.Replace)
	if err != nil {
		return map[string]any{"schema": Schema, "imported": false, "receive": receive, "error": err.Error()}
	}
	summary := SummarizeDir(dest, args.LogTailLines)
	summary["imported"] = true
	summary["archive"] = archive
	summary["receive"] = receive
	return summary
}

func MarshalJSON(v any) ([]byte, error) {
	return json.MarshalIndent(v, "", "  ")
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

func decodeUTF16(raw []byte, order binary.ByteOrder) string {
	u16 := make([]uint16, 0, len(raw)/2)
	for len(raw) >= 2 {
		u16 = append(u16, order.Uint16(raw[:2]))
		raw = raw[2:]
	}
	return string(utf16.Decode(u16))
}

func utf8ish(raw []byte) bool {
	for _, b := range raw {
		if b == 0 {
			return false
		}
	}
	return true
}

func walkGlob(root, pattern string) []string {
	var out []string
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if ok, _ := filepath.Match(pattern, filepath.Base(path)); ok {
				out = append(out, path)
			}
		}
		return nil
	})
	return out
}

func sortByMod(paths []string) {
	sort.Slice(paths, func(i, j int) bool {
		ai, _ := os.Stat(paths[i])
		aj, _ := os.Stat(paths[j])
		if ai == nil || aj == nil {
			return paths[i] < paths[j]
		}
		return ai.ModTime().Before(aj.ModTime())
	})
}

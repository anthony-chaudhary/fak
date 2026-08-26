package fleetpane

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

func AppVersion(root string) string {
	raw, err := os.ReadFile(filepath.Join(root, "VERSION"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(raw))
}

func MachineID(cfg Config) string {
	raw := cfg.MachineID
	if raw == "" {
		raw = hostName()
	}
	var b strings.Builder
	for _, ch := range raw {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.' {
			b.WriteRune(ch)
		} else {
			b.WriteByte('-')
		}
	}
	slug := strings.ToLower(strings.Trim(b.String(), "-."))
	if slug == "" {
		return "machine"
	}
	return slug
}

func ExpandCmd(parts []string, cfg Config) []string {
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		out = append(out, expandValue(part, cfg))
	}
	return out
}

func expandValue(value string, cfg Config) string {
	repl := strings.NewReplacer(
		"{repo}", cfg.Root,
		"{root}", cfg.Root,
		"{job_dir}", cfg.JobDir,
		"{python}", cfg.Python,
		"{target}", strconv.Itoa(cfg.Target),
		"{claude_exe}", cfg.ClaudeExe,
		"{user_home}", cfg.UserHome,
		"{registry_dir}", cfg.RegistryDir,
		"{machine_id}", MachineID(cfg),
	)
	return repl.Replace(value)
}

func LoopCheckNeedsAction(check LoopCheck) bool {
	return check.State != "OK" && check.State != "SKIPPED"
}

func CompactCounts(counts map[string]any) string {
	keys := make([]string, 0, len(counts))
	for key, val := range counts {
		if key != "" && intValueDefault(val, 0) != 0 {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		return "(none)"
	}
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", key, intValueDefault(counts[key], 0)))
	}
	return strings.Join(parts, " ")
}

func statusDocToMap(status StatusDoc) map[string]any {
	raw, _ := json.Marshal(status)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	return out
}

func loopCheckSnapshot(check LoopCheck) map[string]any {
	out := map[string]any{
		"name":            check.Name,
		"state":           stringValueDefault(check.State, "UNKNOWN"),
		"detail":          strmatch.FirstNonEmpty(check.Detail, check.Reason),
		"action":          check.Action,
		"auto_recover":    check.AutoRecover,
		"has_recover_cmd": check.HasRecoverCmd,
	}
	if check.ReturnCode != nil {
		out["returncode"] = *check.ReturnCode
	}
	return out
}

func loopCheckSnapshotMap(check map[string]any) map[string]any {
	return map[string]any{
		"name":            check["name"],
		"state":           stringValueDefault(check["state"], "UNKNOWN"),
		"detail":          strmatch.FirstNonEmpty(stringValue(check["detail"]), stringValue(check["reason"])),
		"action":          stringValue(check["action"]),
		"auto_recover":    boolValue(check["auto_recover"]),
		"has_recover_cmd": boolValue(check["has_recover_cmd"]),
		"returncode":      check["returncode"],
	}
}

func machineVersionDrift(machine map[string]any, currentVersion string) map[string]any {
	if currentVersion == "" {
		return nil
	}
	machineVersion := stringValueDefault(machine["app_version"], "unknown")
	if machineVersion == currentVersion {
		return nil
	}
	gitStatus := mapValue(machine["git"])
	return map[string]any{
		"machine_version": machineVersion,
		"current_version": currentVersion,
		"git_state":       stringValueDefault(gitStatus["safe_ff_state"], "unknown"),
		"command":         "python tools/fleet_control_pane.py sync --fetch --json",
		"detail":          "pane version differs from this operator pane",
	}
}

func parseISOAgeMin(raw string, opts Options) *float64 {
	if raw == "" {
		return nil
	}
	stamp, err := time.Parse(time.RFC3339, strings.Replace(raw, "Z", "+00:00", 1))
	if err != nil {
		return nil
	}
	age := opts.now().Sub(stamp.UTC()).Minutes()
	age = float64(int(age*10)) / 10
	return &age
}

func blockedAccountSnapshot(account map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range account {
		out[k] = v
	}
	return out
}

func accountBlockIsAuthLike(account map[string]any) bool {
	kind := strings.ToLower(stringValue(account["block_kind"]))
	reason := strings.ToLower(stringValue(account["reason"]))
	return strings.Contains(kind, "auth") || strings.Contains(reason, "auth") || strings.Contains(reason, "login")
}

func accountBlockIsUsageLike(account map[string]any) bool {
	kind := strings.ToLower(stringValue(account["block_kind"]))
	reason := strings.ToLower(stringValue(account["reason"]))
	return strings.Contains(kind, "usage") || strings.Contains(reason, "limit") || strings.Contains(reason, "quota")
}

func loopNames(loops map[string]LoopSpec) []string {
	names := make([]string, 0, len(loops))
	for name := range loops {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func loopCwd(spec LoopSpec, cfg Config) string {
	if spec.Cwd == "" {
		return cfg.Root
	}
	return resolvePathString(expandValue(spec.Cwd, cfg), cfg.Root)
}

func expectedReturnCodes(spec LoopSpec) map[int]bool {
	out := map[int]bool{}
	for _, code := range spec.OKReturnCodes {
		out[code] = true
	}
	if len(out) == 0 {
		out[0] = true
	}
	return out
}

func envForTools(cfg Config) map[string]string {
	env := map[string]string{"PYTHONIOENCODING": "utf-8"}
	if cfg.UserHome != "" {
		env["FLEET_USER_HOME"] = cfg.UserHome
	}
	if cfg.RegistryDir != "" {
		env["FLEET_REG_DIR"] = cfg.RegistryDir
	}
	return env
}

func loopScaffoldTemplateCommand(cfg Config) string {
	return cfg.Python + " tools/fleet_control_pane.py loop-scaffold NAME --apply"
}

func loopAddTemplateCommand(cfg Config) string {
	return cfg.Python + " tools/fleet_control_pane.py loop-add NAME --status-cmd '[\"cmd\", \"--json\"]' --apply"
}

func displayCommand(cmd []string) string {
	return strings.Join(cmd, " ")
}

func resolvePathString(value, root string) string {
	value = strings.TrimSpace(os.ExpandEnv(value))
	if value == "" {
		return ""
	}
	if strings.HasPrefix(value, "~") {
		if home := userHome(); home != "" {
			if value == "~" {
				value = home
			} else if strings.HasPrefix(value, "~/") || strings.HasPrefix(value, `~\`) {
				value = filepath.Join(home, value[2:])
			}
		}
	}
	if !filepath.IsAbs(value) {
		value = filepath.Join(root, value)
	}
	if abs, err := filepath.Abs(value); err == nil {
		return abs
	}
	return value
}

func displayPath(path, root string) string {
	if path == "" {
		return ""
	}
	rel, err := filepath.Rel(root, path)
	if err == nil && !strings.HasPrefix(rel, "..") && !filepath.IsAbs(rel) {
		return filepath.ToSlash(rel)
	}
	return path
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func hostName() string {
	host, err := os.Hostname()
	if err != nil || host == "" {
		addrs, _ := net.InterfaceAddrs()
		if len(addrs) > 0 {
			return "machine"
		}
		return "machine"
	}
	return host
}

func okState(ok bool) string {
	if ok {
		return "OK"
	}
	return "ACTION"
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

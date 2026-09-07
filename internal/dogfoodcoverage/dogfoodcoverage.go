package dogfoodcoverage

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const (
	// Schema is the JSON schema version identifier for dogfood-coverage payloads.
	Schema = "dogfood-coverage/1"
)

// KPI represents a single evaluated key performance indicator.
type KPI struct {
	Key      string `json:"key"`
	OK       bool   `json:"ok"`
	Hard     bool   `json:"hard"`
	Detail   string `json:"detail"`
	Evidence string `json:"evidence"`
}

// Report captures the full dogfood-coverage scorecard payload.
type Report struct {
	Schema      string   `json:"schema"`
	Coverage    float64  `json:"coverage"`
	Met         int      `json:"met"`
	Total       int      `json:"total"`
	DogfoodDebt int      `json:"dogfood_debt"`
	Grade       string   `json:"grade"`
	AuditRows   int      `json:"audit_rows"`
	KPIs        []KPI    `json:"kpis"`
	WorstFirst  []string `json:"worst_first"`
}

var guardOffValues = map[string]struct{}{
	"0":       {},
	"off":     {},
	"false":   {},
	"no":      {},
	"disable": {},
}

func envLookup(env map[string]string, key string) (string, bool) {
	if env != nil {
		val, ok := env[key]
		return val, ok
	}
	return os.LookupEnv(key)
}

func envGet(env map[string]string, key string) string {
	val, _ := envLookup(env, key)
	return val
}

func isFile(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}

func fileContains(path, needle string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return strings.Contains(string(data), needle)
}

func findRepoRoot(start string) string {
	if start == "" {
		if wd, err := os.Getwd(); err == nil {
			start = wd
		}
	}
	abs, err := filepath.Abs(start)
	if err == nil {
		start = abs
	}
	dir := start
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return start
		}
		dir = parent
	}
}

func whichOnPath(name, pathVal string) string {
	if pathVal == "" {
		pathVal = os.Getenv("PATH")
	}
	suffixes := []string{""}
	if runtime.GOOS == "windows" && filepath.Ext(name) == "" {
		pathext := os.Getenv("PATHEXT")
		if pathext == "" {
			pathext = ".COM;.EXE;.BAT;.CMD"
		}
		for _, ext := range strings.Split(pathext, string(os.PathListSeparator)) {
			ext = strings.TrimSpace(ext)
			if ext != "" {
				suffixes = append(suffixes, strings.ToLower(ext), strings.ToUpper(ext))
			}
		}
	}

	dirs := strings.Split(pathVal, string(os.PathListSeparator))
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		for _, suffix := range suffixes {
			candidate := filepath.Join(dir, name+suffix)
			if fi, err := os.Stat(candidate); err == nil && !fi.IsDir() {
				return candidate
			}
		}
	}
	return ""
}

func resolveFakBin(root string, env map[string]string) string {
	explicit := strings.TrimSpace(envGet(env, "FAK_BIN"))
	if explicit != "" {
		if isFile(explicit) {
			return explicit
		}
	}

	var candidates []string
	pathVal := envGet(env, "PATH")
	onPath := whichOnPath("fak", pathVal)
	if onPath != "" {
		candidates = append(candidates, onPath)
	}

	exe := "fak"
	if runtime.GOOS == "windows" {
		exe = "fak.exe"
	}

	inTree := filepath.Join(root, "tools", ".bin", exe)
	if isFile(inTree) {
		candidates = append(candidates, inTree)
	}

	rootBin := filepath.Join(root, exe)
	if isFile(rootBin) {
		candidates = append(candidates, rootBin)
	}

	if len(candidates) == 0 {
		return ""
	}

	var best string
	var bestMtime time.Time
	for _, c := range candidates {
		fi, err := os.Stat(c)
		if err != nil {
			continue
		}
		if best == "" || fi.ModTime().After(bestMtime) {
			best = c
			bestMtime = fi.ModTime()
		}
	}
	return best
}

func guardEnabled(env map[string]string) bool {
	raw, ok := envLookup(env, "FLEET_DOGFOOD_GUARD")
	if !ok {
		return true
	}
	trimmed := strings.ToLower(strings.TrimSpace(raw))
	_, isOff := guardOffValues[trimmed]
	return !isOff
}

const pyDispatchRunner = `import importlib.util, os, sys, json
root = sys.argv[1]
p = os.path.join(root, 'tools', 'dispatch_worker.py')
try:
    spec = importlib.util.spec_from_file_location('dispatch_worker', p)
    if not (spec and spec.loader):
        raise ImportError(f"cannot load dispatch_worker from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(p))
    spec.loader.exec_module(mod)
    raw = mod.build_command('probe', 'claude')
    e = json.loads(sys.argv[2])
    launch_cmd, guarded = mod.guarded_launch_command(raw, 'probe', 'claude', root, env=e)
    fak_bin = mod.resolve_fak_bin(root, e)
    live_on = mod.guard_enabled(e)
    print(json.dumps({'ok': True, 'guarded': bool(guarded), 'launch_cmd': list(launch_cmd), 'fak_bin': fak_bin or '', 'live_on': bool(live_on)}))
except (ImportError, AttributeError, OSError) as exc:
    print(json.dumps({'ok': False, 'moved': True, 'error': str(exc)}))
except Exception as exc:
    print(json.dumps({'ok': False, 'moved': False, 'error': str(exc)}))
`

func evalDispatchWorker(root string, env map[string]string) (guarded bool, launchCmd []string, fakBin string, liveOn bool, movedErr string) {
	dwPath := filepath.Join(root, "tools", "dispatch_worker.py")
	if _, err := os.Stat(dwPath); os.IsNotExist(err) {
		return false, nil, "", false, fmt.Sprintf("dispatch surface moved: cannot load dispatch_worker from %s", dwPath)
	}

	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}

	if err == nil {
		envBytes, _ := json.Marshal(env)
		cmd := exec.Command(python, "-c", pyDispatchRunner, root, string(envBytes))
		out, err := cmd.CombinedOutput()
		if err == nil {
			var resp struct {
				OK        bool     `json:"ok"`
				Moved     bool     `json:"moved"`
				Error     string   `json:"error"`
				Guarded   bool     `json:"guarded"`
				LaunchCmd []string `json:"launch_cmd"`
				FakBin    string   `json:"fak_bin"`
				LiveOn    bool     `json:"live_on"`
			}
			if jsonErr := json.Unmarshal(out, &resp); jsonErr == nil {
				if resp.OK {
					return resp.Guarded, resp.LaunchCmd, resp.FakBin, resp.LiveOn, ""
				}
				return false, nil, "", false, fmt.Sprintf("dispatch surface moved: %s", resp.Error)
			}
		}
	}

	contentBytes, err := os.ReadFile(dwPath)
	if err != nil {
		return false, nil, "", false, fmt.Sprintf("dispatch surface moved: %v", err)
	}
	content := string(contentBytes)
	for _, req := range []string{"guarded_launch_command", "build_command", "resolve_fak_bin", "guard_enabled"} {
		if !strings.Contains(content, req) {
			return false, nil, "", false, fmt.Sprintf("dispatch surface moved: module 'dispatch_worker' has no attribute '%s'", req)
		}
	}

	liveOn = guardEnabled(env)
	fakBin = resolveFakBin(root, env)
	guarded = liveOn && fakBin != ""
	if guarded {
		launchCmd = []string{fakBin, "guard", "--provider"}
	}
	return guarded, launchCmd, fakBin, liveOn, ""
}

// CountAuditRows counts decision rows the kernel recorded across the fleet's per-lane decision
// journals. Returns (rows, journals).
func CountAuditRows(root string, env map[string]string) (rows int, journals int) {
	var candidates []string
	fleetDir := filepath.Join(root, ".dispatch-runs", "guard-audit")
	if fi, err := os.Stat(fleetDir); err == nil && fi.IsDir() {
		entries, err := os.ReadDir(fleetDir)
		if err == nil {
			var jsonls []string
			for _, e := range entries {
				if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
					jsonls = append(jsonls, filepath.Join(fleetDir, e.Name()))
				}
			}
			sort.Strings(jsonls)
			candidates = append(candidates, jsonls...)
		}
	}

	if _, err := os.Stat(filepath.Join(root, ".git")); err == nil {
		cfg := envGet(env, "XDG_CONFIG_HOME")
		if cfg == "" {
			cfg = envGet(env, "APPDATA")
		}
		if cfg == "" {
			home := envGet(env, "HOME")
			if home == "" {
				if h, err := os.UserHomeDir(); err == nil {
					home = h
				}
			}
			if home != "" {
				cfg = filepath.Join(home, ".config")
			}
		}
		if cfg != "" {
			userJournal := filepath.Join(cfg, "fak", "guard-audit.jsonl")
			if isFile(userJournal) {
				candidates = append(candidates, userJournal)
			}
		}
	}

	for _, jp := range candidates {
		text, err := os.ReadFile(jp)
		if err != nil {
			continue
		}
		n := 0
		lines := strings.Split(string(text), "\n")
		for _, ln := range lines {
			if strings.TrimSpace(ln) != "" {
				n++
			}
		}
		if n > 0 {
			rows += n
			journals++
		}
	}
	return rows, journals
}

// DiagnoseAuditGap explains WHY the audit journal is empty. Returns "" when rows exist.
func DiagnoseAuditGap(root string) string {
	fleetDir := filepath.Join(root, ".dispatch-runs", "guard-audit")
	fi, err := os.Stat(fleetDir)
	if err != nil || !fi.IsDir() {
		return "no guard-audit journal directory yet — no guarded worker has run on this host (arm `fak guard -- <agent>` so the kernel records verdicts)"
	}
	entries, err := os.ReadDir(fleetDir)
	if err != nil {
		return "no guard-audit journal directory yet — no guarded worker has run on this host (arm `fak guard -- <agent>` so the kernel records verdicts)"
	}
	var jsonls []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
			jsonls = append(jsonls, e.Name())
		}
	}
	sort.Strings(jsonls)
	if len(jsonls) == 0 {
		return "guard-audit directory exists but holds no journal files — the guard wire is configured but never exercised by a launched worker"
	}

	hasRows := false
	for _, j := range jsonls {
		text, err := os.ReadFile(filepath.Join(fleetDir, j))
		if err != nil {
			continue
		}
		for _, ln := range strings.Split(string(text), "\n") {
			if strings.TrimSpace(ln) != "" {
				hasRows = true
				break
			}
		}
		if hasRows {
			break
		}
	}
	if hasRows {
		return ""
	}

	return fmt.Sprintf("%d journal file(s) present but all blank — a guarded worker booted but proposed no adjudicated tool call (check the agent reached a tool use; an auth/login failure exits before the first verdict)", len(jsonls))
}

// Grade maps coverage percent and debt count onto an A-F letter grade.
func Grade(coverage float64, dogfoodDebt int) string {
	if dogfoodDebt == 0 && coverage >= 95.0 {
		return "A"
	}
	if dogfoodDebt == 0 {
		return "B"
	}
	if dogfoodDebt <= 1 {
		return "C"
	}
	if dogfoodDebt <= 2 {
		return "D"
	}
	return "F"
}

// FormatReport formats a Report into a human-readable string.
func FormatReport(r Report) string {
	lines := []string{
		fmt.Sprintf("dogfood-coverage: %.1f%% (%d/%d KPIs)  grade %s  dogfood_debt %d  audit_rows %d",
			r.Coverage, r.Met, r.Total, r.Grade, r.DogfoodDebt, r.AuditRows),
	}
	for _, k := range r.KPIs {
		mark := ".. "
		if k.OK {
			mark = "OK "
		} else if k.Hard {
			mark = "XX "
		}
		tag := "soft"
		if k.Hard {
			tag = "HARD"
		}
		lines = append(lines, fmt.Sprintf("  [%s] %-24s (%s)  %s", mark, k.Key, tag, k.Detail))
		if k.Evidence != "" {
			lines = append(lines, fmt.Sprintf("          -> %s", k.Evidence))
		}
	}
	if len(r.WorstFirst) > 0 {
		lines = append(lines, "  next: "+strings.Join(r.WorstFirst, ", "))
	}
	return strings.Join(lines, "\n")
}

// Evaluate computes the dogfood-coverage KPIs for workspace root under env.
func Evaluate(root string, env map[string]string) Report {
	if root == "" {
		root = findRepoRoot("")
	}

	var e map[string]string
	if env != nil {
		e = env
	} else {
		e = make(map[string]string)
		for _, kv := range os.Environ() {
			idx := strings.IndexByte(kv, '=')
			if idx >= 0 {
				e[kv[:idx]] = kv[idx+1:]
			}
		}
	}

	var kpis []KPI
	add := func(key string, ok, hard bool, detail, evidence string) {
		kpis = append(kpis, KPI{
			Key:      key,
			OK:       ok,
			Hard:     hard,
			Detail:   detail,
			Evidence: evidence,
		})
	}

	guarded, launchCmd, fakBin, liveOn, movedErr := evalDispatchWorker(root, e)
	if movedErr != "" {
		add("fleet_leaf_guarded", false, true,
			"dispatch_worker.guarded_launch_command fronts a claude worker with `fak guard`",
			movedErr)
		add("fak_bin_resolvable", false, true,
			"a `fak` binary resolves (FAK_BIN / tools/.bin / PATH) so guard-wrapping engages",
			movedErr)
		add("guard_default_on", false, true,
			"FLEET_DOGFOOD_GUARD is not disabled in the live environment (default ON)",
			movedErr)
	} else {
		leafEvidence := "UNWRAPPED (coverage=0 on this host)"
		if guarded {
			first := launchCmd
			if len(first) > 3 {
				first = first[:3]
			}
			leafEvidence = strings.Join(first, " ")
		}
		add("fleet_leaf_guarded", guarded, true,
			"dispatch_worker.guarded_launch_command fronts a claude worker with `fak guard`",
			leafEvidence)

		fakBinEvidence := fakBin
		if fakBinEvidence == "" {
			fakBinEvidence = "no fak binary found — workers run UNGUARDED"
		}
		add("fak_bin_resolvable", fakBin != "", true,
			"a `fak` binary resolves (FAK_BIN / tools/.bin / PATH) so guard-wrapping engages",
			fakBinEvidence)

		guardVal, hasGuardVal := e["FLEET_DOGFOOD_GUARD"]
		if !hasGuardVal {
			guardVal = "<unset=ON>"
		}
		add("guard_default_on", liveOn, true,
			"FLEET_DOGFOOD_GUARD is not disabled in the live environment (default ON)",
			"FLEET_DOGFOOD_GUARD="+guardVal)
	}

	wired := fileContains(filepath.Join(root, "tools", "issue_dispatch.py"), "guarded_launch_command")
	wiredEvidence := "MISSING"
	if wired {
		wiredEvidence = "tools/issue_dispatch.py calls guarded_launch_command"
	}
	add("issue_dispatch_wired", wired, true,
		"issue_dispatch.evaluate routes its detached spawn through guarded_launch_command",
		wiredEvidence)

	guardGo := isFile(filepath.Join(root, "cmd", "fak", "guard.go"))
	guardGoEvidence := "MISSING"
	if guardGo {
		guardGoEvidence = "cmd/fak/guard.go"
	}
	add("guard_verb_present", guardGo, true,
		"`fak guard -- claude` exists (cmd/fak/guard.go) as the one-command kernel front door",
		guardGoEvidence)

	documented := fileContains(filepath.Join(root, "DOGFOOD-CLAUDE.md"), "fak guard")
	add("guard_documented", documented, false,
		"DOGFOOD-CLAUDE.md documents the `fak guard` front door",
		"DOGFOOD-CLAUDE.md")

	rows, journals := CountAuditRows(root, e)
	auditEvidence := fmt.Sprintf("%d decision row(s) across %d journal(s)", rows, journals)
	if rows == 0 {
		auditEvidence += " — " + DiagnoseAuditGap(root)
	}
	add("audit_journal_evidence", rows > 0, false,
		"guarded workers have recorded kernel decisions in a durable audit journal",
		auditEvidence)

	doc := isFile(filepath.Join(root, "docs", "fak", "always-on-dogfood-server.md"))
	docEvidence := "MISSING"
	if doc {
		docEvidence = "docs/fak/always-on-dogfood-server.md"
	}
	add("always_on_server_doc", doc, false,
		"an always-on dogfood server design exists (Mac/GCP tiers)",
		docEvidence)

	plist := isFile(filepath.Join(root, "tools", "com.fak.serve-gateway.plist"))
	plistEvidence := "MISSING"
	if plist {
		plistEvidence = "tools/com.fak.serve-gateway.plist"
	}
	add("always_on_serve_plist", plist, false,
		"a launchd unit keeps a shared `fak serve` gateway alive 24/7",
		plistEvidence)

	total := len(kpis)
	met := 0
	var hardUnmet []KPI
	for _, k := range kpis {
		if k.OK {
			met++
		} else if k.Hard {
			hardUnmet = append(hardUnmet, k)
		}
	}
	coverage := 0.0
	if total > 0 {
		coverage = math.Round(1000.0*float64(met)/float64(total)) / 10.0
	}
	dogfoodDebt := len(hardUnmet)
	grade := Grade(coverage, dogfoodDebt)

	worstFirst := make([]string, 0, len(kpis))
	for _, k := range hardUnmet {
		worstFirst = append(worstFirst, k.Key)
	}
	for _, k := range kpis {
		if !k.Hard && !k.OK {
			worstFirst = append(worstFirst, k.Key)
		}
	}

	return Report{
		Schema:      Schema,
		Coverage:    coverage,
		Met:         met,
		Total:       total,
		DogfoodDebt: dogfoodDebt,
		Grade:       grade,
		AuditRows:   rows,
		KPIs:        kpis,
		WorstFirst:  worstFirst,
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleet"
)

const (
	labTargetsSchema        = "fak.lab_targets/v1"
	labTargetResolveSchema  = "fak.lab_target.resolve/v1"
	labTargetAliasPrefix    = "@lab/"
	labTargetDefaultCommand = "fak guard --remote-serve @lab/glm-5.2 --probe -- codex"
)

type labTargetsFile struct {
	Schema  string            `json:"schema,omitempty"`
	Targets []labTargetConfig `json:"targets"`
}

type labTargetConfig struct {
	Alias   string `json:"alias"`
	BaseURL string `json:"base_url"`
	Model   string `json:"model,omitempty"`
	BoxID   string `json:"box_id,omitempty"`
	Roster  string `json:"roster,omitempty"`
}

type labTargetResolution struct {
	Schema         string `json:"schema"`
	Alias          string `json:"alias"`
	MachineClass   string `json:"machine_class"`
	Status         string `json:"status"`
	Model          string `json:"model,omitempty"`
	BoxID          string `json:"box_id,omitempty"`
	Evidence       string `json:"evidence"`
	NextAction     string `json:"next_action"`
	RemoteServeArg string `json:"remote_serve_arg"`
	GuardCommand   string `json:"guard_command"`
	baseURL        string
}

func labTargetPath(flagVal string) (string, error) {
	if flagVal != "" {
		return flagVal, nil
	}
	if env := os.Getenv("FAK_LAB_TARGETS"); env != "" {
		return env, nil
	}
	cfgDir, err := nodeConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(cfgDir, "fleet", "lab-targets.json"), nil
}

func labTarget(stdout, stderr io.Writer, argv []string) int {
	aliasArg := ""
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		aliasArg, argv = argv[0], argv[1:]
	}
	fs := flag.NewFlagSet("lab target", flag.ContinueOnError)
	fs.SetOutput(stderr)
	targetsPath := fs.String("targets", "", "local lab target alias file (default: $FAK_LAB_TARGETS or ~/.config/fak/fleet/lab-targets.json)")
	readinessPath := fs.String("readiness", "", "lab readiness record path (default: $FAK_LAB_READINESS or ~/.config/fak/fleet/lab-readiness.json)")
	rosterPath := fs.String("roster", "", "roster file used to validate reports (default: the embedded generic lab roster)")
	reports := fs.String("reports", "", "reports dir used to validate fresh useful inference (default: $FAK_FLEET_REPORTS or ~/.config/fak/fleet/reports)")
	machineClass := fs.String("class", "gpu-server", "generic machine class")
	asJSON := fs.Bool("json", false, "emit the scrubbed target resolution as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if aliasArg == "" && fs.NArg() == 1 {
		aliasArg = fs.Arg(0)
	} else if aliasArg != "" && fs.NArg() == 0 {
		// alias was consumed before parsing so `fak lab target @lab/glm-5.2 --json`
		// works as documented.
	} else {
		fmt.Fprintln(stderr, "usage: fak lab target @lab/<model> [--targets F] [--readiness F] [--roster F] [--reports DIR] [--json]")
		return 2
	}
	res, err := resolveLabTarget(aliasArg, labTargetResolveOpts{
		targetsPath:   *targetsPath,
		readinessPath: *readinessPath,
		rosterPath:    *rosterPath,
		reportsDir:    *reports,
		machineClass:  *machineClass,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak lab target: %v\n", err)
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak lab target: encode: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "lab target %s: %s (%s)\n", res.Alias, res.Status, res.NextAction)
	fmt.Fprintf(stdout, "guard: %s\n", res.GuardCommand)
	return 0
}

type labTargetResolveOpts struct {
	targetsPath   string
	readinessPath string
	rosterPath    string
	reportsDir    string
	machineClass  string
}

func resolveGuardRemoteServe(operand string) (string, error) {
	if isLabTargetAlias(operand) {
		res, err := resolveLabTarget(operand, labTargetResolveOpts{machineClass: "gpu-server"})
		if err != nil {
			return "", err
		}
		return res.baseURL, nil
	}
	return normalizeRemoteServe(operand)
}

func resolveLabTarget(alias string, opts labTargetResolveOpts) (labTargetResolution, error) {
	alias = strings.TrimSpace(alias)
	if !isLabTargetAlias(alias) {
		return labTargetResolution{}, fmt.Errorf("LAB_TARGET_BAD_ALIAS: %q must be shaped @lab/<model>", alias)
	}
	if opts.machineClass == "" {
		opts.machineClass = "gpu-server"
	}
	ready, err := loadLabTargetReadiness(opts.readinessPath, opts.machineClass)
	if err != nil {
		return labTargetResolution{}, err
	}
	if !ready.AdmitLabDispatch {
		return labTargetResolution{}, fmt.Errorf("LAB_READINESS_NOT_READY: status=%s next=%s; refresh with `fak lab readiness --from-status --write-default --json`", ready.Status, ready.NextAction)
	}
	targets, err := loadLabTargets(opts.targetsPath)
	if err != nil {
		return labTargetResolution{}, err
	}
	target, ok := findLabTarget(targets, alias)
	if !ok {
		return labTargetResolution{}, fmt.Errorf("LAB_TARGET_NOT_FOUND: %s is not in the local lab target config", alias)
	}
	base, err := normalizeLabTargetBaseURL(target.BaseURL)
	if err != nil {
		return labTargetResolution{}, fmt.Errorf("LAB_TARGET_BAD_BASE_URL: %v", err)
	}
	model := strings.TrimSpace(target.Model)
	if model == "" {
		model = strings.TrimPrefix(alias, labTargetAliasPrefix)
	}
	boxID, evidence, err := validateLabTargetReport(target, model, opts)
	if err != nil {
		return labTargetResolution{}, err
	}
	// When a local target config points at a private roster, the matched report key can
	// be a lab identifier. Keep the public resolution scrubbed unless the config
	// explicitly supplied a display-safe box_id.
	if strings.TrimSpace(target.Roster) != "" && strings.TrimSpace(target.BoxID) == "" {
		boxID = ""
	} else if boxID == "" {
		boxID = strings.TrimSpace(target.BoxID)
	}
	return labTargetResolution{
		Schema:         labTargetResolveSchema,
		Alias:          alias,
		MachineClass:   opts.machineClass,
		Status:         fleet.LabReadyForDevWork,
		Model:          model,
		BoxID:          boxID,
		Evidence:       evidence,
		NextAction:     "use-guard-remote-serve-alias",
		RemoteServeArg: alias,
		GuardCommand:   "fak guard --remote-serve " + alias + " --probe -- codex",
		baseURL:        base,
	}, nil
}

func loadLabTargetReadiness(pathFlag, machineClass string) (fleet.LabReadiness, error) {
	path, err := labReadinessPath(pathFlag)
	if err != nil {
		return fleet.LabReadiness{}, err
	}
	f, err := os.Open(path)
	if err != nil {
		return fleet.LabReadiness{}, fmt.Errorf("LAB_READINESS_NOT_READY: missing %s; refresh with `fak lab readiness --from-status --write-default --json`", path)
	}
	defer f.Close()
	rec, err := fleet.LoadLabReadiness(f)
	if err != nil {
		return fleet.LabReadiness{}, fmt.Errorf("LAB_READINESS_NOT_READY: invalid %s: %v", path, err)
	}
	if rec.MachineClass == "" {
		rec.MachineClass = machineClass
	}
	return rec, nil
}

func loadLabTargets(pathFlag string) (labTargetsFile, error) {
	path, err := labTargetPath(pathFlag)
	if err != nil {
		return labTargetsFile{}, err
	}
	f, err := os.Open(path)
	if err != nil {
		return labTargetsFile{}, fmt.Errorf("LAB_TARGET_CONFIG_MISSING: missing %s; the private tunnel producer must write a local %s file", path, labTargetsSchema)
	}
	defer f.Close()
	var doc labTargetsFile
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		return labTargetsFile{}, fmt.Errorf("LAB_TARGET_CONFIG_INVALID: decode %s: %v", path, err)
	}
	if probs := validateLabTargets(doc); len(probs) > 0 {
		return labTargetsFile{}, fmt.Errorf("LAB_TARGET_CONFIG_INVALID: %s", strings.Join(probs, "; "))
	}
	return doc, nil
}

func validateLabTargets(doc labTargetsFile) []string {
	var probs []string
	if doc.Schema != "" && doc.Schema != labTargetsSchema {
		probs = append(probs, fmt.Sprintf("schema %q is not %q", doc.Schema, labTargetsSchema))
	}
	if len(doc.Targets) == 0 {
		probs = append(probs, "targets is empty")
	}
	seen := map[string]bool{}
	for i, t := range doc.Targets {
		if !isLabTargetAlias(t.Alias) {
			probs = append(probs, fmt.Sprintf("targets[%d].alias %q must be shaped @lab/<model>", i, t.Alias))
		}
		if seen[t.Alias] {
			probs = append(probs, fmt.Sprintf("targets[%d].alias %q duplicates an earlier alias", i, t.Alias))
		}
		seen[t.Alias] = true
		if _, err := normalizeLabTargetBaseURL(t.BaseURL); err != nil {
			probs = append(probs, fmt.Sprintf("targets[%d].base_url: %v", i, err))
		}
		if strings.TrimSpace(t.BoxID) != t.BoxID || strings.ContainsAny(t.BoxID, "/\\") {
			probs = append(probs, fmt.Sprintf("targets[%d].box_id must be a clean roster id when set", i))
		}
		if strings.TrimSpace(t.Roster) != t.Roster {
			probs = append(probs, fmt.Sprintf("targets[%d].roster must not carry leading/trailing whitespace", i))
		}
	}
	return probs
}

func findLabTarget(doc labTargetsFile, alias string) (labTargetConfig, bool) {
	for _, t := range doc.Targets {
		if t.Alias == alias {
			return t, true
		}
	}
	return labTargetConfig{}, false
}

func validateLabTargetReport(target labTargetConfig, model string, opts labTargetResolveOpts) (boxID, evidence string, err error) {
	rosterPath := opts.rosterPath
	if strings.TrimSpace(rosterPath) == "" {
		rosterPath = strings.TrimSpace(target.Roster)
	}
	ro, err := labTargetRoster(rosterPath)
	if err != nil {
		return "", "", err
	}
	dir, err := labReportsDir(opts.reportsDir)
	if err != nil {
		return "", "", err
	}
	reps := fleet.ReadReports(dir, ro)
	snap := fleet.Fold(ro, reps, fleet.FoldOpts{})
	wantBox := strings.TrimSpace(target.BoxID)
	var sawBox, sawFresh, sawReady bool
	for _, row := range snap.Rows {
		if wantBox != "" && row.ID != wantBox {
			continue
		}
		if wantBox != "" {
			sawBox = true
		}
		if row.Err != "" || row.AgeSec > fleet.DefaultStaleSec || !row.State.Healthy() || row.Inference == nil {
			continue
		}
		sawFresh = true
		// Fleet summaries count degraded inference as "useful" for operator triage, but a
		// lab target alias feeds an interactive guarded dev session. Admit only a ready
		// route so a degraded/fallback heartbeat cannot silently become a guard target.
		if row.Inference.Status != fleet.InferenceReady {
			continue
		}
		sawReady = true
		if model != "" && !strings.EqualFold(strings.TrimSpace(row.Inference.Model), model) {
			continue
		}
		return row.ID, "scrubbed-fleet-report", nil
	}
	switch {
	case wantBox != "" && !sawBox:
		return "", "", fmt.Errorf("LAB_TARGET_REPORT_NOT_USEFUL: target box %s is not in the roster", wantBox)
	case !sawFresh:
		return "", "", errors.New("LAB_TARGET_REPORT_NOT_USEFUL: no fresh healthy report backs this alias")
	case !sawReady:
		return "", "", errors.New("LAB_TARGET_REPORT_NOT_USEFUL: fresh report exists, but inference is not ready")
	default:
		return "", "", fmt.Errorf("LAB_TARGET_REPORT_NOT_USEFUL: no fresh useful report advertises model %s", model)
	}
}

func labTargetRoster(rosterPath string) (fleet.Roster, error) {
	var (
		ro  fleet.Roster
		err error
	)
	if strings.TrimSpace(rosterPath) != "" {
		ro, err = fleet.LoadRosterFile(rosterPath)
	} else {
		ro, err = fleet.LoadRoster(bytes.NewReader(labDefaultRosterJSON))
	}
	if err != nil {
		return fleet.Roster{}, fmt.Errorf("LAB_TARGET_ROSTER_INVALID: %v", err)
	}
	if probs := ro.Validate(); len(probs) > 0 {
		return fleet.Roster{}, fmt.Errorf("LAB_TARGET_ROSTER_INVALID: %s", strings.Join(probs, "; "))
	}
	return ro, nil
}

func isLabTargetAlias(s string) bool {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, labTargetAliasPrefix) {
		return false
	}
	rest := strings.TrimPrefix(s, labTargetAliasPrefix)
	if rest == "" || rest != strings.TrimSpace(rest) {
		return false
	}
	if strings.ContainsAny(rest, "/\\") || strings.Contains(rest, "..") {
		return false
	}
	return true
}

func normalizeLabTargetBaseURL(raw string) (string, error) {
	s := strings.TrimRight(strings.TrimSpace(raw), "/")
	if s == "" {
		return "", errors.New("empty base_url")
	}
	u, err := url.Parse(s)
	if err != nil {
		return "", err
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("base_url scheme must be http or https, got %q", u.Scheme)
	}
	if u.Host == "" {
		return "", errors.New("base_url host is empty")
	}
	if strings.TrimRight(u.Path, "/") == "/v1" {
		u.Path = strings.TrimSuffix(strings.TrimRight(u.Path, "/"), "/v1")
	}
	u.RawQuery, u.Fragment = "", ""
	return strings.TrimRight(u.String(), "/"), nil
}

package evebridge

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"path"
	"regexp"
	"sort"
	"strings"
)

const ScheduleSchema = "fak-eve-schedules/1"

const (
	CodeScheduleRootOnly   = "EVE_SCHEDULE_ROOT_ONLY"
	CodeScheduleMalformed  = "EVE_SCHEDULE_MALFORMED"
	CodeScheduleDuplicate  = "EVE_SCHEDULE_DUPLICATE"
	CodeScheduleCustomHost = "EVE_SCHEDULE_CUSTOM_HOST_UNWIRED"
)

type ScheduleDiagnostic struct {
	Code         string `json:"code"`
	Severity     string `json:"severity"`
	Message      string `json:"message"`
	EvidencePath string `json:"evidence_path,omitempty"`
}

type SchedulePrincipal struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
}

type CronUnit struct {
	ID              string `json:"id"`
	CronUTC         string `json:"cron_utc"`
	Command         string `json:"command"`
	SourcePath      string `json:"source_path"`
	OverlapPolicy   string `json:"overlap_policy"`
	MissedRunPolicy string `json:"missed_run_policy"`
}

type ScheduleProjection struct {
	ID             string            `json:"id"`
	SourcePath     string            `json:"source_path"`
	CronUTC        string            `json:"cron_utc"`
	Form           string            `json:"form"`
	ChannelTarget  string            `json:"channel_target,omitempty"`
	Principal      SchedulePrincipal `json:"principal"`
	SideEffecting  bool              `json:"side_effecting"`
	HostProjection string            `json:"host_projection"`
	Unit           CronUnit          `json:"cron_unit"`
	Warnings       []string          `json:"warnings,omitempty"`
}

type ScheduleInventory struct {
	Schema      string               `json:"schema"`
	OK          bool                 `json:"ok"`
	Host        string               `json:"host"`
	Schedules   []ScheduleProjection `json:"schedules,omitempty"`
	Diagnostics []ScheduleDiagnostic `json:"diagnostics,omitempty"`
}

func (i ScheduleInventory) JSON() []byte {
	b, _ := json.MarshalIndent(i, "", "  ")
	return append(b, '\n')
}

var (
	cronField    = regexp.MustCompile(`(?m)\bcron\s*:\s*["'\x60]([^"'\x60]+)["'\x60]`)
	receiveField = regexp.MustCompile(`(?m)\breceive\s*\(\s*([A-Za-z_$][\w$]*)`)
)

// InspectSchedules projects authored or compiled Eve schedules into fak's
// recurring-job view. It is descriptive only: it never installs or fires jobs.
func InspectSchedules(root fs.FS, host string) (ScheduleInventory, error) {
	host = strings.TrimSpace(host)
	if host == "" {
		host = "eve-dev"
	}
	out := ScheduleInventory{Schema: ScheduleSchema, Host: host}
	if _, err := fs.Stat(root, "."); err != nil {
		return out, err
	}
	seen := map[string]string{}
	if exists(root, "agent/agent.ts") || exists(root, "agent/agent.js") {
		err := fs.WalkDir(root, "agent", func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			if strings.HasPrefix(p, "agent/subagents/") && strings.Contains(p, "/schedules/") {
				out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleRootOnly, "fail", "Eve schedules are root-only", p))
				return nil
			}
			if !strings.HasPrefix(p, "agent/schedules/") || !(sourceFile(p) || path.Ext(p) == ".md") {
				return nil
			}
			b, readErr := fs.ReadFile(root, p)
			if readErr != nil {
				return readErr
			}
			id := runtimeName(strings.TrimSuffix(strings.TrimPrefix(p, "agent/schedules/"), path.Ext(p)))
			projection, parseErr := sourceSchedule(id, p, string(b), host)
			if parseErr != nil {
				out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleMalformed, "fail", parseErr.Error(), p))
				return nil
			}
			addSchedule(&out, projection, seen)
			return nil
		})
		if err != nil {
			return out, err
		}
	} else if manifest := compiledManifestPath(root); manifest != "" {
		b, err := fs.ReadFile(root, manifest)
		if err != nil {
			return out, err
		}
		var doc struct {
			Schedules []json.RawMessage `json:"schedules"`
		}
		if err := json.Unmarshal(b, &doc); err != nil {
			out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleMalformed, "fail", err.Error(), manifest))
		} else {
			for _, raw := range doc.Schedules {
				var item struct {
					Name, LogicalPath, Cron, Markdown string
					HasRun                            bool   `json:"hasRun"`
					Disabled                          bool   `json:"disabled"`
					ChannelTarget                     string `json:"channelTarget"`
				}
				if err := json.Unmarshal(raw, &item); err != nil || item.Name == "" || item.Cron == "" {
					out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleMalformed, "fail", "compiled schedule lacks name or cron", manifest))
					continue
				}
				if item.Disabled {
					continue
				}
				form := "prompt"
				if item.HasRun {
					form = "handler"
				}
				cron, err := normalizeCron(item.Cron)
				if err != nil {
					out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleMalformed, "fail", err.Error(), item.LogicalPath))
					continue
				}
				addSchedule(&out, makeSchedule(runtimeName(item.Name), item.LogicalPath, cron, form, item.ChannelTarget, host), seen)
			}
		}
	} else {
		out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleMalformed, "fail", "expected an Eve source tree or compiled manifest", "."))
	}
	if hostProjection(host) == "custom" {
		out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleCustomHost, "warn", "custom hosts compile schedules but must wire recurring dispatch explicitly", "."))
	}
	sort.Slice(out.Schedules, func(a, b int) bool { return out.Schedules[a].ID < out.Schedules[b].ID })
	sort.Slice(out.Diagnostics, func(a, b int) bool {
		if out.Diagnostics[a].Code == out.Diagnostics[b].Code {
			return out.Diagnostics[a].EvidencePath < out.Diagnostics[b].EvidencePath
		}
		return out.Diagnostics[a].Code < out.Diagnostics[b].Code
	})
	out.OK = true
	for _, d := range out.Diagnostics {
		if d.Severity == "fail" {
			out.OK = false
		}
	}
	return out, nil
}

func sourceSchedule(id, p, body, host string) (ScheduleProjection, error) {
	cron := ""
	if path.Ext(p) == ".md" {
		for _, line := range strings.Split(body, "\n") {
			if strings.HasPrefix(strings.TrimSpace(line), "cron:") {
				cron = strings.Trim(strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "cron:")), "\"'")
				break
			}
		}
	} else if match := cronField.FindStringSubmatch(body); len(match) == 2 {
		cron = match[1]
	}
	cron, err := normalizeCron(cron)
	if err != nil {
		return ScheduleProjection{}, err
	}
	form := "prompt"
	if path.Ext(p) != ".md" && regexp.MustCompile(`\brun\s*[:(]`).MatchString(body) {
		form = "handler"
	}
	target := ""
	if match := receiveField.FindStringSubmatch(body); len(match) == 2 {
		target = match[1]
	}
	return makeSchedule(id, p, cron, form, target, host), nil
}
func makeSchedule(id, p, cron, form, target, host string) ScheduleProjection {
	side := form == "handler" || target != ""
	warnings := []string(nil)
	overlap, missed := "skip-while-running", "skip"
	if side {
		warnings = []string{"side-effecting schedule requires overlap and missed-run review"}
	}
	projection := hostProjection(host)
	return ScheduleProjection{ID: id, SourcePath: p, CronUTC: cron, Form: form, ChannelTarget: target, Principal: SchedulePrincipal{Kind: "app", Name: "eve-app"}, SideEffecting: side, HostProjection: projection, Warnings: warnings, Unit: CronUnit{ID: "eve-" + id, CronUTC: cron, Command: "eve dev dispatch --schedule " + id, SourcePath: p, OverlapPolicy: overlap, MissedRunPolicy: missed}}
}
func addSchedule(out *ScheduleInventory, p ScheduleProjection, seen map[string]string) {
	if prior, ok := seen[p.ID]; ok {
		out.Diagnostics = append(out.Diagnostics, scheduleDiag(CodeScheduleDuplicate, "fail", "schedule id collides with "+prior, p.SourcePath))
		return
	}
	seen[p.ID] = p.SourcePath
	out.Schedules = append(out.Schedules, p)
}
func normalizeCron(value string) (string, error) {
	fields := strings.Fields(value)
	if len(fields) != 5 {
		return "", fmt.Errorf("cron must contain exactly five UTC fields")
	}
	for _, f := range fields {
		if strings.HasPrefix(strings.ToUpper(f), "TZ=") || strings.HasPrefix(strings.ToUpper(f), "CRON_TZ=") {
			return "", fmt.Errorf("cron timezone prefixes are unsupported; Eve schedules are UTC")
		}
	}
	return strings.Join(fields, " "), nil
}
func hostProjection(host string) string {
	switch strings.ToLower(host) {
	case "eve-dev", "dev":
		return "eve-dev-dispatch"
	case "eve-start", "start":
		return "eve-start"
	case "vercel", "vercel-cron":
		return "vercel-cron"
	default:
		return "custom"
	}
}
func scheduleDiag(code, severity, message, p string) ScheduleDiagnostic {
	return ScheduleDiagnostic{Code: code, Severity: severity, Message: message, EvidencePath: p}
}

type DispatchReceipt struct {
	ScheduleID   string `json:"schedule_id"`
	SessionID    string `json:"session_id"`
	LedgerUnitID string `json:"ledger_unit_id"`
}

func RecordDevDispatch(inventory ScheduleInventory, scheduleID, sessionID string) (DispatchReceipt, error) {
	if sessionID == "" {
		return DispatchReceipt{}, fmt.Errorf("dev dispatch returned no session id")
	}
	for _, s := range inventory.Schedules {
		if s.ID == scheduleID {
			return DispatchReceipt{ScheduleID: s.ID, SessionID: sessionID, LedgerUnitID: s.Unit.ID}, nil
		}
	}
	return DispatchReceipt{}, fmt.Errorf("schedule %q is absent from the ledger", scheduleID)
}

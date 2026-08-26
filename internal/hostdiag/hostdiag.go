package hostdiag

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	CensusSchema      = "fak.hostdiag-census.v1"
	CorrelationSchema = "fak.hostdiag-correlation.v1"
)

type ProcessSample struct {
	Schema          string `json:"schema"`
	SampleID        string `json:"sample_id"`
	SampledAtMS     int64  `json:"sampled_at_ms"`
	SampledAtUTC    string `json:"sampled_at_utc"`
	PID             int    `json:"pid"`
	ProcessStartMS  int64  `json:"process_start_ms"`
	ProcessStartUTC string `json:"process_start_utc"`
	Executable      string `json:"executable"`
	ExecutableSHA   string `json:"executable_sha256,omitempty"`
	BuildRevision   string `json:"build_revision,omitempty"`
	CommandClass    string `json:"command_class"`
	Session         string `json:"session,omitempty"`
	PrivateBytes    uint64 `json:"private_bytes,omitempty"`
	WorkingSetBytes uint64 `json:"working_set_bytes,omitempty"`
	HandleCount     uint32 `json:"handle_count,omitempty"`
	ThreadCount     uint32 `json:"thread_count,omitempty"`
}

type ResourceCulprit struct {
	Image string `json:"image"`
	PID   int    `json:"pid"`
	Bytes uint64 `json:"bytes"`
}

type OwnedShellLaunch struct {
	TimestampUTCMS    int64  `json:"timestamp_utc_ms"`
	ParentPID         int    `json:"parent_pid"`
	ChildPID          int    `json:"child_pid"`
	ChildCreatedUTCMS int64  `json:"child_created_utc_ms"`
	LaunchID          string `json:"launch_id"`
	LaunchClass       string `json:"launch_class"`
	ShellImage        string `json:"shell_image"`
	ShellEdition      string `json:"shell_edition"`
	ShellVersion      string `json:"shell_version"`
	Outcome           string `json:"outcome"`
	ErrorClass        string `json:"error_class"`
}

type ApplicationFault struct {
	AppVersion    string `json:"app_version,omitempty"`
	Module        string `json:"module,omitempty"`
	ModuleVersion string `json:"module_version,omitempty"`
	ExceptionCode string `json:"exception_code,omitempty"`
	FaultOffset   string `json:"fault_offset,omitempty"`
}

var lowVirtualMemoryCulpritRE = regexp.MustCompile(`(?i)([a-z0-9_.-]+\.exe)\s+\((\d+)\)\s+consumed\s+(\d+)\s+bytes`)

// ParseLowVirtualMemoryCulprits extracts the typed process rows present in the
// English Event 2004 rendering. An empty result is intentional: callers retain
// the event itself when Windows renders a localized or otherwise unfamiliar message.
func ParseLowVirtualMemoryCulprits(message string) []ResourceCulprit {
	matches := lowVirtualMemoryCulpritRE.FindAllStringSubmatch(message, -1)
	culprits := make([]ResourceCulprit, 0, len(matches))
	for _, match := range matches {
		pid, pidErr := strconv.Atoi(match[2])
		bytes, bytesErr := strconv.ParseUint(match[3], 10, 64)
		if pidErr != nil || bytesErr != nil {
			continue
		}
		culprits = append(culprits, ResourceCulprit{Image: match[1], PID: pid, Bytes: bytes})
	}
	return culprits
}

type ResourceEvent struct {
	TimeMS         int64             `json:"time_ms"`
	Source         string            `json:"source"`
	EventID        int               `json:"windows_event_id"`
	RecordID       string            `json:"record_id,omitempty"`
	Name           string            `json:"event_name"`
	ReportID       string            `json:"report_id,omitempty"`
	App            string            `json:"app,omitempty"`
	Message        string            `json:"message,omitempty"`
	Culprits       []ResourceCulprit `json:"culprits,omitempty"`
	Fault          *ApplicationFault `json:"application_fault,omitempty"`
	ProcessID      int               `json:"process_id,omitempty"`
	ProcessStartMS int64             `json:"process_start_ms,omitempty"`
}

type Correlation struct {
	Schema        string            `json:"schema"`
	CorrelationID string            `json:"correlation_id"`
	TimeMS        int64             `json:"time_ms"`
	TimeUTC       string            `json:"time_utc"`
	Source        string            `json:"source"`
	WindowsID     int               `json:"windows_event_id"`
	EventName     string            `json:"event_name"`
	ReportID      string            `json:"report_id,omitempty"`
	App           string            `json:"app,omitempty"`
	Culprits      []ResourceCulprit `json:"culprits,omitempty"`
	Fault         *ApplicationFault `json:"application_fault,omitempty"`
	OwnedLaunch   *OwnedShellLaunch `json:"owned_shell_launch,omitempty"`
	Status        string            `json:"status"`
	Reason        string            `json:"reason"`
	Candidates    []ProcessSample   `json:"candidates,omitempty"`
	Observational bool              `json:"observational"`
}

func NewProcessSample(at time.Time, pid int, started time.Time, executable, executableSHA, buildRevision, commandClass, session string, privateBytes, workingSetBytes uint64, handles, threads uint32) ProcessSample {
	at = at.UTC()
	started = started.UTC()
	key := strings.Join([]string{at.Format(time.RFC3339Nano), itoa(pid), started.Format(time.RFC3339Nano), strings.ToLower(executableSHA), commandClass, session}, "|")
	return ProcessSample{
		Schema: CensusSchema, SampleID: "proc-" + digest(key), SampledAtMS: at.UnixMilli(), SampledAtUTC: at.Format(time.RFC3339Nano), PID: pid,
		ProcessStartMS: started.UnixMilli(), ProcessStartUTC: started.Format(time.RFC3339Nano), Executable: executable, ExecutableSHA: executableSHA,
		BuildRevision: buildRevision, CommandClass: commandClass, Session: session, PrivateBytes: privateBytes, WorkingSetBytes: workingSetBytes,
		HandleCount: handles, ThreadCount: threads,
	}
}

func ClassifyCommand(commandLine string) string {
	fields := strings.Fields(strings.TrimSpace(commandLine))
	if len(fields) < 2 {
		return "root"
	}
	verb := strings.ToLower(strings.TrimLeft(fields[1], "-"))
	for _, allowed := range []string{"guard", "serve", "agent", "gateway", "stallscan", "host-crash", "watchdog-audit-run", "ultracode", "run", "preflight"} {
		if verb == allowed {
			return verb
		}
	}
	return "other"
}

func Correlate(event ResourceEvent, samples []ProcessSample) (Correlation, bool) {
	return CorrelateWithOwnedLaunches(event, samples, nil)
}

func CorrelateWithOwnedLaunches(event ResourceEvent, samples []ProcessSample, launches []OwnedShellLaunch) (Correlation, bool) {
	name := strings.ToUpper(strings.TrimSpace(event.Name))
	isLowVirtualMemory := name == "LOW_VIRTUAL_MEMORY" && event.EventID == 2004
	isShellCrash := name == "POWERSHELL_PROCESS_CRASH" && event.EventID == 1000
	isWindowsShellCrash := name == "WINDOWS_SHELL_PROCESS_CRASH" && event.EventID == 1000
	isRadar := name == "RADAR_PRE_LEAK_64" && event.EventID == 1001
	isResolver := strings.HasPrefix(name, "RESOURCE_EXHAUSTION_") && (event.EventID == 1014 || event.EventID == 1015)
	if event.TimeMS <= 0 || (!isLowVirtualMemory && !isShellCrash && !isWindowsShellCrash && !isRadar && !isResolver) {
		return Correlation{}, false
	}
	app := strings.TrimSpace(event.App)
	if isShellCrash && !strings.EqualFold(app, "pwsh.exe") && !strings.EqualFold(app, "powershell.exe") {
		return Correlation{}, false
	}
	if isWindowsShellCrash && !strings.EqualFold(app, "explorer.exe") {
		return Correlation{}, false
	}
	if isRadar && !strings.EqualFold(app, "fak.exe") {
		return Correlation{}, false
	}
	candidates := make([]ProcessSample, 0)
	for _, sample := range samples {
		if sample.Schema != CensusSchema || sample.ProcessStartMS > event.TimeMS || sample.SampledAtMS < event.TimeMS {
			continue
		}
		if sample.Executable != "" && !strings.EqualFold(baseName(sample.Executable), "fak.exe") {
			continue
		}
		if isLowVirtualMemory && !eventNamesFakPID(event.Culprits, sample.PID) {
			continue
		}
		if isShellCrash || isWindowsShellCrash || isResolver {
			continue
		}
		candidates = append(candidates, sample)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].ProcessStartMS != candidates[j].ProcessStartMS {
			return candidates[i].ProcessStartMS > candidates[j].ProcessStartMS
		}
		return candidates[i].PID < candidates[j].PID
	})
	var owned *OwnedShellLaunch
	if isShellCrash && event.ProcessID > 0 && event.ProcessStartMS > 0 {
		for i := range launches {
			launch := &launches[i]
			if launch.ChildPID == event.ProcessID && launch.ChildCreatedUTCMS == event.ProcessStartMS && (owned == nil || launch.TimestampUTCMS > owned.TimestampUTCMS) {
				copy := *launch
				owned = &copy
			}
		}
	}
	status, reason := "historical_unresolved", "no durable fak process census spans the event time"
	if isWindowsShellCrash {
		reason = "Windows shell crash is retained as observational host evidence and is not attributed to fak"
	}
	if isShellCrash {
		if owned != nil {
			status, reason = "identified", "PowerShell crash PID and process creation time exactly match a fak-owned launch receipt"
		} else {
			reason = "no exact PID and process creation time match in fak-owned launch receipts"
		}
	}
	if len(candidates) == 1 {
		status, reason = "identified", "one durable fak process census spans the event time"
	} else if len(candidates) > 1 {
		status, reason = "ambiguous", "multiple durable fak process census rows span the event time"
	}
	key := strings.Join([]string{strconv.FormatInt(event.TimeMS, 10), strings.ToLower(event.Source), itoa(event.EventID), strings.ToLower(event.RecordID), strings.ToLower(event.ReportID), name}, "|")
	return Correlation{Schema: CorrelationSchema, CorrelationID: "corr-" + digest(key), TimeMS: event.TimeMS, TimeUTC: time.UnixMilli(event.TimeMS).UTC().Format(time.RFC3339Nano), Source: event.Source, WindowsID: event.EventID, EventName: name, ReportID: event.ReportID, App: app, Culprits: append([]ResourceCulprit(nil), event.Culprits...), Fault: event.Fault, OwnedLaunch: owned, Status: status, Reason: reason, Candidates: candidates, Observational: true}, true
}

func eventNamesFakPID(culprits []ResourceCulprit, pid int) bool {
	for _, culprit := range culprits {
		if culprit.PID == pid && strings.EqualFold(culprit.Image, "fak.exe") {
			return true
		}
	}
	return false
}

func digest(value string) string {
	hash := sha256.Sum256([]byte(value))
	return hex.EncodeToString(hash[:12])
}

func baseName(path string) string {
	path = strings.ReplaceAll(path, "\\", "/")
	if index := strings.LastIndexByte(path, '/'); index >= 0 {
		return path[index+1:]
	}
	return path
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	negative := value < 0
	if negative {
		value = -value
	}
	var buf [24]byte
	index := len(buf)
	for value > 0 {
		index--
		buf[index] = byte('0' + value%10)
		value /= 10
	}
	if negative {
		index--
		buf[index] = '-'
	}
	return string(buf[index:])
}

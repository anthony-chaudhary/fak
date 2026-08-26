package hostdiag

import (
	"crypto/sha256"
	"encoding/hex"
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

type ResourceEvent struct {
	TimeMS   int64  `json:"time_ms"`
	Source   string `json:"source"`
	EventID  int    `json:"windows_event_id"`
	RecordID string `json:"record_id,omitempty"`
	Name     string `json:"event_name"`
	ReportID string `json:"report_id,omitempty"`
	App      string `json:"app,omitempty"`
	Message  string `json:"message,omitempty"`
}

type Correlation struct {
	Schema        string          `json:"schema"`
	CorrelationID string          `json:"correlation_id"`
	TimeMS        int64           `json:"time_ms"`
	TimeUTC       string          `json:"time_utc"`
	Source        string          `json:"source"`
	WindowsID     int             `json:"windows_event_id"`
	EventName     string          `json:"event_name"`
	ReportID      string          `json:"report_id,omitempty"`
	App           string          `json:"app,omitempty"`
	Status        string          `json:"status"`
	Reason        string          `json:"reason"`
	Candidates    []ProcessSample `json:"candidates,omitempty"`
	Observational bool            `json:"observational"`
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
	name := strings.ToUpper(strings.TrimSpace(event.Name))
	if event.TimeMS <= 0 || !(name == "RADAR_PRE_LEAK_64" || event.EventID == 1014 || event.EventID == 1015) {
		return Correlation{}, false
	}
	app := strings.TrimSpace(event.App)
	if app != "" && !strings.EqualFold(app, "fak.exe") {
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
		candidates = append(candidates, sample)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].ProcessStartMS != candidates[j].ProcessStartMS {
			return candidates[i].ProcessStartMS > candidates[j].ProcessStartMS
		}
		return candidates[i].PID < candidates[j].PID
	})
	status, reason := "historical_unresolved", "no durable fak process census spans the event time"
	if len(candidates) == 1 {
		status, reason = "identified", "one durable fak process census spans the event time"
	} else if len(candidates) > 1 {
		status, reason = "ambiguous", "multiple durable fak process census rows span the event time"
	}
	key := strings.Join([]string{strconv.FormatInt(event.TimeMS, 10), strings.ToLower(event.Source), itoa(event.EventID), strings.ToLower(event.RecordID), strings.ToLower(event.ReportID), name}, "|")
	return Correlation{Schema: CorrelationSchema, CorrelationID: "corr-" + digest(key), TimeMS: event.TimeMS, TimeUTC: time.UnixMilli(event.TimeMS).UTC().Format(time.RFC3339Nano), Source: event.Source, WindowsID: event.EventID, EventName: name, ReportID: event.ReportID, App: app, Status: status, Reason: reason, Candidates: candidates, Observational: true}, true
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

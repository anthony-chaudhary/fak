package serviceledger

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// AdapterConfig binds a native manager's own identifiers (unit name, service
// display name, launchd label) to one servicespec identity. Node and Service
// are required; Unit filters which native rows belong to this workload (empty
// accepts every row the mapper understands).
type AdapterConfig struct {
	Identity servicespec.Identity
	Unit     string
}

func (c AdapterConfig) validate() error {
	if strings.TrimSpace(c.Identity.Node) == "" || strings.TrimSpace(c.Identity.Service) == "" {
		return errors.New("serviceledger: adapter config needs identity.node and identity.service")
	}
	return nil
}

// Source names for the native adapters.
const (
	SourceWindowsEventLog = "windows-eventlog"
	SourceJournald        = "journald"
	SourceLaunchd         = "launchd"
	SourceFak             = "fak"
)

// ---------------------------------------------------------------------------
// Windows Event Log (wevtutil qe <channel> /f:xml). Covers SCM
// service-crash/restart/state rows (7031/7034/7036/7040), EventLog boot
// markers (6005), and Application Error 1000 host-crash rows.
// ---------------------------------------------------------------------------

type winEventXML struct {
	System struct {
		Provider struct {
			Name string `xml:"Name,attr"`
		} `xml:"Provider"`
		EventID     int `xml:"EventID"`
		TimeCreated struct {
			SystemTime string `xml:"SystemTime,attr"`
		} `xml:"TimeCreated"`
		EventRecordID string `xml:"EventRecordID"`
		Execution     struct {
			ProcessID int `xml:"ProcessID,attr"`
		} `xml:"Execution"`
		Channel  string `xml:"Channel"`
		Computer string `xml:"Computer"`
	} `xml:"System"`
	EventData struct {
		Data []struct {
			Name  string `xml:"Name,attr"`
			Value string `xml:",chardata"`
		} `xml:"Data"`
	} `xml:"EventData"`
}

func (w winEventXML) data(name string) string {
	for _, d := range w.EventData.Data {
		if d.Name == name {
			return strings.TrimSpace(d.Value)
		}
	}
	return ""
}

func (w winEventXML) anyDataContains(s string) bool {
	for _, d := range w.EventData.Data {
		if strings.Contains(d.Value, s) {
			return true
		}
	}
	return false
}

// AdaptWindowsEventXML parses a Windows Event Log XML export (a
// sequence of <Event> elements, exactly what `wevtutil qe ... /f:xml`
// prints) into v1 events. Exact-once UID = channel + event record ID.
func AdaptWindowsEventXML(r io.Reader, cfg AdapterConfig) ([]Event, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	dec := xml.NewDecoder(r)
	var out []Event
	for {
		var we winEventXML
		err := dec.Decode(&we)
		if errors.Is(err, io.EOF) {
			return out, nil
		}
		if err != nil {
			return nil, fmt.Errorf("serviceledger: windows event xml: %w", err)
		}
		ts, err := time.Parse(time.RFC3339Nano, we.System.TimeCreated.SystemTime)
		if err != nil {
			return nil, fmt.Errorf("serviceledger: windows event time %q: %w", we.System.TimeCreated.SystemTime, err)
		}
		base := Event{
			Type:      "", // set per mapping below
			AtUnixMS:  ts.UnixMilli(),
			Source:    SourceWindowsEventLog,
			SourceUID: we.System.Channel + "/" + we.System.EventRecordID,
			Identity:  cfg.Identity,
			Correlation: Correlation{
				PID: we.System.Execution.ProcessID,
			},
		}
		provider, id := we.System.Provider.Name, we.System.EventID
		unitMatch := cfg.Unit == "" || we.data("param1") == cfg.Unit || we.anyDataContains(cfg.Unit)
		switch {
		case provider == "Service Control Manager" && (id == 7031 || id == 7034):
			if !unitMatch {
				continue
			}
			base.Type = EventProcessExit
			base.Exit = &servicespec.ExitRecord{Class: servicespec.ExitCrash, AtUnixMS: base.AtUnixMS}
			base.Detail = fmt.Sprintf("SCM %d: service %s terminated unexpectedly", id, we.data("param1"))
		case provider == "Service Control Manager" && id == 7036:
			if !unitMatch {
				continue
			}
			state := strings.ToLower(we.data("param2"))
			switch state {
			case "running":
				base.Type = EventReadiness
				base.Phase = servicespec.PhaseReady
			case "stopped":
				base.Type = EventProcessExit
				base.Exit = &servicespec.ExitRecord{Class: servicespec.ExitOperatorStop, AtUnixMS: base.AtUnixMS}
			default:
				base.Type = EventReadiness
				base.Phase = servicespec.PhaseStarting
			}
			base.Detail = fmt.Sprintf("SCM 7036: %s entered %s state", we.data("param1"), state)
		case provider == "Service Control Manager" && id == 7040:
			if !unitMatch {
				continue
			}
			base.Type = EventDesiredChange
			if strings.Contains(strings.ToLower(we.data("param3")), "disabled") {
				base.Desired = servicespec.DesiredStopped
			} else {
				base.Desired = servicespec.DesiredRunning
			}
			base.Detail = fmt.Sprintf("SCM 7040: %s start type %s -> %s", we.data("param1"), we.data("param2"), we.data("param3"))
		case provider == "EventLog" && (id == 6005 || id == 6009):
			base.Type = EventBootChange
			base.Correlation.BootID = "win-boot-" + we.System.TimeCreated.SystemTime
			base.Detail = fmt.Sprintf("event log %d: host boot", id)
		case provider == "Microsoft-Windows-TaskScheduler":
			if cfg.Unit != "" && we.data("TaskName") != cfg.Unit && !we.anyDataContains(cfg.Unit) {
				continue
			}
			switch id {
			case 100: // TaskStartEvent: the scheduler launched the task.
				base.Type = EventManagerRestart
				base.Correlation.ManagerInvocation = we.data("InstanceId")
				base.Detail = fmt.Sprintf("task scheduler 100: task %s started", we.data("TaskName"))
			case 102: // TaskSuccessEvent.
				base.Type = EventProcessExit
				base.Exit = &servicespec.ExitRecord{Class: servicespec.ExitClean, AtUnixMS: base.AtUnixMS}
				base.Detail = fmt.Sprintf("task scheduler 102: task %s completed", we.data("TaskName"))
			case 103, 203: // TaskStartFailedEvent / action failed.
				rec := &servicespec.ExitRecord{Class: servicespec.ExitCrash, AtUnixMS: base.AtUnixMS}
				if c, err := strconv.Atoi(we.data("ResultCode")); err == nil {
					rec.Code = c
				}
				base.Type = EventProcessExit
				base.Exit = rec
				base.Detail = fmt.Sprintf("task scheduler %d: task %s failed result=%s", id, we.data("TaskName"), we.data("ResultCode"))
			default:
				continue
			}
		case provider == "Application Error" && id == 1000:
			if cfg.Unit != "" && !we.anyDataContains(cfg.Unit) {
				continue
			}
			base.Type = EventProcessExit
			base.Exit = &servicespec.ExitRecord{Class: servicespec.ExitCrash, AtUnixMS: base.AtUnixMS}
			base.Detail = "application error 1000: " + firstDataValue(we)
		default:
			continue
		}
		out = append(out, base)
	}
}

func firstDataValue(we winEventXML) string {
	if len(we.EventData.Data) > 0 {
		return strings.TrimSpace(we.EventData.Data[0].Value)
	}
	return ""
}

// ---------------------------------------------------------------------------
// systemd journal (journalctl -u UNIT -o json). Exact-once UID = __CURSOR;
// _BOOT_ID transitions synthesize boot-change events, so incarnation changes
// are ledgered even when no single row announces them.
// ---------------------------------------------------------------------------

type journaldEntry struct {
	Cursor   string `json:"__CURSOR"`
	Realtime string `json:"__REALTIME_TIMESTAMP"` // microseconds since epoch
	BootID   string `json:"_BOOT_ID"`
	PID      string `json:"_PID"`
	Unit     string `json:"_SYSTEMD_UNIT"`
	UnitAlt  string `json:"UNIT"`
	Message  string `json:"MESSAGE"`
}

// AdaptJournaldExport parses `journalctl -o json` line output into v1 events.
func AdaptJournaldExport(r io.Reader, cfg AdapterConfig) ([]Event, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var out []Event
	prevBoot := ""
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var je journaldEntry
		if err := json.Unmarshal([]byte(raw), &je); err != nil {
			return nil, fmt.Errorf("serviceledger: journald line %d: %w", line, err)
		}
		usec, err := strconv.ParseInt(je.Realtime, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("serviceledger: journald line %d timestamp %q: %w", line, je.Realtime, err)
		}
		atMS := usec / 1000
		unit := je.Unit
		if unit == "" {
			unit = je.UnitAlt
		}
		if cfg.Unit != "" && unit != cfg.Unit && unit != cfg.Unit+".service" && !strings.Contains(je.Message, cfg.Unit) {
			continue
		}
		pid, _ := strconv.Atoi(je.PID)
		if je.BootID != "" && prevBoot != "" && je.BootID != prevBoot {
			out = append(out, Event{
				Type: EventBootChange, AtUnixMS: atMS, Source: SourceJournald,
				SourceUID: je.Cursor + ":boot", Identity: cfg.Identity,
				Correlation: Correlation{BootID: je.BootID},
				Detail:      "journal boot id changed",
			})
		}
		if je.BootID != "" {
			prevBoot = je.BootID
		}
		base := Event{
			AtUnixMS: atMS, Source: SourceJournald, SourceUID: je.Cursor,
			Identity:    cfg.Identity,
			Correlation: Correlation{BootID: je.BootID, PID: pid},
			Detail:      je.Message,
		}
		msg := je.Message
		switch {
		case strings.HasPrefix(msg, "Started "):
			base.Type = EventReadiness
			base.Phase = servicespec.PhaseReady
		case strings.Contains(msg, "Main process exited"):
			base.Type = EventProcessExit
			rec := &servicespec.ExitRecord{Class: servicespec.ExitCrash, AtUnixMS: atMS}
			if strings.Contains(msg, "status=0/SUCCESS") {
				rec.Class = servicespec.ExitClean
			}
			if i := strings.Index(msg, "status="); i >= 0 {
				code := msg[i+len("status="):]
				if j := strings.IndexAny(code, "/ "); j > 0 {
					code = code[:j]
				}
				rec.Code, _ = strconv.Atoi(code)
			}
			base.Exit = rec
		case strings.Contains(msg, "Watchdog timeout"):
			base.Type = EventWatchdogTimeout
		case strings.Contains(msg, "Scheduled restart job"):
			base.Type = EventManagerRestart
		case strings.HasPrefix(msg, "Stopping "):
			base.Type = EventDesiredChange
			base.Desired = servicespec.DesiredStopped
		default:
			continue
		}
		out = append(out, base)
	}
	return out, sc.Err()
}

// ---------------------------------------------------------------------------
// launchd (log show --style ndjson --predicate 'subsystem == "com.apple.xpc.launchd"').
// Exact-once UID = digest of the raw line (launchd rows carry no record id).
// ---------------------------------------------------------------------------

type launchdEntry struct {
	Timestamp    string `json:"timestamp"`
	EventMessage string `json:"eventMessage"`
	ProcessID    int    `json:"processID"`
}

const launchdTimeRef = "2006-01-02 15:04:05.999999-0700"

// AdaptLaunchdNDJSON parses `log show --style ndjson` launchd output into v1
// events.
func AdaptLaunchdNDJSON(r io.Reader, cfg AdapterConfig) ([]Event, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var out []Event
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var le launchdEntry
		if err := json.Unmarshal([]byte(raw), &le); err != nil {
			return nil, fmt.Errorf("serviceledger: launchd line %d: %w", line, err)
		}
		if cfg.Unit != "" && !strings.Contains(le.EventMessage, cfg.Unit) {
			continue
		}
		ts, err := time.Parse(launchdTimeRef, le.Timestamp)
		if err != nil {
			return nil, fmt.Errorf("serviceledger: launchd line %d timestamp %q: %w", line, le.Timestamp, err)
		}
		sum := sha256.Sum256([]byte(raw))
		base := Event{
			AtUnixMS: ts.UnixMilli(), Source: SourceLaunchd,
			SourceUID: "line-" + hex.EncodeToString(sum[:16]),
			Identity:  cfg.Identity,
			Correlation: Correlation{
				PID: le.ProcessID,
			},
			Detail: le.EventMessage,
		}
		msg := le.EventMessage
		switch {
		case strings.Contains(msg, "Service exited with abnormal code") || strings.Contains(msg, "exited due to signal"):
			rec := &servicespec.ExitRecord{Class: servicespec.ExitCrash, AtUnixMS: base.AtUnixMS}
			if i := strings.Index(msg, "code: "); i >= 0 {
				rec.Code, _ = strconv.Atoi(strings.TrimSpace(msg[i+len("code: "):]))
			}
			base.Type = EventProcessExit
			base.Exit = rec
		case strings.Contains(msg, "exited voluntarily"):
			base.Type = EventProcessExit
			base.Exit = &servicespec.ExitRecord{Class: servicespec.ExitClean, AtUnixMS: base.AtUnixMS}
		case strings.Contains(msg, "WILL_SPAWN") || strings.Contains(msg, "service state: spawning"):
			base.Type = EventManagerRestart
		case strings.Contains(msg, "service state: running"):
			base.Type = EventReadiness
			base.Phase = servicespec.PhaseReady
		case strings.Contains(msg, "throttl"):
			base.Type = EventManagerRestart
		default:
			continue
		}
		out = append(out, base)
	}
	return out, sc.Err()
}

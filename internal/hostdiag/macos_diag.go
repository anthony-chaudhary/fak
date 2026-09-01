package hostdiag

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	MacOSResourceIncidentEventName  = "MACOS_RESOURCE_INCIDENT"
	MacOSResourceIncidentDiskWrites = "disk_writes"
	MacOSDiagnosticReportsSource    = "macOS DiagnosticReports"
	MacOSDiagFixtureMaxBytes        = 64 << 10
	macOSDiagSanitizedArtifactName  = "macos-resource-incident.diag"
	macOSDiagTimestampLayout        = "2006-01-02 15:04:05.999999999 -0700"
)

var (
	macOSDiagStartRE      = regexp.MustCompile(`(?m)^Date/Time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)? [+-]\d{4})\s*$`)
	macOSDiagEndRE        = regexp.MustCompile(`(?m)^End time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)? [+-]\d{4})\s*$`)
	macOSDiagCommandRE    = regexp.MustCompile(`(?m)^Command:\s*([A-Za-z0-9._-]+)\s*$`)
	macOSDiagPIDRE        = regexp.MustCompile(`(?m)^PID:\s*(\d+)\s*$`)
	macOSDiagEventRE      = regexp.MustCompile(`(?m)^Event:\s*(disk writes)\s*$`)
	macOSDiagActionRE     = regexp.MustCompile(`(?m)^Action taken:\s*(none)\s*$`)
	macOSDiagWritesRE     = regexp.MustCompile(`(?m)^Writes:\s*([0-9][0-9,]*(?:\.[0-9]+)?) MB of file backed memory dirtied over (\d+) seconds \(([0-9][0-9,]*(?:\.[0-9]+)?) MB per second average\)(?:,[^\r\n]*)?\s*$`)
	macOSDiagFootprintRE  = regexp.MustCompile(`(?m)^Footprint:\s*([0-9][0-9,]*(?:\.[0-9]+)?) MB\s*$`)
	macOSDiagBinaryRE     = regexp.MustCompile(`(?mi)^\s*0x[0-9a-f]+\s+-\s+0x[0-9a-f]+\s+\+?fak(?:\s+\([^)]*\))?\s+<([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})>(?:\s+.*)?$`)
	macOSDiagStackFrameRE = regexp.MustCompile(`^\s+\d+\s+(.+?)\s*$`)
	macOSDiagWriteFrameRE = regexp.MustCompile(`^write(?:\s+\+\s+\d+)?(?:\s+\([^)]*\))?$`)
)

// ParseMacOSResourceIncident projects the allowlisted fields from a sanitized
// macOS resource .diag fixture. The raw body is used only for parsing and its
// digest; it is never retained in the normalized event.
func ParseMacOSResourceIncident(sourceName string, data []byte) (ResourceEvent, error) {
	if len(data) == 0 {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident fixture is empty")
	}
	if len(data) > MacOSDiagFixtureMaxBytes {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident fixture exceeds %d bytes", MacOSDiagFixtureMaxBytes)
	}
	body := strings.ReplaceAll(string(data), "\r\n", "\n")
	startText, err := oneMacOSDiagField(macOSDiagStartRE, body, "Date/Time")
	if err != nil {
		return ResourceEvent{}, err
	}
	endText, err := oneMacOSDiagField(macOSDiagEndRE, body, "End time")
	if err != nil {
		return ResourceEvent{}, err
	}
	start, err := time.Parse(macOSDiagTimestampLayout, startText)
	if err != nil {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident Date/Time: %w", err)
	}
	end, err := time.Parse(macOSDiagTimestampLayout, endText)
	if err != nil {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident End time: %w", err)
	}
	if !end.After(start) {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident interval is not increasing")
	}
	process, err := oneMacOSDiagField(macOSDiagCommandRE, body, "Command")
	if err != nil {
		return ResourceEvent{}, err
	}
	if !strings.EqualFold(process, "fak") {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident Command is unsupported")
	}
	pidText, err := oneMacOSDiagField(macOSDiagPIDRE, body, "PID")
	if err != nil {
		return ResourceEvent{}, err
	}
	pid, err := strconv.Atoi(pidText)
	if err != nil || pid <= 0 {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident PID is invalid")
	}
	classification, err := oneMacOSDiagField(macOSDiagEventRE, body, "Event")
	if err != nil {
		return ResourceEvent{}, err
	}
	action, err := oneMacOSDiagField(macOSDiagActionRE, body, "Action taken")
	if err != nil {
		return ResourceEvent{}, err
	}
	writes := macOSDiagWritesRE.FindAllStringSubmatch(body, -1)
	if len(writes) != 1 {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident Writes field must appear exactly once")
	}
	dirtiedMB, err := parseMacOSDiagDecimal(writes[0][1])
	if err != nil || dirtiedMB <= 0 {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident dirtied MB is invalid")
	}
	durationSeconds, err := strconv.ParseInt(writes[0][2], 10, 64)
	if err != nil || durationSeconds <= 0 {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident duration is invalid")
	}
	minimumInterval := time.Duration(durationSeconds-1) * time.Second
	maximumInterval := time.Duration(durationSeconds+1) * time.Second
	if elapsed := end.Sub(start); elapsed < minimumInterval || elapsed > maximumInterval {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident duration does not match report interval")
	}
	rateMBPerSecond, err := parseMacOSDiagDecimal(writes[0][3])
	if err != nil || rateMBPerSecond <= 0 {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident average rate is invalid")
	}
	footprintText, err := oneMacOSDiagField(macOSDiagFootprintRE, body, "Footprint")
	if err != nil {
		return ResourceEvent{}, err
	}
	footprintMB, err := parseMacOSDiagDecimal(footprintText)
	if err != nil || footprintMB <= 0 {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident footprint is invalid")
	}
	binaryUUID, err := oneMacOSDiagField(macOSDiagBinaryRE, body, "fak binary UUID")
	if err != nil {
		return ResourceEvent{}, err
	}
	binaryUUID = strings.ToUpper(binaryUUID)
	if !validUUID(binaryUUID) {
		return ResourceEvent{}, fmt.Errorf("macOS resource incident fak binary UUID is invalid")
	}
	stackEnd, err := parseMacOSDiagStackEnd(body)
	if err != nil {
		return ResourceEvent{}, err
	}
	sum := sha256.Sum256(data)
	incident := &MacOSResourceIncident{
		IncidentType: MacOSResourceIncidentDiskWrites, ReportStartMS: start.UTC().UnixMilli(),
		ReportEndMS: end.UTC().UnixMilli(), Classification: classification, ActionTaken: action,
		DirtiedMB: dirtiedMB, DurationSeconds: durationSeconds, AverageMBPerSecond: rateMBPerSecond,
		Process: "fak", PID: pid, FootprintMB: footprintMB, BinaryUUID: binaryUUID,
		SampledStackEnd: stackEnd,
		Artifact: ArtifactProvenance{
			Basename: macOSDiagArtifactBasename(sourceName), ByteCount: int64(len(data)),
			SHA256: "sha256:" + hex.EncodeToString(sum[:]),
		},
	}
	return ResourceEvent{
		TimeMS: end.UTC().UnixMilli(), Source: MacOSDiagnosticReportsSource,
		Name: MacOSResourceIncidentEventName, App: "fak", ResourceIncident: incident,
	}, nil
}

func oneMacOSDiagField(re *regexp.Regexp, body, name string) (string, error) {
	matches := re.FindAllStringSubmatch(body, -1)
	if len(matches) != 1 {
		return "", fmt.Errorf("macOS resource incident %s field must appear exactly once", name)
	}
	return strings.TrimSpace(matches[0][1]), nil
}

func parseMacOSDiagDecimal(value string) (float64, error) {
	return strconv.ParseFloat(strings.ReplaceAll(value, ",", ""), 64)
}

func parseMacOSDiagStackEnd(body string) (string, error) {
	const header = "Heaviest stack for the target process:"
	index := strings.Index(body, header)
	if index < 0 {
		return "", fmt.Errorf("macOS resource incident sampled stack is missing")
	}
	var lastFrame string
	for _, line := range strings.Split(body[index+len(header):], "\n") {
		if strings.TrimSpace(line) == "" {
			if lastFrame != "" {
				break
			}
			continue
		}
		match := macOSDiagStackFrameRE.FindStringSubmatch(line)
		if len(match) != 2 {
			if lastFrame != "" {
				break
			}
			continue
		}
		lastFrame = strings.TrimSpace(match[1])
	}
	if !macOSDiagWriteFrameRE.MatchString(lastFrame) {
		return "", fmt.Errorf("macOS resource incident sampled stack does not end in write(2)")
	}
	return "write(2)", nil
}

func macOSDiagArtifactBasename(_ string) string {
	return macOSDiagSanitizedArtifactName
}

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/hostdiag"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/shellprov"
)

const hostdiagDefaultMaxBytes int64 = 16 << 20
const hostdiagJSONFixtureMaxBytes int64 = 1 << 20

func cmdHostdiag(args []string) { os.Exit(runHostdiag(os.Stdout, os.Stderr, args)) }

func runHostdiag(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak hostdiag census|correlate|trend [flags]")
		return 2
	}
	switch args[0] {
	case "census":
		return runHostdiagCensus(stdout, stderr, args[1:])
	case "correlate":
		return runHostdiagCorrelate(stdout, stderr, args[1:])
	case "trend":
		return runHostdiagTrend(stdout, stderr, args[1:])
	default:
		fmt.Fprintln(stderr, "usage: fak hostdiag census|correlate|trend [flags]")
		return 2
	}
}

func runHostdiagTrend(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("hostdiag trend", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultHostdiagLedger(), "mixed-schema JSONL host diagnostics ledger")
	recent := fs.Duration("recent", 7*24*time.Hour, "recent half-open window duration")
	baseline := fs.Duration("baseline", 7*24*time.Hour, "preceding baseline window duration")
	nowText := fs.String("now", "", "window end in RFC3339 (defaults to current time)")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *recent <= 0 || *baseline <= 0 {
		return 2
	}
	now := time.Now().UTC()
	if *nowText != "" {
		parsed, err := time.Parse(time.RFC3339Nano, *nowText)
		if err != nil {
			fmt.Fprintf(stderr, "fak hostdiag trend: --now: %v\n", err)
			return 2
		}
		now = parsed.UTC()
	}
	file, err := os.Open(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak hostdiag trend: ledger: %v\n", err)
		return 1
	}
	defer file.Close()
	trend, err := hostdiag.SummarizeTrend(file, now, *recent, *baseline)
	if err != nil {
		fmt.Fprintf(stderr, "fak hostdiag trend: %v\n", err)
		return 1
	}
	data, err := json.Marshal(trend)
	if err != nil {
		fmt.Fprintf(stderr, "fak hostdiag trend: encode: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(data))
	return 0
}

func runHostdiagCensus(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("hostdiag census", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultHostdiagLedger(), "bounded mixed-schema JSONL ledger")
	regDir := fs.String("reg-dir", "", "guard session registry directory")
	maxBytes := fs.Int64("max-bytes", hostdiagDefaultMaxBytes, "maximum ledger bytes")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *maxBytes <= 0 {
		return 2
	}
	samples, err := gatherHostdiagCensus(resolveSweepRegDir(*regDir), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak hostdiag census: %v\n", err)
		return 1
	}
	for _, sample := range samples {
		if err := appendHostdiagRow(*ledger, sample, *maxBytes); err != nil {
			fmt.Fprintf(stderr, "fak hostdiag census: %v\n", err)
			return 1
		}
		data, _ := json.Marshal(sample)
		fmt.Fprintln(stdout, string(data))
	}
	return 0
}

func runHostdiagCorrelate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("hostdiag correlate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultHostdiagLedger(), "bounded mixed-schema JSONL ledger")
	fixture := fs.String("fixture", "", "normalized resource-event JSON or sanitized macOS resource .diag fixture")
	since := fs.Duration("since", 30*24*time.Hour, "Windows event lookback")
	maxBytes := fs.Int64("max-bytes", hostdiagDefaultMaxBytes, "maximum ledger bytes")
	shellProvenance := fs.String("shell-provenance", "", "privacy-safe fak-owned shell launch receipt JSONL")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *since <= 0 || *maxBytes <= 0 {
		return 2
	}
	var events []hostdiag.ResourceEvent
	var err error
	if *fixture != "" {
		events, err = readHostdiagFixture(*fixture)
	} else {
		events, err = gatherHostdiagEvents(*since)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak hostdiag correlate: events: %v\n", err)
		return 1
	}
	samples, err := hostdiag.LoadCensus(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak hostdiag correlate: census: %v\n", err)
		return 1
	}
	launches, err := loadOwnedShellLaunches(*shellProvenance)
	if err != nil {
		fmt.Fprintf(stderr, "fak hostdiag correlate: shell provenance: %v\n", err)
		return 1
	}
	correlations := make([]hostdiag.Correlation, 0, len(events))
	for _, event := range events {
		if correlation, ok := hostdiag.CorrelateWithOwnedLaunches(event, samples, launches); ok {
			correlations = append(correlations, correlation)
		}
	}
	for _, correlation := range correlations {
		if err := appendHostdiagRow(*ledger, correlation, *maxBytes); err != nil {
			fmt.Fprintf(stderr, "fak hostdiag correlate: %v\n", err)
			return 1
		}
		data, _ := json.Marshal(correlation)
		fmt.Fprintln(stdout, string(data))
	}
	return 0
}

func loadOwnedShellLaunches(path string) ([]hostdiag.OwnedShellLaunch, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer file.Close()

	launches := make([]hostdiag.OwnedShellLaunch, 0)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, shellprov.MaxReceiptLineBytes), shellprov.MaxReceiptLineBytes+1)
	for scanner.Scan() {
		var receipt shellprov.Receipt
		if err := json.Unmarshal(scanner.Bytes(), &receipt); err != nil {
			return nil, fmt.Errorf("decode receipt: %w", err)
		}
		if err := receipt.Validate(); err != nil {
			return nil, fmt.Errorf("validate receipt: %w", err)
		}
		launches = append(launches, hostdiag.OwnedShellLaunch{
			TimestampUTCMS: receipt.TimestampUTCMS, ParentPID: receipt.ParentPID, ChildPID: receipt.ChildPID,
			ChildCreatedUTCMS: receipt.ChildCreatedUTCMS, LaunchID: receipt.LaunchID,
			LaunchClass: string(receipt.LaunchClass), ShellImage: string(receipt.ShellImage),
			ShellEdition: string(receipt.ShellEdition), ShellVersion: receipt.ShellVersion,
			Outcome: string(receipt.Outcome), ErrorClass: string(receipt.ErrorClass),
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return launches, nil
}
func defaultHostdiagLedger() string {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base, _ = os.UserConfigDir()
	}
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "fak", "host", "hostdiag.jsonl")
}

func gatherHostdiagCensus(regDir string, now time.Time) ([]hostdiag.ProcessSample, error) {
	procs, note := procguard.CollectRelations()
	if strings.TrimSpace(note) != "" && len(procs) == 0 {
		return nil, fmt.Errorf("process census: %s", note)
	}
	rows := guardsessions.Load(regDir)
	sessions := make(map[int]string)
	for _, row := range rows {
		if row.EndedAt == "" && row.PID > 0 {
			sessions[row.PID] = row.Handle
		}
	}
	refs := make([]harnessres.ProcRef, 0)
	cmdlines := make(map[int]string)
	for _, process := range procs {
		name := strings.ToLower(filepath.Base(process.Name))
		if name != "fak" && name != "fak.exe" {
			continue
		}
		refs = append(refs, harnessres.ProcRef{PID: process.PID, Name: process.Name, Cmdline: process.Cmdline})
		cmdlines[process.PID] = process.Cmdline
	}
	resources := harnessres.SampleFleet(refs)
	out := make([]hostdiag.ProcessSample, 0, len(resources))
	for _, resource := range resources {
		started, ok := hostdiag.ProcessStartedAt(resource.PID)
		if !ok {
			continue
		}
		executable, hash := currentExecutableIdentity(resource.PID)
		var handles, threads uint32
		for _, process := range procs {
			if process.PID != resource.PID {
				continue
			}
			if process.Handles != nil && *process.Handles > 0 {
				handles = uint32(*process.Handles)
			}
			if process.Threads != nil && *process.Threads > 0 {
				threads = uint32(*process.Threads)
			}
			break
		}
		out = append(out, hostdiag.NewProcessSample(now, resource.PID, started, executable, hash, currentBuildRevision(), hostdiag.ClassifyCommand(cmdlines[resource.PID]), sessions[resource.PID], resource.PrivateBytes, resource.RSSBytes, handles, threads))
	}
	return out, nil
}

func currentExecutableIdentity(pid int) (string, string) {
	path, ok := hostdiag.ProcessImage(pid)
	if !ok {
		return "fak.exe", ""
	}
	file, err := os.Open(path)
	if err != nil {
		return path, ""
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return path, ""
	}
	return path, hex.EncodeToString(hash.Sum(nil))
}

func currentBuildRevision() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	for _, setting := range info.Settings {
		if setting.Key == "vcs.revision" {
			return setting.Value
		}
	}
	return info.Main.Version
}

func readHostdiagFixture(path string) ([]hostdiag.ResourceEvent, error) {
	maxBytes := hostdiagJSONFixtureMaxBytes
	if strings.EqualFold(filepath.Ext(path), ".diag") {
		maxBytes = hostdiag.MacOSDiagFixtureMaxBytes
	}
	data, err := readHostdiagFixtureBytes(path, maxBytes)
	if err != nil {
		return nil, err
	}
	if strings.EqualFold(filepath.Ext(path), ".diag") {
		event, err := hostdiag.ParseMacOSResourceIncident(filepath.Base(path), data)
		if err != nil {
			return nil, fmt.Errorf("parse fixture: %w", err)
		}
		return []hostdiag.ResourceEvent{event}, nil
	}
	var events []hostdiag.ResourceEvent
	if err := json.Unmarshal(data, &events); err != nil {
		return nil, fmt.Errorf("parse fixture: %w", err)
	}
	return events, nil
}

func readHostdiagFixtureBytes(path string, maxBytes int64) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() > maxBytes {
		return nil, fmt.Errorf("fixture exceeds %d bytes", maxBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("fixture exceeds %d bytes", maxBytes)
	}
	return data, nil
}

func appendHostdiagRow(path string, row any, maxBytes int64) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	lockFile, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lockFile.Close()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if err = flock.TryLock(lockFile); err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) || time.Now().After(deadline) {
			return fmt.Errorf("acquire hostdiag ledger lock: %w", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	defer flock.Unlock(lockFile) //nolint:errcheck -- close releases the lock on return
	data, err := json.Marshal(row)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if int64(len(data)) > maxBytes {
		return fmt.Errorf("row exceeds %d-byte ledger bound", maxBytes)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return boundHostdiagLedger(path, maxBytes)
}

func boundHostdiagLedger(path string, maxBytes int64) error {
	info, err := os.Stat(path)
	if err != nil || info.Size() <= maxBytes {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	start := len(data) - int(maxBytes)
	if index := strings.IndexByte(string(data[start:]), '\n'); index >= 0 {
		start += index + 1
	} else {
		return errors.New("newest hostdiag row exceeds ledger bound")
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".hostdiag-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err == nil {
		_, err = tmp.Write(data[start:])
	}
	if err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(tmpPath, path)
}

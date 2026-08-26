// Package shellprov records privacy-safe provenance for PowerShell processes
// launched by fak-owned surfaces. Receipts intentionally carry only bounded
// process and shell metadata; command lines, scripts, paths, environment values,
// and other launch content have no representation in the schema.
package shellprov

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

const (
	// ReceiptSchema is the stable JSONL schema for an owned shell launch.
	ReceiptSchema = "fak.shellprov.receipt.v1"

	// DefaultMaxRows bounds the active receipt history when a caller does not
	// choose a smaller retention window.
	DefaultMaxRows = 4096

	// MaxRows is the largest retention window an appender may request.
	MaxRows = 65536

	// MaxReceiptLineBytes bounds one serialized receipt, excluding its newline.
	MaxReceiptLineBytes = 4096

	maxShellVersionBytes = 48
	lockWait             = 10 * time.Second
	lockPoll             = 10 * time.Millisecond
)

// LaunchClass is the closed set of fak-owned launch purposes.
type LaunchClass string

const (
	LaunchTool   LaunchClass = "tool"
	LaunchHook   LaunchClass = "hook"
	LaunchWorker LaunchClass = "worker"
	LaunchProbe  LaunchClass = "probe"
)

// ShellImage names an executable family, never an executable path.
type ShellImage string

const (
	ShellPwsh       ShellImage = "pwsh"
	ShellPowerShell ShellImage = "powershell"
)

// ShellEdition is PowerShell's bounded edition vocabulary.
type ShellEdition string

const (
	EditionCore    ShellEdition = "core"
	EditionDesktop ShellEdition = "desktop"
)

// Outcome is the observed launch/process outcome at receipt time.
type Outcome string

const (
	OutcomeStarted   Outcome = "started"
	OutcomeSucceeded Outcome = "succeeded"
	OutcomeFailed    Outcome = "failed"
)

// ErrorClass is a content-free failure bucket. ErrorNone is required for
// non-failed outcomes; failed outcomes require one of the other classes.
type ErrorClass string

const (
	ErrorNone         ErrorClass = "none"
	ErrorLaunch       ErrorClass = "launch"
	ErrorExitNonzero  ErrorClass = "exit_nonzero"
	ErrorTimeout      ErrorClass = "timeout"
	ErrorConsoleFault ErrorClass = "console_fault"
	ErrorIO           ErrorClass = "io"
	ErrorUnknown      ErrorClass = "unknown"
)

// Fields contains the caller-supplied, content-free launch facts. New adds the
// schema, UTC millisecond timestamp, and deterministic child identity.
type Fields struct {
	ParentPID         int
	ChildPID          int
	ChildCreatedUTCMS int64
	LaunchClass       LaunchClass
	ShellImage        ShellImage
	ShellEdition      ShellEdition
	ShellVersion      string
	Outcome           Outcome
	ErrorClass        ErrorClass
}

// Receipt is one JSONL row. It deliberately has no generic metadata, message,
// argv, command, script, path, environment, or error-text field.
type Receipt struct {
	Schema            string       `json:"schema"`
	TimestampUTCMS    int64        `json:"timestamp_utc_ms"`
	ParentPID         int          `json:"parent_pid"`
	ChildPID          int          `json:"child_pid"`
	ChildCreatedUTCMS int64        `json:"child_created_utc_ms"`
	LaunchID          string       `json:"launch_id"`
	LaunchClass       LaunchClass  `json:"launch_class"`
	ShellImage        ShellImage   `json:"shell_image"`
	ShellEdition      ShellEdition `json:"shell_edition"`
	ShellVersion      string       `json:"shell_version"`
	Outcome           Outcome      `json:"outcome"`
	ErrorClass        ErrorClass   `json:"error_class"`
}

// New constructs and validates one versioned receipt. Unix milliseconds are UTC
// by definition; converting now to UTC makes that intent explicit at the seam.
func New(now time.Time, fields Fields) (Receipt, error) {
	receipt := Receipt{
		Schema:            ReceiptSchema,
		TimestampUTCMS:    now.UTC().UnixMilli(),
		ParentPID:         fields.ParentPID,
		ChildPID:          fields.ChildPID,
		ChildCreatedUTCMS: fields.ChildCreatedUTCMS,
		LaunchID:          ChildIdentity(fields.ChildPID, fields.ChildCreatedUTCMS),
		LaunchClass:       fields.LaunchClass,
		ShellImage:        fields.ShellImage,
		ShellEdition:      fields.ShellEdition,
		ShellVersion:      fields.ShellVersion,
		Outcome:           fields.Outcome,
		ErrorClass:        fields.ErrorClass,
	}
	if err := receipt.Validate(); err != nil {
		return Receipt{}, err
	}
	return receipt, nil
}

// ChildIdentity returns a deterministic identity over exactly child PID and
// child creation time. Reusing a PID at a later creation time therefore cannot
// alias the earlier process.
func ChildIdentity(childPID int, childCreatedUTCMS int64) string {
	h := sha256.New()
	_, _ = io.WriteString(h, strconv.Itoa(childPID))
	_, _ = h.Write([]byte{0})
	_, _ = io.WriteString(h, strconv.FormatInt(childCreatedUTCMS, 10))
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// Validate enforces the closed receipt schema and its cross-field invariants.
func (r Receipt) Validate() error {
	if r.Schema != ReceiptSchema {
		return errors.New("shellprov: invalid receipt schema")
	}
	if r.TimestampUTCMS <= 0 {
		return errors.New("shellprov: timestamp_utc_ms must be positive")
	}
	if r.ParentPID <= 0 {
		return errors.New("shellprov: parent_pid must be positive")
	}
	if r.ChildPID <= 0 {
		return errors.New("shellprov: child_pid must be positive")
	}
	if r.ChildCreatedUTCMS <= 0 {
		return errors.New("shellprov: child_created_utc_ms must be positive")
	}
	if r.LaunchID != ChildIdentity(r.ChildPID, r.ChildCreatedUTCMS) {
		return errors.New("shellprov: launch_id does not match child identity")
	}
	if !validLaunchClass(r.LaunchClass) {
		return errors.New("shellprov: invalid launch_class (want tool, hook, worker, or probe)")
	}
	if !validShellImage(r.ShellImage) {
		return errors.New("shellprov: invalid shell_image (want pwsh or powershell)")
	}
	if !validShellEdition(r.ShellEdition) {
		return errors.New("shellprov: invalid shell_edition (want core or desktop)")
	}
	if (r.ShellImage == ShellPwsh && r.ShellEdition != EditionCore) ||
		(r.ShellImage == ShellPowerShell && r.ShellEdition != EditionDesktop) {
		return errors.New("shellprov: shell_image and shell_edition do not match")
	}
	if !validShellVersion(r.ShellVersion) {
		return errors.New("shellprov: invalid shell_version")
	}
	if !validOutcome(r.Outcome) {
		return errors.New("shellprov: invalid outcome (want started, succeeded, or failed)")
	}
	if !validErrorClass(r.ErrorClass) {
		return errors.New("shellprov: invalid error_class")
	}
	if r.Outcome == OutcomeFailed && r.ErrorClass == ErrorNone {
		return errors.New("shellprov: failed outcome requires an error_class")
	}
	if r.Outcome != OutcomeFailed && r.ErrorClass != ErrorNone {
		return errors.New("shellprov: non-failed outcome requires error_class none")
	}
	return nil
}

// Append validates receipt, serializes one bounded JSONL row, and retains only
// the newest maxRows valid, complete rows. A sidecar OS lock serializes the whole
// read/retain/rewrite transaction across processes. The replacement file is
// fsynced before rename, so a successful return has durable complete lines.
func Append(path string, receipt Receipt, maxRows int) error {
	if filepath.Clean(path) == "." || path == "" {
		return errors.New("shellprov: receipt path is required")
	}
	if err := receipt.Validate(); err != nil {
		return err
	}
	if maxRows == 0 {
		maxRows = DefaultMaxRows
	}
	if maxRows < 0 {
		return errors.New("shellprov: max rows must be positive")
	}
	if maxRows > MaxRows {
		return fmt.Errorf("shellprov: max rows exceeds %d", MaxRows)
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		return fmt.Errorf("shellprov: encode receipt: %w", err)
	}
	if len(encoded) > MaxReceiptLineBytes {
		return errors.New("shellprov: encoded receipt exceeds line bound")
	}

	return withLedgerLock(path, func() error {
		rows, err := readCompleteRows(path, maxRows-1)
		if err != nil {
			return err
		}
		rows = append(rows, encoded)
		if len(rows) > maxRows {
			rows = rows[len(rows)-maxRows:]
		}
		return replaceRows(path, rows)
	})
}

func validLaunchClass(v LaunchClass) bool {
	switch v {
	case LaunchTool, LaunchHook, LaunchWorker, LaunchProbe:
		return true
	default:
		return false
	}
}

func validShellImage(v ShellImage) bool {
	return v == ShellPwsh || v == ShellPowerShell
}

func validShellEdition(v ShellEdition) bool {
	return v == EditionCore || v == EditionDesktop
}

func validOutcome(v Outcome) bool {
	return v == OutcomeStarted || v == OutcomeSucceeded || v == OutcomeFailed
}

func validErrorClass(v ErrorClass) bool {
	switch v {
	case ErrorNone, ErrorLaunch, ErrorExitNonzero, ErrorTimeout, ErrorConsoleFault, ErrorIO, ErrorUnknown:
		return true
	default:
		return false
	}
}

func validShellVersion(v string) bool {
	if len(v) == 0 || len(v) > maxShellVersionBytes || v[0] < '0' || v[0] > '9' {
		return false
	}
	for i := 0; i < len(v); i++ {
		c := v[i]
		if (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
			(c >= 'a' && c <= 'z') || c == '.' || c == '-' || c == '+' {
			continue
		}
		return false
	}
	return true
}

func withLedgerLock(path string, fn func() error) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("shellprov: create receipt directory: %w", err)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("shellprov: open receipt lock: %w", err)
	}
	defer lock.Close()

	deadline := time.Now().Add(lockWait)
	for {
		err = flock.TryLock(lock)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("shellprov: acquire receipt lock: %w", err)
		}
		if time.Now().After(deadline) {
			return errors.New("shellprov: receipt lock busy")
		}
		time.Sleep(lockPoll)
	}
	defer func() { _ = flock.Unlock(lock) }()
	return fn()
}

// readCompleteRows rejects malformed, wrong-schema, overlong, and trailing
// partial rows. A corrupt tail therefore cannot poison the next append.
func readCompleteRows(path string, keep int) ([][]byte, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("shellprov: open receipt ledger: %w", err)
	}
	defer f.Close()

	reader := bufio.NewReaderSize(f, MaxReceiptLineBytes+1)
	ring := make([][]byte, keep)
	seen := 0
	for {
		line, readErr := reader.ReadSlice('\n')
		if errors.Is(readErr, bufio.ErrBufferFull) {
			for errors.Is(readErr, bufio.ErrBufferFull) {
				_, readErr = reader.ReadSlice('\n')
			}
			if errors.Is(readErr, io.EOF) {
				return orderedRing(ring, seen), nil
			}
			if readErr != nil {
				return nil, fmt.Errorf("shellprov: read receipt ledger: %w", readErr)
			}
			continue
		}
		if errors.Is(readErr, io.EOF) {
			return orderedRing(ring, seen), nil // a non-newline tail is incomplete and is dropped
		}
		if readErr != nil {
			return nil, fmt.Errorf("shellprov: read receipt ledger: %w", readErr)
		}

		line = bytes.TrimSuffix(line, []byte{'\n'})
		line = bytes.TrimSuffix(line, []byte{'\r'})
		if len(line) == 0 || len(line) > MaxReceiptLineBytes {
			continue
		}
		var prior Receipt
		if json.Unmarshal(line, &prior) != nil || prior.Validate() != nil {
			continue
		}
		if keep > 0 {
			ring[seen%keep] = append([]byte(nil), line...)
		}
		seen++
	}
}

func orderedRing(ring [][]byte, seen int) [][]byte {
	if len(ring) == 0 || seen == 0 {
		return nil
	}
	count := seen
	if count > len(ring) {
		count = len(ring)
	}
	start := 0
	if seen > len(ring) {
		start = seen % len(ring)
	}
	rows := make([][]byte, 0, count)
	for i := 0; i < count; i++ {
		rows = append(rows, ring[(start+i)%len(ring)])
	}
	return rows
}

func replaceRows(path string, rows [][]byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".shellprov-*.tmp")
	if err != nil {
		return fmt.Errorf("shellprov: create receipt temp: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	for _, row := range rows {
		if _, err = tmp.Write(row); err == nil {
			_, err = tmp.Write([]byte{'\n'})
		}
		if err != nil {
			_ = tmp.Close()
			return fmt.Errorf("shellprov: write receipt temp: %w", err)
		}
	}
	if err = tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("shellprov: sync receipt temp: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("shellprov: close receipt temp: %w", err)
	}
	if err = os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("shellprov: replace receipt ledger: %w", err)
	}
	return nil
}

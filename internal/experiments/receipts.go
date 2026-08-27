package experiments

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	ReceiptSchema          = "fak-experiment-receipt/1"
	ReceiptLedgerRel       = "experiments/receipts.jsonl"
	MaxReceiptLineBytes    = 1 << 20
	receiptScannerInitSize = 64 << 10
	maxLockMetadataBytes   = 4 << 10
	ledgerLockLease        = 2 * time.Minute
)

type Verdict string
type EvidenceClass string

const (
	VerdictWon          Verdict = "won"
	VerdictLost         Verdict = "lost"
	VerdictInconclusive Verdict = "inconclusive"
	VerdictInvalid      Verdict = "invalid"

	// Receipt evidence is deliberately pre-claim. Promotion remains the job of
	// the existing performance claim gates, not this exploratory ledger.
	EvidenceClassScreening EvidenceClass = "screening"
)

type Metric struct {
	Name           string   `json:"name,omitempty"`
	Unit           string   `json:"unit,omitempty"`
	BaselineValue  *float64 `json:"baseline_value,omitempty"`
	CandidateValue *float64 `json:"candidate_value,omitempty"`
}

type Receipt struct {
	Schema            string        `json:"schema"`
	ID                string        `json:"id"`
	RecordedAt        string        `json:"recorded_at"`
	EvidenceClass     EvidenceClass `json:"evidence_class"`
	Hypothesis        string        `json:"hypothesis"`
	Verdict           Verdict       `json:"verdict"`
	Baseline          string        `json:"baseline,omitempty"`
	Candidate         string        `json:"candidate,omitempty"`
	Metric            Metric        `json:"metric,omitempty"`
	Revision          string        `json:"revision,omitempty"`
	Environment       string        `json:"environment,omitempty"`
	EnvironmentDigest string        `json:"environment_digest,omitempty"`
	ArtifactDigest    string        `json:"artifact_digest,omitempty"`
	Scope             string        `json:"scope"`
	Reason            string        `json:"reason,omitempty"`
	NextAction        string        `json:"next_action"`
	Supersedes        string        `json:"supersedes,omitempty"`
}

type ReceiptIdentity struct {
	Hypothesis        string `json:"hypothesis"`
	Revision          string `json:"revision"`
	Environment       string `json:"environment"`
	EnvironmentDigest string `json:"environment_digest"`
	ArtifactDigest    string `json:"artifact_digest"`
}

type LookupStatus string

const (
	LookupExact            LookupStatus = "exact"
	LookupIdentityMismatch LookupStatus = "identity_mismatch"
	LookupNotFound         LookupStatus = "not_found"
)

type LookupResult struct {
	Status        LookupStatus `json:"status"`
	Receipt       *Receipt     `json:"receipt,omitempty"`
	MeasuredLoss  bool         `json:"measured_loss"`
	ClaimEligible bool         `json:"claim_eligible"`
	Matches       int          `json:"matches"`
}

// LedgerBusyError means another writer owns the receipt ledger sidecar lock.
// Callers should retry later; AppendReceipt never waits or rewrites history.
type LedgerBusyError struct {
	LockPath string
	Owner    string
	Expires  time.Time
}

func (e *LedgerBusyError) Error() string {
	if e.Owner == "" {
		return fmt.Sprintf("receipt ledger busy: lock %s has unreadable owner metadata", e.LockPath)
	}
	return fmt.Sprintf("receipt ledger busy: lock %s owned by %s until %s", e.LockPath, e.Owner, e.Expires.UTC().Format(time.RFC3339Nano))
}

type ledgerLockMetadata struct {
	Owner     string `json:"owner"`
	PID       int    `json:"pid"`
	Host      string `json:"host"`
	CreatedAt string `json:"created_at"`
	ExpiresAt string `json:"expires_at"`
}

type ledgerLock struct {
	path  string
	file  *os.File
	owner string
}

func DigestEnvironment(environment string) string {
	sum := sha256.Sum256([]byte(environment))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func (r Receipt) Validate() error {
	var missing []string
	need := func(name, value string) {
		if strings.TrimSpace(value) == "" {
			missing = append(missing, name)
		}
	}
	if r.Schema != ReceiptSchema {
		return fmt.Errorf("schema must be %q", ReceiptSchema)
	}
	need("id", r.ID)
	need("recorded_at", r.RecordedAt)
	if r.EvidenceClass != EvidenceClassScreening {
		return fmt.Errorf("evidence_class must be %q", EvidenceClassScreening)
	}
	need("hypothesis", r.Hypothesis)
	need("scope", r.Scope)
	need("next_action", r.NextAction)
	need("revision", r.Revision)
	need("environment", r.Environment)
	need("environment_digest", r.EnvironmentDigest)
	need("artifact_digest", r.ArtifactDigest)
	if r.Supersedes != "" && r.Supersedes == r.ID {
		return errors.New("supersedes must name a different receipt")
	}
	switch r.Verdict {
	case VerdictWon, VerdictLost:
		need("baseline", r.Baseline)
		need("candidate", r.Candidate)
		need("metric.name", r.Metric.Name)
		need("metric.unit", r.Metric.Unit)
		if r.Metric.BaselineValue == nil {
			missing = append(missing, "metric.baseline_value")
		}
		if r.Metric.CandidateValue == nil {
			missing = append(missing, "metric.candidate_value")
		}
	case VerdictInconclusive, VerdictInvalid:
		need("reason", r.Reason)
	default:
		return fmt.Errorf("verdict must be one of won, lost, inconclusive, invalid")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required fields: %s", strings.Join(missing, ", "))
	}
	if _, err := time.Parse(time.RFC3339, r.RecordedAt); err != nil {
		return fmt.Errorf("recorded_at must be RFC3339: %w", err)
	}
	if err := validateSHA256Digest("environment_digest", r.EnvironmentDigest); err != nil {
		return err
	}
	if want := DigestEnvironment(r.Environment); r.EnvironmentDigest != want {
		return fmt.Errorf("environment_digest does not match SHA-256 of exact environment string: got %s, want %s", r.EnvironmentDigest, want)
	}
	return validateSHA256Digest("artifact_digest", r.ArtifactDigest)
}

func (q ReceiptIdentity) Validate() error {
	var missing []string
	for name, value := range map[string]string{
		"hypothesis": q.Hypothesis, "revision": q.Revision, "environment": q.Environment,
		"environment_digest": q.EnvironmentDigest, "artifact_digest": q.ArtifactDigest,
	} {
		if strings.TrimSpace(value) == "" {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required identity fields: %s", strings.Join(missing, ", "))
	}
	if err := validateSHA256Digest("environment_digest", q.EnvironmentDigest); err != nil {
		return err
	}
	if want := DigestEnvironment(q.Environment); q.EnvironmentDigest != want {
		return fmt.Errorf("environment_digest does not match SHA-256 of exact environment string: got %s, want %s", q.EnvironmentDigest, want)
	}
	return validateSHA256Digest("artifact_digest", q.ArtifactDigest)
}

func validateSHA256Digest(name, value string) error {
	if len(value) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(value, "sha256:") || value != strings.ToLower(value) {
		return fmt.Errorf("%s must be lowercase sha256: followed by 64 hex characters", name)
	}
	if _, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:")); err != nil {
		return fmt.Errorf("%s must be lowercase sha256: followed by 64 hex characters", name)
	}
	return nil
}

func ParseReceiptLedger(content string) ([]Receipt, error) {
	var receipts []Receipt
	history := newReceiptHistory()
	err := scanReceiptLedger(strings.NewReader(content), func(receipt Receipt) error {
		if err := history.add(receipt); err != nil {
			return err
		}
		receipts = append(receipts, receipt)
		return nil
	})
	return receipts, err
}

func scanReceiptLedger(r io.Reader, visit func(Receipt) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, receiptScannerInitSize), MaxReceiptLineBytes+1)
	line := 0
	for scanner.Scan() {
		line++
		if len(scanner.Bytes()) > MaxReceiptLineBytes {
			return fmt.Errorf("line %d: receipt line exceeds %d bytes", line, MaxReceiptLineBytes)
		}
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		var receipt Receipt
		if err := decodeStrictJSON(scanner.Bytes(), &receipt); err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
		if err := receipt.Validate(); err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
		if err := visit(receipt); err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("receipt line exceeds %d bytes or could not be scanned: %w", MaxReceiptLineBytes, err)
	}
	return nil
}

func decodeStrictJSON(line []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(line))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return errors.New("expected exactly one JSON object")
	}
	return nil
}

func ReadReceiptLedger(path string) ([]Receipt, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var receipts []Receipt
	history := newReceiptHistory()
	err = scanReceiptLedger(f, func(receipt Receipt) error {
		if err := history.add(receipt); err != nil {
			return err
		}
		receipts = append(receipts, receipt)
		return nil
	})
	return receipts, err
}

func AppendReceipt(path string, receipt Receipt) error {
	if err := receipt.Validate(); err != nil {
		return err
	}
	line, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	if len(line) > MaxReceiptLineBytes {
		return fmt.Errorf("receipt line is %d bytes; maximum is %d", len(line), MaxReceiptLineBytes)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	lock, err := acquireLedgerLock(path+".lock", time.Now().UTC())
	if err != nil {
		return err
	}
	released := false
	defer func() {
		if !released {
			_ = lock.release()
		}
	}()
	err = appendReceiptUnderLock(path, line, receipt)
	releaseErr := lock.release()
	released = true
	if err != nil {
		return err
	}
	return releaseErr
}

func appendReceiptUnderLock(path string, line []byte, receipt Receipt) error {
	history := newReceiptHistory()
	existing, err := os.Open(path)
	if err == nil {
		ok, newlineErr := hasTerminalNewline(existing)
		if newlineErr != nil {
			existing.Close()
			return newlineErr
		}
		if !ok {
			existing.Close()
			return errors.New("existing nonempty receipt ledger is missing terminal newline")
		}
		if _, err := existing.Seek(0, io.SeekStart); err != nil {
			existing.Close()
			return err
		}
		err = scanReceiptLedger(existing, history.add)
		closeErr := existing.Close()
		if err != nil {
			return fmt.Errorf("existing receipt ledger is invalid: %w", err)
		}
		if closeErr != nil {
			return closeErr
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if history.seen[receipt.ID] {
		return fmt.Errorf("duplicate receipt id %q", receipt.ID)
	}
	if receipt.Supersedes != "" && !history.active[receipt.Supersedes] {
		return fmt.Errorf("supersedes target %q does not exist or is not active", receipt.Supersedes)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	line = append(line, '\n')
	n, err := f.Write(line)
	if err != nil {
		return err
	}
	if n != len(line) {
		return io.ErrShortWrite
	}
	return f.Sync()
}

func hasTerminalNewline(f *os.File) (bool, error) {
	info, err := f.Stat()
	if err != nil {
		return false, err
	}
	if info.Size() == 0 {
		return true, nil
	}
	var last [1]byte
	if _, err := f.ReadAt(last[:], info.Size()-1); err != nil {
		return false, err
	}
	return last[0] == '\n', nil
}

type receiptHistory struct {
	seen   map[string]bool
	active map[string]bool
}

func newReceiptHistory() *receiptHistory {
	return &receiptHistory{seen: make(map[string]bool), active: make(map[string]bool)}
}

func (h *receiptHistory) add(receipt Receipt) error {
	if h.seen[receipt.ID] {
		return fmt.Errorf("duplicate receipt id %q", receipt.ID)
	}
	if receipt.Supersedes != "" && !h.active[receipt.Supersedes] {
		return fmt.Errorf("receipt %q supersedes target %q that does not exist or is not active", receipt.ID, receipt.Supersedes)
	}
	h.seen[receipt.ID] = true
	h.active[receipt.ID] = true
	if receipt.Supersedes != "" {
		h.active[receipt.Supersedes] = false
	}
	return nil
}

func acquireLedgerLock(path string, now time.Time) (*ledgerLock, error) {
	return acquireLedgerLockWithLease(path, now, ledgerLockLease)
}

func acquireLedgerLockWithLease(path string, now time.Time, lease time.Duration) (*ledgerLock, error) {
	gate, err := os.OpenFile(path+".guard", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	locked, err := tryExclusiveFileLock(gate)
	if err != nil {
		gate.Close()
		return nil, err
	}
	if !locked {
		gate.Close()
		// Do not inspect metadata while another process owns the OS guard. On
		// Windows even a read handle can race the owner's remove-before-unlock;
		// the authoritative result is simply busy until the kernel releases it.
		return nil, &LedgerBusyError{LockPath: path}
	}
	metadata, err := newLedgerLockMetadata(now, lease)
	if err == nil {
		err = publishLedgerLockMetadata(path, metadata)
	}
	if err != nil {
		_ = unlockFile(gate)
		_ = gate.Close()
		return nil, err
	}
	return &ledgerLock{path: path, file: gate, owner: metadata.Owner}, nil
}

func newLedgerLockMetadata(now time.Time, lease time.Duration) (ledgerLockMetadata, error) {
	if lease <= 0 || lease > ledgerLockLease {
		return ledgerLockMetadata{}, fmt.Errorf("ledger lock lease must be positive and at most %s", ledgerLockLease)
	}
	var token [16]byte
	if _, err := rand.Read(token[:]); err != nil {
		return ledgerLockMetadata{}, err
	}
	host, err := os.Hostname()
	if err != nil {
		return ledgerLockMetadata{}, fmt.Errorf("resolve lock owner hostname: %w", err)
	}
	if host == "" {
		return ledgerLockMetadata{}, errors.New("resolve lock owner hostname: empty hostname")
	}
	return ledgerLockMetadata{
		Owner:     hex.EncodeToString(token[:]),
		PID:       os.Getpid(),
		Host:      host,
		CreatedAt: now.UTC().Format(time.RFC3339Nano),
		ExpiresAt: now.Add(lease).UTC().Format(time.RFC3339Nano),
	}, nil
}

func publishLedgerLockMetadata(path string, metadata ledgerLockMetadata) error {
	tempPath := path + ".owner." + metadata.Owner + ".tmp"
	f, err := os.OpenFile(tempPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	removeTemp := true
	defer func() {
		_ = f.Close()
		if removeTemp {
			_ = os.Remove(tempPath)
		}
	}()
	b, err := json.Marshal(metadata)
	if err == nil {
		b = append(b, '\n')
		_, err = f.Write(b)
	}
	if err == nil {
		err = f.Sync()
	}
	if err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	// The OS guard is authoritative. Metadata is fully synced before publication,
	// and no other permitted writer can observe the remove/rename cutover.
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.Rename(tempPath, path); err != nil {
		return err
	}
	removeTemp = false
	return nil
}

type ledgerLockSnapshot struct {
	owner   string
	expires time.Time
}

func readLedgerLockSnapshot(path string, now time.Time) (ledgerLockSnapshot, error) {
	f, err := os.Open(path)
	if err != nil {
		return ledgerLockSnapshot{}, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return ledgerLockSnapshot{}, err
	}
	raw, err := io.ReadAll(io.LimitReader(f, maxLockMetadataBytes+1))
	if err != nil {
		return ledgerLockSnapshot{}, err
	}
	snapshot := ledgerLockSnapshot{}
	var metadata ledgerLockMetadata
	if len(raw) <= maxLockMetadataBytes {
		decodeErr := decodeStrictJSON(raw, &metadata)
		created, createdErr := time.Parse(time.RFC3339Nano, metadata.CreatedAt)
		expires, expiresErr := time.Parse(time.RFC3339Nano, metadata.ExpiresAt)
		ownerBytes, ownerErr := hex.DecodeString(metadata.Owner)
		const clockTolerance = 5 * time.Second
		mtimeDelta := info.ModTime().Sub(created)
		if mtimeDelta < 0 {
			mtimeDelta = -mtimeDelta
		}
		validLease := expiresErr == nil && createdErr == nil && expires.After(created) && expires.Sub(created) <= ledgerLockLease
		validClock := createdErr == nil && !created.After(now.Add(clockTolerance)) && mtimeDelta <= clockTolerance
		if decodeErr == nil && ownerErr == nil && len(ownerBytes) == 16 && metadata.PID > 0 && metadata.Host != "" && validLease && validClock {
			snapshot.owner = metadata.Owner
			snapshot.expires = expires
		}
	}
	return snapshot, nil
}

func (l *ledgerLock) release() error {
	if l.file == nil {
		return nil
	}
	// Metadata removal happens while the OS lock is still held. A successor
	// cannot publish until unlock, so there is no check-then-remove transition.
	current, metadataErr := readLedgerLockSnapshot(l.path, time.Now().UTC())
	if metadataErr == nil && current.owner == l.owner {
		metadataErr = os.Remove(l.path)
	} else if metadataErr == nil {
		metadataErr = fmt.Errorf("receipt ledger lock owner changed from %s to %s; refusing metadata removal", l.owner, current.owner)
	}
	unlockErr := unlockFile(l.file)
	closeErr := l.file.Close()
	l.file = nil
	if metadataErr != nil {
		return metadataErr
	}
	if unlockErr != nil {
		return unlockErr
	}
	return closeErr
}

func LookupReceipt(receipts []Receipt, query ReceiptIdentity) LookupResult {
	latestByID := make(map[string]int, len(receipts))
	for i := range receipts {
		latestByID[receipts[i].ID] = i
	}
	superseded := make(map[string]bool)
	for i, receipt := range receipts {
		if latestByID[receipt.ID] != i || receipt.Supersedes == "" {
			continue
		}
		superseded[receipt.Supersedes] = true
	}

	result := LookupResult{Status: LookupNotFound}
	var exact *Receipt
	for i := range receipts {
		receipt := &receipts[i]
		if latestByID[receipt.ID] != i || superseded[receipt.ID] || receipt.Hypothesis != query.Hypothesis {
			continue
		}
		result.Matches++
		if receipt.Revision == query.Revision &&
			receipt.Environment == query.Environment &&
			receipt.EnvironmentDigest == query.EnvironmentDigest &&
			receipt.ArtifactDigest == query.ArtifactDigest {
			exact = receipt
		}
	}
	if exact != nil {
		copy := *exact
		result.Status = LookupExact
		result.Receipt = &copy
		result.MeasuredLoss = copy.Verdict == VerdictLost
		return result
	}
	if result.Matches > 0 {
		result.Status = LookupIdentityMismatch
	}
	return result
}

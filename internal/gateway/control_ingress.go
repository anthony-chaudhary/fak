package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	durableControlDirectivePath  = "/v1/fak/control/directives"
	defaultControlJournalBytes   = int64(16 << 20)
	defaultControlPending        = 128
	maxControlJournalBytes       = int64(64 << 20)
	maxControlPending            = 4096
	controlIngressMaxBodyBytes   = int64(1 << 20)
	controlIngressMaxStringBytes = 1024
	controlDeliveryRetryDelay    = 25 * time.Millisecond
)

var errControlJournalFull = errors.New("control journal is full")

var errControlJournalOwned = errors.New("control ingress: journal already has a writer")

var errControlJournalReplaced = errors.New("control ingress: journal path no longer names the owned file")

// ControlDirective is a durable request for an operator-side control action.
// This provisional transport records the directive but does not confer human
// authority or execution priority on its caller.
type ControlDirective struct {
	ID         string          `json:"id"`
	Target     string          `json:"target"`
	Action     string          `json:"action"`
	Generation uint64          `json:"generation"`
	Payload    json.RawMessage `json:"payload,omitempty"`
}

// ControlReceipt reports the latest state of a directive. State "unknown" with
// reason "journal_indeterminate" means an append or sync failed after the write
// began; callers should retry the same ID and digest or read it back after the
// ingress restarts and recovers the journal.
type ControlReceipt struct {
	ID         string    `json:"id"`
	Target     string    `json:"target"`
	Action     string    `json:"action"`
	State      string    `json:"state"`
	Digest     string    `json:"digest,omitempty"`
	Reason     string    `json:"reason,omitempty"`
	Generation uint64    `json:"generation"`
	Sequence   uint64    `json:"sequence"`
	AcceptedAt time.Time `json:"accepted_at,omitempty"`
}

// DurableControlIngressOptions bounds the independent durable control store
// and supplies its asynchronous delivery callback.
type DurableControlIngressOptions struct {
	JournalPath     string
	MaxJournalBytes int64
	MaxPending      int
	Deliver         func(context.Context, ControlDirective) error
}

type controlIngressJournalFile interface {
	Write([]byte) (int, error)
	Sync() error
	Close() error
}

var openControlIngressJournal = func(path string, flag int, perm fs.FileMode) (controlIngressJournalFile, error) {
	return os.OpenFile(path, flag, perm)
}

type controlJournalRecord struct {
	Kind      string           `json:"kind"`
	Directive ControlDirective `json:"directive"`
	Receipt   ControlReceipt   `json:"receipt"`
	Checksum  string           `json:"checksum"`
}

type controlJournalOwnership struct {
	file    *os.File
	info    fs.FileInfo
	release func() error
}

func (o *controlJournalOwnership) Close() error {
	if o == nil {
		return nil
	}
	var releaseErr error
	if o.release != nil {
		releaseErr = o.release()
	}
	if o.file != nil {
		if err := o.file.Close(); releaseErr == nil {
			releaseErr = err
		}
	}
	return releaseErr
}

// DurableControlIngress owns a bounded index and append-only journal. Its
// acknowledgement path never calls Deliver; a dedicated worker performs
// at-least-once delivery after the accepted record has been synced.
type DurableControlIngress struct {
	mu sync.Mutex

	journal         controlIngressJournalFile
	ownership       *controlJournalOwnership
	journalPath     string
	journalBytes    int64
	maxJournalBytes int64
	maxPending      int
	deliver         func(context.Context, ControlDirective) error

	receipts    map[string]ControlReceipt
	directives  map[string]ControlDirective
	generations map[string]uint64
	sequence    uint64
	pending     int
	poisoned    bool
	closed      bool

	queue     chan string
	done      chan struct{}
	closeOnce sync.Once
	closeErr  error
}

// OpenDurableControlIngress opens and validates the complete journal before
// making the ingress available. Any malformed, corrupt, or torn frame fails
// closed; recovery never silently truncates accepted control intent.
func OpenDurableControlIngress(opts DurableControlIngressOptions) (*DurableControlIngress, error) {
	configuredPath := strings.TrimSpace(opts.JournalPath)
	if configuredPath == "" {
		return nil, errors.New("control ingress: journal path is required")
	}
	path, err := filepath.Abs(configuredPath)
	if err != nil {
		return nil, fmt.Errorf("control ingress: resolve journal path: %w", err)
	}
	resolvedPath, err := filepath.EvalSymlinks(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, errors.New("control ingress: journal must be provisioned before open")
	}
	if err != nil {
		return nil, fmt.Errorf("control ingress: resolve journal path: %w", err)
	}
	maxJournalBytes := opts.MaxJournalBytes
	if maxJournalBytes == 0 {
		maxJournalBytes = defaultControlJournalBytes
	}
	if maxJournalBytes < 2 {
		return nil, errors.New("control ingress: max journal bytes is too small")
	}
	if maxJournalBytes > maxControlJournalBytes {
		return nil, fmt.Errorf("control ingress: max journal bytes exceeds hard limit %d", maxControlJournalBytes)
	}
	maxPending := opts.MaxPending
	if maxPending == 0 {
		maxPending = defaultControlPending
	}
	if maxPending < 0 {
		return nil, errors.New("control ingress: max pending must be positive")
	}
	if maxPending > maxControlPending {
		return nil, fmt.Errorf("control ingress: max pending exceeds hard limit %d", maxControlPending)
	}

	ownership, err := acquireControlJournalOwnership(path, resolvedPath)
	if err != nil {
		return nil, err
	}
	releaseOwnership := true
	defer func() {
		if releaseOwnership {
			_ = ownership.Close()
		}
	}()

	if err := verifyControlJournalPath(path, ownership.info); err != nil {
		return nil, err
	}
	data, recoveredFile, err := readControlJournal(ownership, maxJournalBytes)
	if err != nil {
		return nil, err
	}

	d := &DurableControlIngress{
		journalBytes:    int64(len(data)),
		maxJournalBytes: maxJournalBytes,
		maxPending:      maxPending,
		deliver:         opts.Deliver,
		receipts:        make(map[string]ControlReceipt),
		directives:      make(map[string]ControlDirective),
		generations:     make(map[string]uint64),
		queue:           make(chan string, maxPending),
		done:            make(chan struct{}),
		ownership:       ownership,
		journalPath:     path,
	}
	if err := d.recover(data); err != nil {
		return nil, fmt.Errorf("control ingress: recover journal: %w", err)
	}
	if d.pending > maxPending {
		return nil, fmt.Errorf("control ingress: recovered pending directives exceed bound (%d > %d)", d.pending, maxPending)
	}

	journal, err := openControlIngressJournal(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("control ingress: open journal: %w", err)
	}
	d.journal = journal
	if statter, ok := journal.(interface{ Stat() (fs.FileInfo, error) }); ok {
		appendFile, err := statter.Stat()
		if err != nil {
			_ = journal.Close()
			return nil, fmt.Errorf("control ingress: stat append journal: %w", err)
		}
		if !os.SameFile(recoveredFile, appendFile) {
			_ = journal.Close()
			return nil, errors.New("control ingress: journal changed between recovery and append open")
		}
	}
	if chmodder, ok := journal.(interface{ Chmod(fs.FileMode) error }); ok {
		if err := chmodder.Chmod(0o600); err != nil {
			_ = journal.Close()
			return nil, fmt.Errorf("control ingress: restrict journal permissions: %w", err)
		}
	}
	// Flush the exact recovered file before replaying pending delivery or
	// accepting a new directive. A prior process may have returned an unknown
	// receipt after Sync failed even though its complete frame remained cached.
	if err := journal.Sync(); err != nil {
		_ = journal.Close()
		return nil, fmt.Errorf("control ingress: sync recovered journal: %w", err)
	}

	type pendingDirective struct {
		id       string
		sequence uint64
	}
	pending := make([]pendingDirective, 0, d.pending)
	for id, receipt := range d.receipts {
		if receipt.State == "accepted" {
			pending = append(pending, pendingDirective{id: id, sequence: receipt.Sequence})
		}
	}
	sort.Slice(pending, func(i, j int) bool { return pending[i].sequence < pending[j].sequence })
	for _, directive := range pending {
		d.queue <- directive.id
	}
	if d.deliver != nil {
		go d.deliveryLoop()
	}
	releaseOwnership = false
	return d, nil
}

func readControlJournal(ownership *controlJournalOwnership, maxBytes int64) ([]byte, fs.FileInfo, error) {
	data, err := io.ReadAll(io.NewSectionReader(ownership.file, 0, maxBytes+1))
	if err != nil {
		return nil, nil, fmt.Errorf("control ingress: read journal: %w", err)
	}
	if int64(len(data)) > maxBytes {
		return nil, nil, errors.New("control ingress: journal exceeds configured bound")
	}
	return data, ownership.info, nil
}

// Close stops new acceptance and closes the journal. A delivery callback that
// ignores context cannot delay Close; it has no access to the journal itself.
func (d *DurableControlIngress) Close() error {
	if d == nil {
		return nil
	}
	d.closeOnce.Do(func() {
		d.mu.Lock()
		defer d.mu.Unlock()
		d.closed = true
		close(d.done)
		if d.journal != nil {
			d.closeErr = d.journal.Close()
		}
		if err := d.ownership.Close(); d.closeErr == nil {
			d.closeErr = err
		}
	})
	return d.closeErr
}

// ServeHTTP handles the provisional collection POST and item GET directly.
// Authentication and canonical route registration are owned by the later
// gateway wiring leaf; possession of a bearer token is not human proof.
func (d *DurableControlIngress) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == durableControlDirectivePath {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		d.handleSubmit(w, r)
		return
	}
	prefix := durableControlDirectivePath + "/"
	if !strings.HasPrefix(r.URL.Path, prefix) || strings.Contains(strings.TrimPrefix(r.URL.Path, prefix), "/") {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	id, err := url.PathUnescape(strings.TrimPrefix(r.URL.EscapedPath(), prefix))
	if err != nil || id == "" {
		http.NotFound(w, r)
		return
	}
	d.handleRead(w, id)
}

func (d *DurableControlIngress) handleSubmit(w http.ResponseWriter, r *http.Request) {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, controlIngressMaxBodyBytes))
	dec.DisallowUnknownFields()
	var directive ControlDirective
	if err := dec.Decode(&directive); err != nil {
		writeControlReceipt(w, http.StatusBadRequest, rejectedControlReceipt(directive, "denied", "invalid_body"))
		return
	}
	if err := requireJSONEOF(dec); err != nil {
		writeControlReceipt(w, http.StatusBadRequest, rejectedControlReceipt(directive, "denied", "invalid_body"))
		return
	}
	canonical, err := normalizeControlDirective(directive)
	if err != nil {
		writeControlReceipt(w, http.StatusBadRequest, rejectedControlReceipt(directive, "denied", "invalid_directive"))
		return
	}
	receipt, status := d.accept(canonical)
	writeControlReceipt(w, status, receipt)
}

func (d *DurableControlIngress) handleRead(w http.ResponseWriter, id string) {
	d.mu.Lock()
	receipt, ok := d.receipts[id]
	d.mu.Unlock()
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	writeControlReceipt(w, http.StatusOK, receipt)
}

func (d *DurableControlIngress) accept(directive ControlDirective) (ControlReceipt, int) {
	digest, err := controlDirectiveDigest(directive)
	if err != nil {
		return rejectedControlReceipt(directive, "denied", "invalid_directive"), http.StatusBadRequest
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return rejectedControlReceipt(directive, "unavailable", "journal_unavailable"), http.StatusServiceUnavailable
	}
	if err := d.verifyJournalPathLocked(); err != nil {
		d.poisoned = true
		return rejectedControlReceipt(directive, "unavailable", "journal_replaced"), http.StatusServiceUnavailable
	}
	if existing, ok := d.receipts[directive.ID]; ok {
		if existing.Digest != digest {
			return rejectedControlReceipt(directive, "denied", "id_conflict"), http.StatusConflict
		}
		if existing.State == "unknown" {
			return existing, http.StatusServiceUnavailable
		}
		if d.poisoned {
			return rejectedControlReceipt(directive, "unavailable", "journal_unavailable"), http.StatusServiceUnavailable
		}
		return existing, http.StatusAccepted
	}
	if d.poisoned {
		return rejectedControlReceipt(directive, "unavailable", "journal_unavailable"), http.StatusServiceUnavailable
	}
	currentGeneration := d.generations[directive.Target]
	if currentGeneration == ^uint64(0) {
		return rejectedControlReceipt(directive, "denied", "generation_conflict"), http.StatusConflict
	}
	wantGeneration := currentGeneration + 1
	if directive.Generation != wantGeneration {
		return rejectedControlReceipt(directive, "denied", "generation_conflict"), http.StatusConflict
	}
	if d.pending >= d.maxPending {
		return rejectedControlReceipt(directive, "unavailable", "pending_full"), http.StatusServiceUnavailable
	}

	receipt := ControlReceipt{
		ID:         directive.ID,
		Target:     directive.Target,
		Action:     directive.Action,
		State:      "accepted",
		Digest:     digest,
		Generation: directive.Generation,
		Sequence:   d.sequence + 1,
		AcceptedAt: time.Now().UTC(),
	}
	record := controlJournalRecord{Kind: "accepted", Directive: directive, Receipt: receipt}
	if err := d.appendRecordLocked(record); err != nil {
		if errors.Is(err, errControlJournalFull) {
			return rejectedControlReceipt(directive, "unavailable", "journal_full"), http.StatusServiceUnavailable
		}
		d.poisoned = true
		unknown := receipt
		unknown.State = "unknown"
		if errors.Is(err, errControlJournalReplaced) {
			unknown.Reason = "journal_replaced"
		} else {
			unknown.Reason = "journal_indeterminate"
		}
		// Retain the indeterminate receipt so GET and a same-ID retry report
		// uncertainty until restart recovery establishes whether this exact
		// digest reached durable storage.
		d.receipts[directive.ID] = unknown
		return unknown, http.StatusServiceUnavailable
	}

	d.sequence = receipt.Sequence
	d.receipts[directive.ID] = receipt
	d.directives[directive.ID] = cloneControlDirective(directive)
	d.generations[directive.Target] = directive.Generation
	d.pending++
	d.queue <- directive.ID
	return receipt, http.StatusAccepted
}

func (d *DurableControlIngress) deliveryLoop() {
	for {
		select {
		case <-d.done:
			return
		case id := <-d.queue:
			if !d.deliverUntilAcknowledged(id) {
				return
			}
		}
	}
}

func (d *DurableControlIngress) deliverUntilAcknowledged(id string) bool {
	for {
		d.mu.Lock()
		directive, ok := d.directives[id]
		receipt := d.receipts[id]
		closed := d.closed
		d.mu.Unlock()
		if closed || !ok || receipt.State != "accepted" {
			return !closed
		}

		ctx, cancel := context.WithCancel(context.Background())
		result := make(chan error, 1)
		go func(delivery ControlDirective) {
			result <- d.callDeliver(ctx, cloneControlDirective(delivery))
		}(directive)

		var err error
		select {
		case <-d.done:
			cancel()
			return false
		case err = <-result:
			cancel()
		}
		if err == nil {
			d.recordDelivered(id)
			return true
		}

		timer := time.NewTimer(controlDeliveryRetryDelay)
		select {
		case <-d.done:
			if !timer.Stop() {
				<-timer.C
			}
			return false
		case <-timer.C:
		}
	}
}

func (d *DurableControlIngress) callDeliver(ctx context.Context, directive ControlDirective) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("control delivery panic: %v", recovered)
		}
	}()
	return d.deliver(ctx, directive)
}

func (d *DurableControlIngress) recordDelivered(id string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed || d.poisoned {
		return
	}
	receipt, ok := d.receipts[id]
	if !ok || receipt.State != "accepted" {
		return
	}
	receipt.State = "delivered"
	receipt.Sequence = d.sequence + 1
	record := controlJournalRecord{Kind: "delivered", Directive: d.directives[id], Receipt: receipt}
	if err := d.appendRecordLocked(record); err != nil {
		if errors.Is(err, errControlJournalFull) {
			return
		}
		d.poisoned = true
		receipt.State = "unknown"
		if errors.Is(err, errControlJournalReplaced) {
			receipt.Reason = "journal_replaced"
		} else {
			receipt.Reason = "journal_indeterminate"
		}
		d.receipts[id] = receipt
		return
	}
	d.sequence = receipt.Sequence
	d.receipts[id] = receipt
	if d.pending > 0 {
		d.pending--
	}
}

func (d *DurableControlIngress) appendRecordLocked(record controlJournalRecord) error {
	if err := d.verifyJournalPathLocked(); err != nil {
		return err
	}
	record.Checksum = ""
	checksumPayload, err := json.Marshal(record)
	if err != nil {
		return err
	}
	checksum := sha256.Sum256(checksumPayload)
	record.Checksum = hex.EncodeToString(checksum[:])
	payload, err := json.Marshal(record)
	if err != nil {
		return err
	}
	payload = append(payload, '\n')
	frameBytes := int64(len(payload))
	if frameBytes > d.maxJournalBytes-d.journalBytes {
		return errControlJournalFull
	}
	if err := writeControlJournalFrame(d.journal, payload); err != nil {
		return err
	}
	if err := d.journal.Sync(); err != nil {
		return err
	}
	if err := d.verifyJournalPathLocked(); err != nil {
		return err
	}
	d.journalBytes += frameBytes
	return nil
}

func (d *DurableControlIngress) verifyJournalPathLocked() error {
	return verifyControlJournalPath(d.journalPath, d.ownership.info)
}

func verifyControlJournalPath(path string, owned fs.FileInfo) error {
	current, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%w: %v", errControlJournalReplaced, err)
	}
	if !os.SameFile(owned, current) {
		return errControlJournalReplaced
	}
	return nil
}

func writeControlJournalFrame(w io.Writer, frame []byte) error {
	for len(frame) > 0 {
		n, err := w.Write(frame)
		if n > 0 {
			frame = frame[n:]
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func (d *DurableControlIngress) recover(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if data[len(data)-1] != '\n' {
		return errors.New("torn final journal record")
	}
	lines := bytes.Split(data[:len(data)-1], []byte{'\n'})
	for _, payload := range lines {
		if len(payload) == 0 {
			return errors.New("empty journal record")
		}
		var record controlJournalRecord
		if err := unmarshalControlRecord(payload, &record); err != nil {
			return err
		}
		checksum := record.Checksum
		record.Checksum = ""
		checksumPayload, err := json.Marshal(record)
		if err != nil {
			return err
		}
		sum := sha256.Sum256(checksumPayload)
		if checksum == "" || checksum != hex.EncodeToString(sum[:]) {
			return errors.New("journal record checksum mismatch")
		}
		record.Checksum = checksum
		if err := d.applyRecoveredRecord(record); err != nil {
			return err
		}
	}
	return nil
}

func (d *DurableControlIngress) applyRecoveredRecord(record controlJournalRecord) error {
	directive, err := normalizeControlDirective(record.Directive)
	if err != nil {
		return fmt.Errorf("invalid %s directive: %w", record.Kind, err)
	}
	digest, err := controlDirectiveDigest(directive)
	if err != nil {
		return err
	}
	if record.Receipt.ID != directive.ID || record.Receipt.Target != directive.Target ||
		record.Receipt.Action != directive.Action || record.Receipt.Generation != directive.Generation ||
		record.Receipt.Digest != digest || record.Receipt.AcceptedAt.IsZero() ||
		record.Receipt.Sequence != d.sequence+1 {
		return errors.New("journal receipt invariant failed")
	}

	switch record.Kind {
	case "accepted":
		if record.Receipt.State != "accepted" || record.Receipt.Reason != "" {
			return errors.New("invalid accepted receipt")
		}
		if _, exists := d.receipts[directive.ID]; exists {
			return errors.New("duplicate accepted directive id")
		}
		if directive.Generation != d.generations[directive.Target]+1 {
			return errors.New("non-monotonic recovered generation")
		}
		d.receipts[directive.ID] = record.Receipt
		d.directives[directive.ID] = cloneControlDirective(directive)
		d.generations[directive.Target] = directive.Generation
		d.pending++
	case "delivered":
		prior, exists := d.receipts[directive.ID]
		if !exists || prior.State != "accepted" || record.Receipt.State != "delivered" ||
			!prior.AcceptedAt.Equal(record.Receipt.AcceptedAt) || prior.Digest != record.Receipt.Digest {
			return errors.New("invalid delivered transition")
		}
		d.receipts[directive.ID] = record.Receipt
		if d.pending <= 0 {
			return errors.New("invalid pending count")
		}
		d.pending--
	default:
		return errors.New("unknown journal record kind")
	}
	d.sequence = record.Receipt.Sequence
	return nil
}

func normalizeControlDirective(directive ControlDirective) (ControlDirective, error) {
	if directive.ID == "" || directive.Target == "" || directive.Action == "" {
		return ControlDirective{}, errors.New("id, target, and action are required")
	}
	if len(directive.ID) > controlIngressMaxStringBytes || len(directive.Target) > controlIngressMaxStringBytes || len(directive.Action) > controlIngressMaxStringBytes {
		return ControlDirective{}, errors.New("directive string exceeds bound")
	}
	if strings.TrimSpace(directive.ID) != directive.ID || strings.TrimSpace(directive.Target) != directive.Target || strings.TrimSpace(directive.Action) != directive.Action {
		return ControlDirective{}, errors.New("directive strings must not have surrounding whitespace")
	}
	if !isControlDirectiveID(directive.ID) {
		return ControlDirective{}, errors.New("directive id must be a URL-safe path segment")
	}
	if !isControlDirectiveAction(directive.Action) {
		return ControlDirective{}, errors.New("unknown control directive action")
	}
	payload := directive.Payload
	if len(bytes.TrimSpace(payload)) == 0 {
		payload = json.RawMessage("null")
	}
	canonical, err := canonicalControlJSON(payload)
	if err != nil {
		return ControlDirective{}, fmt.Errorf("invalid payload: %w", err)
	}
	directive.Payload = canonical
	return directive, nil
}

func isControlDirectiveID(id string) bool {
	if id == "." || id == ".." {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
			continue
		}
		switch c {
		case '-', '_', '.', '~':
			continue
		default:
			return false
		}
	}
	return true
}

func isControlDirectiveAction(action string) bool {
	switch action {
	case "cancel", "pause", "resume", "steer", "reprioritize", "redirect", "stop":
		return true
	default:
		return false
	}
}

func canonicalControlJSON(raw []byte) (json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var value any
	if err := dec.Decode(&value); err != nil {
		return nil, err
	}
	if err := requireJSONEOF(dec); err != nil {
		return nil, err
	}
	canonical, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(canonical), nil
}

func controlDirectiveDigest(directive ControlDirective) (string, error) {
	payload, err := json.Marshal(directive)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:]), nil
}

func cloneControlDirective(directive ControlDirective) ControlDirective {
	directive.Payload = append(json.RawMessage(nil), directive.Payload...)
	return directive
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	err := dec.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("multiple JSON values")
	}
	return err
}

func unmarshalControlRecord(payload []byte, dst *controlJournalRecord) error {
	dec := json.NewDecoder(bytes.NewReader(payload))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("invalid journal record: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return fmt.Errorf("invalid journal record: %w", err)
	}
	return nil
}

func rejectedControlReceipt(directive ControlDirective, state, reason string) ControlReceipt {
	return ControlReceipt{
		ID:         directive.ID,
		Target:     directive.Target,
		Action:     directive.Action,
		State:      state,
		Reason:     reason,
		Generation: directive.Generation,
	}
}

func writeControlReceipt(w http.ResponseWriter, status int, receipt ControlReceipt) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(receipt)
}

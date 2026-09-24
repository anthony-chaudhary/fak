package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

const testControlDirectivePath = "/v1/fak/control/directives"

func openTestControlIngress(t *testing.T, opts DurableControlIngressOptions) *DurableControlIngress {
	t.Helper()
	if opts.JournalPath == "" {
		opts.JournalPath = filepath.Join(t.TempDir(), "control.jsonl")
	}
	if _, err := os.Stat(opts.JournalPath); errors.Is(err, fs.ErrNotExist) {
		f, createErr := os.OpenFile(opts.JournalPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if createErr != nil {
			t.Fatalf("provision test journal: %v", createErr)
		}
		if err := f.Sync(); err != nil {
			t.Fatalf("sync test journal: %v", err)
		}
		if err := f.Close(); err != nil {
			t.Fatalf("close test journal: %v", err)
		}
	} else if err != nil {
		t.Fatalf("stat test journal: %v", err)
	}
	if opts.MaxJournalBytes == 0 {
		opts.MaxJournalBytes = 1 << 20
	}
	if opts.MaxPending == 0 {
		opts.MaxPending = 8
	}
	ingress, err := OpenDurableControlIngress(opts)
	if err != nil {
		t.Fatalf("OpenDurableControlIngress: %v", err)
	}
	t.Cleanup(func() { _ = ingress.Close() })
	return ingress
}

func TestDurableControlIngressRequiresProvisionedJournal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing-control.jsonl")
	if ingress, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path}); err == nil {
		_ = ingress.Close()
		t.Fatal("ingress created a journal without a durable directory-entry witness")
	}
}

func postTestControlDirective(t *testing.T, ingress http.Handler, directive ControlDirective) (int, ControlReceipt, time.Duration) {
	t.Helper()
	body, err := json.Marshal(directive)
	if err != nil {
		t.Fatalf("marshal directive: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, testControlDirectivePath, bytes.NewReader(body))
	w := httptest.NewRecorder()
	started := time.Now()
	ingress.ServeHTTP(w, req)
	elapsed := time.Since(started)
	var receipt ControlReceipt
	if w.Body.Len() > 0 {
		if err := json.Unmarshal(w.Body.Bytes(), &receipt); err != nil {
			t.Fatalf("decode POST response (status %d, body %q): %v", w.Code, w.Body.String(), err)
		}
	}
	return w.Code, receipt, elapsed
}

func getTestControlReceipt(t *testing.T, ingress http.Handler, id string) (int, ControlReceipt) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, testControlDirectivePath+"/"+id, nil)
	w := httptest.NewRecorder()
	ingress.ServeHTTP(w, req)
	var receipt ControlReceipt
	if w.Code == http.StatusNotFound {
		return w.Code, receipt
	}
	if w.Body.Len() > 0 {
		if err := json.Unmarshal(w.Body.Bytes(), &receipt); err != nil {
			t.Fatalf("decode GET response (status %d, body %q): %v", w.Code, w.Body.String(), err)
		}
	}
	return w.Code, receipt
}

func TestDurableControlIngressPersistsAcceptedReceiptBeforeAsyncDelivery(t *testing.T) {
	journalPath := filepath.Join(t.TempDir(), "control.jsonl")
	deliveryStarted := make(chan struct{}, 1)
	releaseDelivery := make(chan struct{})
	defer close(releaseDelivery)
	ingress := openTestControlIngress(t, DurableControlIngressOptions{
		JournalPath: journalPath,
		MaxPending:  1,
		Deliver: func(context.Context, ControlDirective) error {
			deliveryStarted <- struct{}{}
			<-releaseDelivery
			return nil
		},
	})

	directive := ControlDirective{
		ID: "stop-1", Target: "mission-7", Generation: 1, Action: "cancel",
		Payload: json.RawMessage(`{"reason":"operator"}`),
	}
	status, accepted, elapsed := postTestControlDirective(t, ingress, directive)
	if status != http.StatusAccepted {
		t.Fatalf("POST status = %d, want %d", status, http.StatusAccepted)
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("POST waited %s for delivery; acceptance must not call Deliver synchronously", elapsed)
	}
	if accepted.ID != directive.ID || accepted.Target != directive.Target || accepted.Action != directive.Action || accepted.Generation != directive.Generation {
		t.Fatalf("accepted receipt = %+v, want directive identity %+v", accepted, directive)
	}
	if accepted.State != "accepted" || accepted.Sequence == 0 || accepted.Digest == "" || accepted.AcceptedAt.IsZero() {
		t.Fatalf("accepted receipt lacks durable identity: %+v", accepted)
	}
	if accepted.State == "quiesced" || accepted.State == "applied" {
		t.Fatalf("acceptance falsely claimed terminal state %q", accepted.State)
	}
	select {
	case <-deliveryStarted:
	case <-time.After(time.Second):
		t.Fatal("asynchronous delivery did not start")
	}

	if err := ingress.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	reopened, err := OpenDurableControlIngress(DurableControlIngressOptions{
		JournalPath: journalPath,
		MaxPending:  1,
	})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer reopened.Close()
	getStatus, restored := getTestControlReceipt(t, reopened, directive.ID)
	if getStatus != http.StatusOK {
		t.Fatalf("GET after reopen status = %d, want %d", getStatus, http.StatusOK)
	}
	if restored.ID != accepted.ID || restored.Sequence != accepted.Sequence || restored.Digest != accepted.Digest || !restored.AcceptedAt.Equal(accepted.AcceptedAt) {
		t.Fatalf("restored receipt = %+v, want durable identity %+v", restored, accepted)
	}
}

func TestDurableControlIngressRestartDeliversPendingInSequenceOrder(t *testing.T) {
	journalPath := filepath.Join(t.TempDir(), "control.jsonl")
	initial := openTestControlIngress(t, DurableControlIngressOptions{
		JournalPath: journalPath,
		MaxPending:  2,
	})
	for generation := uint64(1); generation <= 2; generation++ {
		directive := ControlDirective{
			ID: "ordered-" + string(rune('0'+generation)), Target: "mission-order",
			Generation: generation, Action: "cancel",
		}
		if status, _, _ := postTestControlDirective(t, initial, directive); status != http.StatusAccepted {
			t.Fatalf("generation %d POST status = %d, want %d", generation, status, http.StatusAccepted)
		}
	}
	if err := initial.Close(); err != nil {
		t.Fatalf("close initial ingress: %v", err)
	}

	delivered := make(chan uint64, 2)
	reopened, err := OpenDurableControlIngress(DurableControlIngressOptions{
		JournalPath: journalPath,
		MaxPending:  2,
		Deliver: func(_ context.Context, directive ControlDirective) error {
			delivered <- directive.Generation
			return nil
		},
	})
	if err != nil {
		t.Fatalf("reopen pending journal: %v", err)
	}
	defer reopened.Close()
	for want := uint64(1); want <= 2; want++ {
		select {
		case got := <-delivered:
			if got != want {
				t.Fatalf("delivery order got generation %d, want %d", got, want)
			}
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for generation %d", want)
		}
	}
}

func TestDurableControlIngressRejectsIDThatCannotRoundTripItemRoute(t *testing.T) {
	for _, id := range []string{"bad/id", ".", ".."} {
		t.Run(id, func(t *testing.T) {
			ingress := openTestControlIngress(t, DurableControlIngressOptions{})
			directive := ControlDirective{ID: id, Target: "mission", Generation: 1, Action: "cancel"}
			if status, receipt, _ := postTestControlDirective(t, ingress, directive); status != http.StatusBadRequest {
				t.Fatalf("ID %q POST = (%d, %+v), want %d", id, status, receipt, http.StatusBadRequest)
			}
		})
	}
}

func TestDurableControlIngressIdempotentSemanticReplayAndGenerationCAS(t *testing.T) {
	ingress := openTestControlIngress(t, DurableControlIngressOptions{})
	first := ControlDirective{
		ID: "control-1", Target: "mission-1", Generation: 1, Action: "cancel",
		Payload: json.RawMessage(`{"alpha":1,"nested":{"x":true,"items":[2,3]}}`),
	}
	status, original, _ := postTestControlDirective(t, ingress, first)
	if status != http.StatusAccepted {
		t.Fatalf("initial POST status = %d, want %d", status, http.StatusAccepted)
	}

	reordered := first
	reordered.Payload = json.RawMessage(" { \"nested\" : { \"items\" : [2, 3], \"x\" : true }, \"alpha\" : 1 } ")
	status, replay, _ := postTestControlDirective(t, ingress, reordered)
	if status != http.StatusAccepted {
		t.Fatalf("semantic replay status = %d, want %d", status, http.StatusAccepted)
	}
	if replay.Sequence != original.Sequence || replay.Digest != original.Digest || !replay.AcceptedAt.Equal(original.AcceptedAt) {
		t.Fatalf("semantic replay minted a new receipt: original=%+v replay=%+v", original, replay)
	}

	conflict := first
	conflict.Payload = json.RawMessage(`{"alpha":2}`)
	if status, _, _ := postTestControlDirective(t, ingress, conflict); status != http.StatusConflict {
		t.Fatalf("same-ID conflicting replay status = %d, want %d", status, http.StatusConflict)
	}
	next := ControlDirective{
		ID: "control-2", Target: first.Target, Generation: 2, Action: "pause",
	}
	if status, _, _ := postTestControlDirective(t, ingress, next); status != http.StatusAccepted {
		t.Fatalf("next monotonic generation status = %d, want %d", status, http.StatusAccepted)
	}
	stale := ControlDirective{
		ID: "control-3", Target: first.Target, Generation: 1, Action: "cancel",
	}
	if status, _, _ := postTestControlDirective(t, ingress, stale); status != http.StatusConflict {
		t.Fatalf("nonmonotonic generation status = %d, want %d", status, http.StatusConflict)
	}
	if status, current := getTestControlReceipt(t, ingress, first.ID); status != http.StatusOK || current.State == "quiesced" {
		t.Fatalf("GET after conflicts = (%d, %+v), want original non-quiesced receipt", status, current)
	}
}

func TestDurableControlIngressBoundedPendingRefusesWithoutBlocking(t *testing.T) {
	deliveryStarted := make(chan struct{}, 1)
	releaseDelivery := make(chan struct{})
	ingress := openTestControlIngress(t, DurableControlIngressOptions{
		MaxPending: 1,
		Deliver: func(context.Context, ControlDirective) error {
			deliveryStarted <- struct{}{}
			<-releaseDelivery
			return nil
		},
	})
	defer close(releaseDelivery)

	first := ControlDirective{ID: "queued-1", Target: "mission-q", Generation: 1, Action: "cancel"}
	if status, _, _ := postTestControlDirective(t, ingress, first); status != http.StatusAccepted {
		t.Fatalf("first POST status = %d, want %d", status, http.StatusAccepted)
	}
	select {
	case <-deliveryStarted:
	case <-time.After(time.Second):
		t.Fatal("delivery callback did not block as arranged")
	}

	second := ControlDirective{ID: "queued-2", Target: "mission-q", Generation: 2, Action: "pause"}
	status, refused, elapsed := postTestControlDirective(t, ingress, second)
	if status != http.StatusServiceUnavailable {
		t.Fatalf("full queue POST status = %d, want %d", status, http.StatusServiceUnavailable)
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("full queue POST took %s while delivery was blocked", elapsed)
	}
	if refused.State == "applied" || refused.State == "quiesced" {
		t.Fatalf("refusal falsely claimed terminal state: %+v", refused)
	}
	if status, _ := getTestControlReceipt(t, ingress, second.ID); status != http.StatusNotFound {
		t.Fatalf("refused directive was persisted: GET status = %d, want %d", status, http.StatusNotFound)
	}
}

func TestDurableControlIngressJournalByteLimitRefusesWithoutPersisting(t *testing.T) {
	ingress := openTestControlIngress(t, DurableControlIngressOptions{
		MaxJournalBytes: 256,
		MaxPending:      1,
	})
	directive := ControlDirective{
		ID: "too-large", Target: "mission-limit", Generation: 1, Action: "cancel",
		Payload: json.RawMessage(`{"reason":"the request fits but its durable receipt frame exceeds the journal limit"}`),
	}
	status, receipt, _ := postTestControlDirective(t, ingress, directive)
	if status != http.StatusServiceUnavailable {
		t.Fatalf("oversize journal POST status = %d, want %d", status, http.StatusServiceUnavailable)
	}
	if receipt.State == "applied" || receipt.State == "quiesced" {
		t.Fatalf("journal limit falsely claimed terminal state: %+v", receipt)
	}
	if status, _ := getTestControlReceipt(t, ingress, directive.ID); status != http.StatusNotFound {
		t.Fatalf("oversize directive was persisted: GET status = %d, want %d", status, http.StatusNotFound)
	}
}

func TestDurableControlIngressRecoveryFailsClosedOnCorruptOrTornJournal(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*testing.T, string)
	}{
		{
			name: "corrupt record",
			mutate: func(t *testing.T, path string) {
				f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
				if err != nil {
					t.Fatalf("open journal to corrupt: %v", err)
				}
				if _, err := f.WriteString("not-json\n"); err != nil {
					t.Fatalf("append corrupt record: %v", err)
				}
				if err := f.Sync(); err != nil {
					t.Fatalf("sync corrupt record: %v", err)
				}
				if err := f.Close(); err != nil {
					t.Fatalf("close corrupt journal: %v", err)
				}
			},
		},
		{
			name: "torn final record",
			mutate: func(t *testing.T, path string) {
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatalf("read journal: %v", err)
				}
				if len(data) == 0 {
					t.Fatal("journal is empty")
				}
				if err := os.Truncate(path, int64(len(data)-1)); err != nil {
					t.Fatalf("tear final record: %v", err)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "control.jsonl")
			ingress := openTestControlIngress(t, DurableControlIngressOptions{JournalPath: path})
			if status, _, _ := postTestControlDirective(t, ingress, ControlDirective{ID: "durable", Target: "mission", Generation: 1, Action: "cancel"}); status != http.StatusAccepted {
				t.Fatalf("seed POST status = %d, want %d", status, http.StatusAccepted)
			}
			if err := ingress.Close(); err != nil {
				t.Fatalf("close seed ingress: %v", err)
			}
			tc.mutate(t, path)
			if reopened, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path, MaxPending: 1}); err == nil {
				reopened.Close()
				t.Fatal("OpenDurableControlIngress accepted an untrustworthy journal")
			}
		})
	}
}

type failingControlJournalFile struct {
	writeErr error
	syncErr  error
	writes   atomic.Int32
	syncs    atomic.Int32
}

type failNthControlJournalSync struct {
	*os.File
	calls  int
	failAt int
}

func (f *failNthControlJournalSync) Sync() error {
	f.calls++
	if f.calls == f.failAt {
		return errors.New("injected sync failure before flush")
	}
	return f.File.Sync()
}

func TestDurableControlIngressSyncErrorRequiresSameIDReconciliation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "control.jsonl")
	provision := openTestControlIngress(t, DurableControlIngressOptions{JournalPath: path})
	if err := provision.Close(); err != nil {
		t.Fatalf("close provisioned ingress: %v", err)
	}

	originalOpen := openControlIngressJournal
	failAt := 2 // Open recovery sync succeeds; the accepted-frame sync fails.
	openControlIngressJournal = func(path string, flag int, perm fs.FileMode) (controlIngressJournalFile, error) {
		f, err := os.OpenFile(path, flag, perm)
		if err != nil {
			return nil, err
		}
		return &failNthControlJournalSync{File: f, failAt: failAt}, nil
	}
	defer func() { openControlIngressJournal = originalOpen }()

	ingress, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path})
	if err != nil {
		t.Fatalf("open failing-sync ingress: %v", err)
	}
	defer ingress.Close()
	directive := ControlDirective{ID: "uncertain-1", Target: "mission-uncertain", Generation: 1, Action: "cancel"}
	status, unknown, _ := postTestControlDirective(t, ingress, directive)
	if status != http.StatusServiceUnavailable || unknown.State != "unknown" || unknown.Reason != "journal_indeterminate" || unknown.Digest == "" {
		t.Fatalf("sync-error POST = (%d, %+v), want indeterminate same-ID retry receipt", status, unknown)
	}
	if err := ingress.Close(); err != nil {
		t.Fatalf("close failing-sync ingress: %v", err)
	}
	failAt = 1 // Recovery cannot claim acceptance until its own flush succeeds.
	if unsynced, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path}); err == nil {
		_ = unsynced.Close()
		t.Fatal("recovery exposed an uncertain frame without successful Sync")
	}
	openControlIngressJournal = originalOpen

	reopened, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path})
	if err != nil {
		t.Fatalf("reopen after indeterminate sync: %v", err)
	}
	defer reopened.Close()
	status, replay, _ := postTestControlDirective(t, reopened, directive)
	if status != http.StatusAccepted || replay.ID != directive.ID || replay.Digest != unknown.Digest || replay.Generation != directive.Generation {
		t.Fatalf("same-ID reconciliation = (%d, %+v), want durable original after uncertainty", status, replay)
	}
}

func (f *failingControlJournalFile) Write(p []byte) (int, error) {
	f.writes.Add(1)
	if f.writeErr != nil {
		return 0, f.writeErr
	}
	return len(p), nil
}

func (f *failingControlJournalFile) Sync() error {
	if f.syncs.Add(1) == 1 {
		return nil // Open's recovery fence succeeds; inject on the POST write.
	}
	return f.syncErr
}

func (*failingControlJournalFile) Close() error { return nil }

func TestDurableControlIngressJournalFailureNeverReportsApplied(t *testing.T) {
	for _, tc := range []struct {
		name     string
		writeErr error
		syncErr  error
	}{
		{name: "append", writeErr: errors.New("injected append failure")},
		{name: "sync", syncErr: errors.New("injected sync failure")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fake := &failingControlJournalFile{writeErr: tc.writeErr, syncErr: tc.syncErr}
			originalOpen := openControlIngressJournal
			openControlIngressJournal = func(string, int, fs.FileMode) (controlIngressJournalFile, error) {
				return fake, nil
			}
			t.Cleanup(func() { openControlIngressJournal = originalOpen })

			var delivered atomic.Int32
			ingress := openTestControlIngress(t, DurableControlIngressOptions{
				Deliver: func(context.Context, ControlDirective) error {
					delivered.Add(1)
					return nil
				},
			})
			status, receipt, _ := postTestControlDirective(t, ingress, ControlDirective{
				ID: "failure-" + tc.name, Target: "mission-f", Generation: 1, Action: "cancel",
			})
			if status != http.StatusServiceUnavailable {
				t.Fatalf("POST status = %d, want %d", status, http.StatusServiceUnavailable)
			}
			if receipt.State != "unknown" || receipt.Reason != "journal_indeterminate" || receipt.Digest == "" {
				t.Fatalf("journal failure did not report indeterminate same-ID retry: %+v", receipt)
			}
			time.Sleep(20 * time.Millisecond)
			if got := delivered.Load(); got != 0 {
				t.Fatalf("journal failure invoked Deliver %d times", got)
			}
			if tc.syncErr != nil && fake.syncs.Load() == 0 {
				t.Fatal("sync failure case never reached Sync")
			}
		})
	}
}

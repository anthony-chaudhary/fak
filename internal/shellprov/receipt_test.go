package shellprov

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func testReceipt(t *testing.T, childPID int) Receipt {
	t.Helper()
	receipt, err := New(time.Date(2026, 8, 26, 12, 34, 56, 789123000, time.FixedZone("west", -7*60*60)), Fields{
		ParentPID:         100,
		ChildPID:          childPID,
		ChildCreatedUTCMS: 1_777_777_000_000 + int64(childPID),
		LaunchClass:       LaunchTool,
		ShellImage:        ShellPwsh,
		ShellEdition:      EditionCore,
		ShellVersion:      "7.6.5",
		Outcome:           OutcomeStarted,
		ErrorClass:        ErrorNone,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return receipt
}

func TestChildIdentityStableAndDistinguishesPIDReuse(t *testing.T) {
	const pid = 4321
	a := ChildIdentity(pid, 1_700_000_000_001)
	b := ChildIdentity(pid, 1_700_000_000_001)
	reused := ChildIdentity(pid, 1_700_000_000_999)
	if a != b {
		t.Fatalf("same child identity was unstable: %q != %q", a, b)
	}
	if a == reused {
		t.Fatalf("PID reuse aliased identity %q", a)
	}
	if !strings.HasPrefix(a, "sha256:") {
		t.Fatalf("identity = %q, want sha256 prefix", a)
	}
}

func TestReceiptSerializationIsPrivacySafe(t *testing.T) {
	receipt := testReceipt(t, 101)
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	secret := "TOP_SECRET_DO_NOT_SERIALIZE"
	rawArgv := "pwsh -Command Write-Output " + secret
	for _, forbidden := range []string{
		secret, rawArgv, "command_line", "commandline", "argv", "script", "environment", "path",
	} {
		if bytes.Contains(bytes.ToLower(encoded), bytes.ToLower([]byte(forbidden))) {
			t.Fatalf("serialized receipt contains forbidden launch content/key %q: %s", forbidden, encoded)
		}
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &keys); err != nil {
		t.Fatal(err)
	}
	if len(keys) != 12 { //boundarylint:ignore CHANGE_DETECTOR_TEST the receipt schema contract contains exactly 12 serialized keys
		t.Fatalf("receipt key count = %d, want closed 12-field schema: %s", len(keys), encoded)
	}
	wantTimestamp := time.Date(2026, 8, 26, 12, 34, 56, 789123000, time.FixedZone("west", -7*60*60)).UTC().UnixMilli()
	if got := receipt.TimestampUTCMS; got != wantTimestamp {
		t.Fatalf("timestamp_utc_ms = %d, want %d UTC Unix milliseconds", got, wantTimestamp)
	}
}

func TestReceiptValidation(t *testing.T) {
	base := testReceipt(t, 102)
	tests := []struct {
		name string
		edit func(*Receipt)
	}{
		{"schema", func(r *Receipt) { r.Schema = "fak.shellprov.receipt.v2" }},
		{"parent pid", func(r *Receipt) { r.ParentPID = 0 }},
		{"child pid", func(r *Receipt) { r.ChildPID = 0 }},
		{"creation time", func(r *Receipt) { r.ChildCreatedUTCMS = 0 }},
		{"identity", func(r *Receipt) { r.LaunchID = "sha256:nope" }},
		{"launch class", func(r *Receipt) { r.LaunchClass = "arbitrary" }},
		{"shell image", func(r *Receipt) { r.ShellImage = "C:/secret/pwsh.exe" }},
		{"edition mismatch", func(r *Receipt) { r.ShellEdition = EditionDesktop }},
		{"shell version", func(r *Receipt) { r.ShellVersion = "7.6.5 C:/secret" }},
		{"outcome", func(r *Receipt) { r.Outcome = "maybe" }},
		{"error class", func(r *Receipt) { r.ErrorClass = "raw error text" }},
		{"failed without class", func(r *Receipt) { r.Outcome = OutcomeFailed }},
		{"success with class", func(r *Receipt) { r.Outcome = OutcomeSucceeded; r.ErrorClass = ErrorTimeout }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			receipt := base
			tc.edit(&receipt)
			if err := receipt.Validate(); err == nil {
				t.Fatal("Validate accepted invalid receipt")
			}
		})
	}
}

func TestAppendConcurrentProducesCompleteLines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	const writers = 48
	receipts := make([]Receipt, writers)
	for i := range receipts {
		receipts[i] = testReceipt(t, 1000+i)
	}
	start := make(chan struct{})
	errs := make(chan error, writers)
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			errs <- Append(path, receipts[i], writers)
		}(i)
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Append: %v", err)
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 || data[len(data)-1] != '\n' {
		t.Fatalf("ledger lacks complete final line: %q", data)
	}
	lines := bytes.Split(bytes.TrimSuffix(data, []byte{'\n'}), []byte{'\n'})
	if len(lines) != writers {
		t.Fatalf("line count = %d, want %d", len(lines), writers)
	}
	seen := make(map[string]bool, writers)
	for i, line := range lines {
		var receipt Receipt
		if err := json.Unmarshal(line, &receipt); err != nil {
			t.Fatalf("line %d is incomplete JSON: %v: %q", i, err, line)
		}
		if err := receipt.Validate(); err != nil {
			t.Fatalf("line %d validation: %v", i, err)
		}
		if seen[receipt.LaunchID] {
			t.Fatalf("duplicate launch identity %s", receipt.LaunchID)
		}
		seen[receipt.LaunchID] = true
	}
}

func TestAppendRetentionKeepsNewestCompleteRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	for pid := 1; pid <= 5; pid++ {
		if err := Append(path, testReceipt(t, pid), 3); err != nil {
			t.Fatalf("Append pid %d: %v", pid, err)
		}
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"partial":`); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	for pid := 6; pid <= 8; pid++ {
		if err := Append(path, testReceipt(t, pid), 3); err != nil {
			t.Fatalf("Append pid %d after torn tail: %v", pid, err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSuffix(data, []byte{'\n'}), []byte{'\n'})
	if len(lines) != 3 {
		t.Fatalf("retained rows = %d, want 3: %s", len(lines), data)
	}
	for i, wantPID := range []int{6, 7, 8} {
		var receipt Receipt
		if err := json.Unmarshal(lines[i], &receipt); err != nil {
			t.Fatalf("row %d: %v", i, err)
		}
		if receipt.ChildPID != wantPID {
			t.Fatalf("row %d child_pid = %d, want %d", i, receipt.ChildPID, wantPID)
		}
	}
}

package harnessprotocol

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestDisconnectResumeFlowIdempotencyCancellationAndBoundaries(t *testing.T) {
	p := NewProducer("run-1", []byte("test-secret"))
	must := func(typ harnesskit.EventType, payload any) {
		t.Helper()
		if _, err := p.Append(typ, "corr", "", harnesskit.SensitivityPublic, payload); err != nil {
			t.Fatal(err)
		}
	}
	must(harnesskit.EventRunStarted, harnesskit.RunPayload{Status: "running"})
	must(harnesskit.EventApprovalRequested, harnesskit.ApprovalPayload{ApprovalID: "a1", Status: "pending"})
	first, c1, err := p.Resume(context.Background(), harnesskit.Cursor{}, 1)
	if err != nil || len(first) != 1 {
		t.Fatalf("first=%d err=%v", len(first), err)
	}
	rest, c2, err := p.Resume(context.Background(), c1, 1)
	if err != nil || len(rest) != 1 || c2.Sequence != 2 {
		t.Fatalf("resume=%d cursor=%+v err=%v", len(rest), c2, err)
	}
	bad := c1
	bad.Checkpoint = "forged"
	if _, _, err = p.Resume(context.Background(), bad, 1); !errors.Is(err, ErrBadCheckpoint) {
		t.Fatalf("forged cursor: %v", err)
	}
	in := harnesskit.Input{Version: harnesskit.ProtocolVersion, RunID: "run-1", InputID: "approve-1", Type: harnesskit.InputApproval, Approval: &harnesskit.ApprovalInput{ApprovalID: "a1", Decision: "approve"}}
	dup, err := p.Apply(in)
	if err != nil || dup {
		t.Fatalf("first input dup=%v err=%v", dup, err)
	}
	dup, err = p.Apply(in)
	if err != nil || !dup {
		t.Fatalf("repeat input dup=%v err=%v", dup, err)
	}
	if _, err = p.Apply(harnesskit.Input{Version: harnesskit.ProtocolVersion, RunID: "run-1", InputID: "cancel", Type: harnesskit.InputCancel, Reason: "stop"}); err != nil {
		t.Fatal(err)
	}
	if _, err = p.Append(harnesskit.EventUsage, "", "", harnesskit.SensitivityPublic, harnesskit.UsagePayload{}); !errors.Is(err, ErrCanceled) {
		t.Fatalf("append after cancel: %v", err)
	}
	secret, _ := json.Marshal(map[string]string{"token": "do-not-leak"})
	red := Redact([]harnesskit.Envelope{{Sensitivity: harnesskit.SensitivitySecret, Payload: secret}}, true)
	if string(red[0].Payload) != "{\"redacted\":true}" {
		t.Fatal(string(red[0].Payload))
	}
}

func TestProjectionAdvancesAcrossUnknownAndRejectsGap(t *testing.T) {
	events := []harnesskit.Envelope{{Version: harnesskit.ProtocolVersion, RunID: "r", Sequence: 1, EventID: "1", Type: harnesskit.EventType("future.x")}, {Version: harnesskit.ProtocolVersion, RunID: "r", Sequence: 2, EventID: "2", Type: harnesskit.EventRunCompleted, Payload: json.RawMessage(`{"status":"completed"}`)}}
	p, err := Project(events)
	if err != nil || p.Cursor != 2 || p.Status != "completed" {
		t.Fatalf("%+v %v", p, err)
	}
	events[1].Sequence = 3
	if _, err = Project(events); err == nil {
		t.Fatal("accepted gap")
	}
}

func TestRecordedProducerDisconnectResumeTwoConsumers(t *testing.T) {
	fixture := filepath.Join("..", "..", "docs", "_witnesses", "harness-protocol", "roundtrip-events.jsonl")
	f, err := os.Open(fixture)
	if err != nil {
		t.Fatal(err)
	}
	recorded, err := ReadJSONL(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	producer := NewProducer("witness-run-6789", []byte("independent-witness"))
	for _, want := range recorded {
		got, err := producer.Append(want.Type, want.CorrelationID, want.CausationID, want.Sensitivity, want.Payload)
		if err != nil {
			t.Fatalf("append %s: %v", want.Type, err)
		}
		if got.Sequence != want.Sequence || got.EventID != want.EventID {
			t.Fatalf("unstable identity: got %+v want %+v", got, want)
		}
	}
	first, disconnectedAt, err := producer.Resume(context.Background(), harnesskit.Cursor{}, 8)
	if err != nil || len(first) != 8 {
		t.Fatalf("first delivery=%d err=%v", len(first), err)
	}
	resumed, finalCursor, err := producer.Resume(context.Background(), disconnectedAt, 8)
	if err != nil || len(resumed) != 8 || finalCursor.Sequence != 16 {
		t.Fatalf("resume=%d cursor=%+v err=%v", len(resumed), finalCursor, err)
	}
	all := append(first, resumed...)
	projection, err := Project(all)
	if err != nil {
		t.Fatal(err)
	}
	if projection.Status != "canceled" || projection.Messages["m1"] != "stable protocol" || projection.Tools["call-1"].Status != "completed" || projection.Usage.InputTokens != 12 || projection.Usage.OutputTokens != 7 {
		t.Fatalf("unstable semantics: %+v", projection)
	}
	cli, tui := CLIText(projection), TUIText(projection)
	if cli == tui {
		t.Fatal("independent consumers rendered identically")
	}
	sha := func(v string) string { sum := sha256.Sum256([]byte(v)); return hex.EncodeToString(sum[:]) }
	if got := sha(cli); got != "9edeb348e094eb003ce1ebab8fe97f67fe8facc24c721ab1b612291849feec99" {
		t.Fatalf("CLI witness hash %s", got)
	}
	if got := sha(tui); got != "1054c96678326058d2447e6866c5b394d236e0598a39fd4facd2cf829ec92e18" {
		t.Fatalf("TUI witness hash %s", got)
	}
}

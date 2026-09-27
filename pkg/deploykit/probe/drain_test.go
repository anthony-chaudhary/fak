package probe

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// liveDrainFixtures are the drain bodies the private router updater's
// TestValidateRouterDrainReceiptRequiresFullyDrained feeds its validator
// (fak-private cmd/fak-sync/update_router_test.go:457-479), with the same
// accept/refuse verdicts.
var liveDrainFixtures = []struct {
	name    string
	status  int
	body    string
	wantErr bool
}{
	{name: "drained", status: 200, body: `{"status":"drained","active_connections":0,"elapsed_ms":12}`},
	{name: "timed out", status: 200, body: `{"status":"timed_out","active_connections":0,"elapsed_ms":15000}`, wantErr: true},
	{name: "aborted", status: 200, body: `{"status":"aborted","active_connections":0,"elapsed_ms":3}`, wantErr: true},
	{name: "connections remain", status: 200, body: `{"status":"drained","active_connections":1,"elapsed_ms":12}`, wantErr: true},
}

// largeSnapshotBody is the 256-row observation_snapshot fixture from
// TestValidateRouterDrainReceiptAcceptsLargeObservationSnapshot.
func largeSnapshotBody() string {
	entry := `{"detail":"` + strings.Repeat("x", 300) + `"}`
	entries := make([]string, 256)
	for i := range entries {
		entries[i] = entry
	}
	return `{"status":"drained","active_connections":0,"elapsed_ms":12,"observation_snapshot":[` + strings.Join(entries, ",") + `]}`
}

func TestDrainResponseRoundTripsLiveFixtureByteForByte(t *testing.T) {
	bodies := []string{largeSnapshotBody()}
	for _, f := range liveDrainFixtures {
		bodies = append(bodies, f.body)
	}
	for _, body := range bodies {
		var resp DrainResponse
		if err := json.Unmarshal([]byte(body), &resp); err != nil {
			t.Fatalf("unmarshal %s: %v", clip(body), err)
		}
		out, err := json.Marshal(resp)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(out, []byte(body)) {
			t.Fatalf("round trip changed the bytes:\n got %s\nwant %s", clip(string(out)), clip(body))
		}
		// fak-server writes through json.Encoder, which appends one newline.
		var buf bytes.Buffer
		if err := json.NewEncoder(&buf).Encode(resp); err != nil {
			t.Fatal(err)
		}
		if buf.String() != body+"\n" {
			t.Fatalf("encoder framing = %q, want the body plus one newline", clip(buf.String()))
		}
	}
}

func TestDecodeDrainResponseMatchesLiveVerdicts(t *testing.T) {
	for _, f := range liveDrainFixtures {
		t.Run(f.name, func(t *testing.T) {
			resp, err := DecodeDrainResponse(f.status, strings.NewReader(f.body))
			if (err != nil) != f.wantErr {
				t.Fatalf("DecodeDrainResponse(%d, %s) error = %v, wantErr %v", f.status, f.body, err, f.wantErr)
			}
			if f.wantErr && !errors.Is(err, ErrNotDrained) {
				t.Fatalf("error %v does not wrap ErrNotDrained", err)
			}
			if resp.Drained() == f.wantErr {
				t.Fatalf("Drained() = %t for %s", resp.Drained(), f.body)
			}
			if resp.Status == "" {
				t.Fatal("a parsed but incomplete drain must still return the decoded response")
			}
		})
	}
	body := largeSnapshotBody()
	if len(body) <= 64<<10 {
		t.Fatalf("fixture size = %d, want greater than the legacy 64 KiB limit", len(body))
	}
	resp, err := DecodeDrainResponse(200, strings.NewReader(body))
	if err != nil || len(resp.ObservationSnapshot) != 256 {
		t.Fatalf("256-row snapshot (%d bytes): rows=%d err=%v", len(body), len(resp.ObservationSnapshot), err)
	}
}

func TestDecodeDrainResponseRefusesOffContractBodies(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
	}{
		{"non-200", 500, `{"status":"drained","active_connections":0,"elapsed_ms":1}`},
		{"unknown field", 200, `{"status":"drained","active_connections":0,"elapsed_ms":1,"extra":true}`},
		{"trailing value", 200, `{"status":"drained","active_connections":0,"elapsed_ms":1} {}`},
		{"not json", 200, `drained`},
		{"empty", 200, ``},
		{"legacy state/active names", 200, `{"state":"drained","active":0}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := DecodeDrainResponse(tt.status, strings.NewReader(tt.body)); err == nil {
				t.Fatalf("DecodeDrainResponse(%d, %s) accepted an off-contract body", tt.status, tt.body)
			}
		})
	}
	if _, err := DecodeDrainResponse(200, nil); err == nil {
		t.Fatal("nil body accepted")
	}
}

func TestDrainResponseOmitsAbsentSnapshot(t *testing.T) {
	out, err := json.Marshal(DrainResponse{Status: DrainStatusDrained})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"status":"drained","active_connections":0,"elapsed_ms":0}` {
		t.Fatalf("legacy shape = %s", out)
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cloudhandoff"
)

func TestCloudHandoffCLI_SelfTest(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runCloudHandoff(&stdout, &stderr, nil, []string{"--self-test"})
	if rc != 0 {
		t.Fatalf("expected 0, got %d, stderr: %s", rc, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte("self-test passed")) {
		t.Fatalf("unexpected output: %s", stdout.String())
	}
}

func TestCloudHandoffCLI_SelfTestJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runCloudHandoff(&stdout, &stderr, nil, []string{"--self-test", "--json"})
	if rc != 0 {
		t.Fatalf("expected 0, got %d, stderr: %s", rc, stderr.String())
	}
	var out cloudHandoffOutput
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		t.Fatalf("invalid json: %v, out: %s", err, stdout.String())
	}
	if out.Schema != cloudhandoff.Schema || out.Receipt.Terminal != "completed" {
		t.Fatalf("unexpected payload: %+v", out)
	}
}

func TestCloudHandoffCLI_EndToEndExecution(t *testing.T) {
	in := cloudHandoffInput{
		Policy: cloudhandoff.Policy{
			Eligible:        true,
			Consent:         cloudhandoff.ConsentPreapproved,
			Destinations:    []string{"vendor-cloud"},
			AllowedTriggers: []cloudhandoff.Trigger{cloudhandoff.TriggerFault},
		},
		Request: cloudhandoff.Request{
			OperationID:      "op-cli-test",
			Trigger:          cloudhandoff.TriggerFault,
			Data:             []cloudhandoff.DataClass{{Name: "tokens"}},
			DestinationClass: "vendor-cloud",
			Payload:          []byte("test data"),
		},
		LocalAttempt: cloudhandoff.Attempt{
			Engine:   "fak-native",
			Location: "local",
			Outcome:  "failed",
		},
	}
	payload, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	rc := runCloudHandoff(&stdout, &stderr, bytes.NewReader(payload), []string{"--json", "-"})
	if rc != 0 {
		t.Fatalf("expected 0, got %d, stderr: %s", rc, stderr.String())
	}

	var out cloudHandoffOutput
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		t.Fatalf("invalid json: %v, out: %s", err, stdout.String())
	}
	if !out.Receipt.RemoteCompleted || out.Receipt.Terminal != "completed" {
		t.Fatalf("unexpected receipt: %+v", out.Receipt)
	}
}

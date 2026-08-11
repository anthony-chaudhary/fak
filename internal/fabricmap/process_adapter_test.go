package fabricmap

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func TestProcessAdapterHelper(t *testing.T) {
	if os.Getenv("FAK_FABRICMAP_HELPER") != "1" {
		return
	}
	var request HelperRequest
	if err := json.NewDecoder(os.Stdin).Decode(&request); err != nil {
		os.Exit(2)
	}
	mode := os.Getenv("FAK_FABRICMAP_HELPER_MODE")
	if mode == "sleep" {
		time.Sleep(2 * time.Second)
		return
	}
	if mode == "exit" {
		fmt.Fprintln(os.Stderr, "public failure")
		os.Exit(7)
	}
	if mode == "malformed" {
		fmt.Print("not-json")
		return
	}
	now := time.Now().UTC()
	receipt := HopReceipt{LinkID: request.Link.ID, From: request.Transfer.Source, To: request.Transfer.Destination, Transport: request.Link.Transport, Bytes: request.Transfer.Bytes, StartedAt: now, CompletedAt: now.Add(time.Millisecond), Integrity: Integrity{Algorithm: "sha256", Digest: "abc"}, State: HopComplete}
	if mode == "reverse" {
		receipt.From, receipt.To = receipt.To, receipt.From
	}
	schema := HelperSchema
	if mode == "schema" {
		schema = "future.v2"
	}
	_ = json.NewEncoder(os.Stdout).Encode(HelperResponse{Schema: schema, Receipt: receipt, CPUPath: "bypass", Evidence: map[string]string{"driver": "test"}})
	os.Exit(0)
}

func helperAdapter(mode string) ProcessAdapter {
	return ProcessAdapter{Command: []string{os.Args[0], "-test.run=TestProcessAdapterHelper"}, Timeout: time.Second, MaxResponseBytes: 4096}
}
func withHelperEnv(t *testing.T, mode string) {
	t.Helper()
	t.Setenv("FAK_FABRICMAP_HELPER", "1")
	t.Setenv("FAK_FABRICMAP_HELPER_MODE", mode)
}
func helperTransfer() (Link, Access, Transfer) {
	link := Link{ID: "ssd-gpu", From: "ssd", To: "gpu", Transport: "gds-v99"}
	access := Access{Principal: "p", Tenant: "t", Operation: OperationCopy}
	return link, access, Transfer{Source: "ssd", Destination: "gpu", Bytes: 4096, Operation: OperationCopy}
}

func TestProcessAdapterReturnsHardwareReceipt(t *testing.T) {
	withHelperEnv(t, "")
	link, access, transfer := helperTransfer()
	receipt, err := helperAdapter("").Transfer(context.Background(), link, access, transfer)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.From != "ssd" || receipt.To != "gpu" || receipt.Transport != "gds-v99" {
		t.Fatalf("receipt=%+v", receipt)
	}
}
func TestProcessAdapterFailsClosedOnReverseReceipt(t *testing.T) {
	withHelperEnv(t, "reverse")
	link, access, transfer := helperTransfer()
	if _, err := helperAdapter("reverse").Transfer(context.Background(), link, access, transfer); err == nil || !strings.Contains(err.Error(), "directed transfer") {
		t.Fatalf("error=%v", err)
	}
}
func TestProcessAdapterFailsClosedOnBadSchema(t *testing.T) {
	withHelperEnv(t, "schema")
	link, access, transfer := helperTransfer()
	if _, err := helperAdapter("schema").Transfer(context.Background(), link, access, transfer); err == nil || !strings.Contains(err.Error(), "schema") {
		t.Fatalf("error=%v", err)
	}
}
func TestProcessAdapterFailsClosedOnMalformedOutput(t *testing.T) {
	withHelperEnv(t, "malformed")
	link, access, transfer := helperTransfer()
	if _, err := helperAdapter("malformed").Transfer(context.Background(), link, access, transfer); err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("error=%v", err)
	}
}
func TestProcessAdapterReportsNonzeroExit(t *testing.T) {
	withHelperEnv(t, "exit")
	link, access, transfer := helperTransfer()
	if _, err := helperAdapter("exit").Transfer(context.Background(), link, access, transfer); err == nil || !strings.Contains(err.Error(), "public failure") {
		t.Fatalf("error=%v", err)
	}
}
func TestProcessAdapterTimesOut(t *testing.T) {
	withHelperEnv(t, "sleep")
	link, access, transfer := helperTransfer()
	adapter := helperAdapter("sleep")
	adapter.Timeout = 20 * time.Millisecond
	if _, err := adapter.Transfer(context.Background(), link, access, transfer); err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("error=%v", err)
	}
}
func TestProcessAdapterRejectsMismatchedInputDirectionBeforeSpawn(t *testing.T) {
	link, access, transfer := helperTransfer()
	transfer.Source = "gpu"
	if _, err := helperAdapter("").Transfer(context.Background(), link, access, transfer); err == nil {
		t.Fatal("mismatch accepted")
	}
}

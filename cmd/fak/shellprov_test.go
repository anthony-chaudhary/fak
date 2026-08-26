package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shellprov"
)

func shellprovRecordArgs(ledger string) []string {
	return []string{
		"record",
		"--ledger", ledger,
		"--parent-pid", "501",
		"--child-pid", "777",
		"--child-created-ms", "1777777000777",
		"--launch-class", "worker",
		"--shell-image", "pwsh",
		"--shell-edition", "core",
		"--shell-version", "7.6.5",
		"--outcome", "succeeded",
		"--error-class", "none",
		"--max-rows", "8",
		"--json",
	}
}

func TestRunShellprovRecordParsesAndAppends(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "shell-receipts.jsonl")
	var stdout, stderr bytes.Buffer
	if code := runShellprov(&stdout, &stderr, shellprovRecordArgs(ledger)); code != 0 {
		t.Fatalf("runShellprov exit = %d, stderr=%s", code, stderr.String())
	}
	var emitted shellprov.Receipt
	if err := json.Unmarshal(stdout.Bytes(), &emitted); err != nil {
		t.Fatalf("decode stdout: %v: %s", err, stdout.String())
	}
	if emitted.Schema != shellprov.ReceiptSchema || emitted.ParentPID != 501 || emitted.ChildPID != 777 {
		t.Fatalf("emitted receipt = %+v", emitted)
	}
	if emitted.LaunchID != shellprov.ChildIdentity(777, 1777777000777) {
		t.Fatalf("launch_id = %q", emitted.LaunchID)
	}
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, stdout.Bytes()) {
		t.Fatalf("ledger row differs from safe JSON output\nledger=%s\nstdout=%s", data, stdout.Bytes())
	}
}

func TestRunShellprovRejectsUnboundedOrInvalidInput(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"missing subcommand", nil, "usage: fak shellprov record"},
		{"unknown subcommand", []string{"kill"}, "usage: fak shellprov record"},
		{"raw argv flag", []string{"record", "--argv", "pwsh -Command SECRET"}, "flag provided but not defined"},
		{"positional script", append(shellprovRecordArgs("ledger.jsonl"), "Write-Output SECRET"), "positional arguments are not accepted"},
		{"invalid launch class", func() []string {
			args := shellprovRecordArgs("ledger.jsonl")
			for i := range args {
				if args[i] == "worker" {
					args[i] = "arbitrary"
				}
			}
			return args
		}(), "invalid launch_class"},
		{"failed without error class", func() []string {
			args := shellprovRecordArgs("ledger.jsonl")
			for i := range args {
				if args[i] == "succeeded" {
					args[i] = "failed"
				}
			}
			return args
		}(), "failed outcome requires"},
		{"retention over bound", func() []string {
			args := shellprovRecordArgs("ledger.jsonl")
			for i := range args {
				if args[i] == "8" {
					args[i] = "65537"
				}
			}
			return args
		}(), "--max-rows must be between"},
		{"negative retention", func() []string {
			args := shellprovRecordArgs("ledger.jsonl")
			for i := range args {
				if args[i] == "8" {
					args[i] = "-1"
				}
			}
			return args
		}(), "--max-rows must be between"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runShellprov(&stdout, &stderr, tc.args); code != 2 {
				t.Fatalf("exit = %d, want usage 2; stderr=%s", code, stderr.String())
			}
			if !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("stderr = %q, want %q", stderr.String(), tc.want)
			}
		})
	}
}

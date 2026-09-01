package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/storagepressure"
)

func TestGuardStoragePressureGate(t *testing.T) {
	tests := []struct {
		name       string
		free       int64
		known      bool
		getwdErr   error
		wantCode   int
		wantText   string
		forbidText string
	}{
		{name: "healthy is silent", free: storagepressure.DefaultWarningFreeBytes + 1, known: true, wantCode: 0},
		{name: "warning boundary continues", free: storagepressure.DefaultWarningFreeBytes, known: true, wantCode: 0, wantText: "storage headroom WARNING"},
		{name: "between floors warns", free: 3 << 30, known: true, wantCode: 0, wantText: "storage headroom WARNING"},
		{name: "refusal boundary stops", free: storagepressure.DefaultRefuseFreeBytes, known: true, wantCode: 1, wantText: "reason=" + guardStoragePressureReason},
		{name: "below reserve stops", free: storagepressure.DefaultRefuseFreeBytes - 1, known: true, wantCode: 1, wantText: "storage headroom REFUSE"},
		{name: "unknown continues explicitly", known: false, wantCode: 0, wantText: "storage headroom UNKNOWN", forbidText: "REFUSE"},
		{name: "cwd failure continues explicitly", getwdErr: errors.New("cwd unavailable"), wantCode: 0, wantText: "storage headroom UNKNOWN", forbidText: "REFUSE"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stderr bytes.Buffer
			code := runGuardStoragePressureGate(&stderr, guardStoragePressureDeps{
				getwd: func() (string, error) {
					if tc.getwdErr != nil {
						return "", tc.getwdErr
					}
					return t.TempDir(), nil
				},
				diskInfo: func(string) (int64, int64, bool) { return 100 << 30, tc.free, tc.known },
			})
			if code != tc.wantCode {
				t.Fatalf("exit = %d, want %d; stderr = %q", code, tc.wantCode, stderr.String())
			}
			if tc.wantText != "" && !strings.Contains(stderr.String(), tc.wantText) {
				t.Fatalf("stderr = %q, want %q", stderr.String(), tc.wantText)
			}
			if tc.wantText == "" && stderr.Len() != 0 {
				t.Fatalf("healthy probe wrote stderr: %q", stderr.String())
			}
			if tc.forbidText != "" && strings.Contains(stderr.String(), tc.forbidText) {
				t.Fatalf("stderr = %q, forbids %q", stderr.String(), tc.forbidText)
			}
		})
	}
}

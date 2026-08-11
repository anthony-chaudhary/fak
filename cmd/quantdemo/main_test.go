package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSelfcheckCapturedOutput(t *testing.T) {
	var out, errw bytes.Buffer
	if code := run(&out, &errw, []string{"-selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errw.String())
	}
	for _, want := range []string{
		"PASS fak-quantdemo/1 fixtures=4",
		"sha256=2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d",
		"runtime=llama.cpp@b6500+ga7a98e0fffed license=MIT",
		"unknown-format=ABSTAIN unknown-runtime=DELEGATE unsupported-combination=REFUSE",
		"composability-only",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestPinsAreMachineReadableAndImmutable(t *testing.T) {
	var out, errw bytes.Buffer
	if code := run(&out, &errw, []string{"-pins"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errw.String())
	}
	var got pins
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Model.SHA256 != ModelSHA256 || got.Model.Bytes != ModelBytes || got.Model.License != "Apache-2.0" {
		t.Fatalf("model pins = %#v", got.Model)
	}
	if got.Runtime.ID != RuntimePin || got.Runtime.LicenseSHA256 == "" {
		t.Fatalf("runtime pins = %#v", got.Runtime)
	}
}

func TestAdjudicateTypesUnknownAndUnsupportedInputs(t *testing.T) {
	for _, tc := range []struct{ format, quant, runtime, result, reason string }{
		{"future@1", QuantQ4KM, RuntimePin, ResultAbstain, ReasonUnknownFormat},
		{FormatGGUFV3, QuantQ4KM, "other@1", ResultRuntimeHandoff, ReasonRuntimeNotPinned},
		{FormatGGUFV3, "iq1_s", RuntimePin, ResultRefuse, ReasonCombinationNotListed},
	} {
		got := adjudicate(tc.format, tc.quant, tc.runtime)
		if got.Result != tc.result || got.Reason != tc.reason {
			t.Fatalf("decision=%#v", got)
		}
	}
}

func TestInspectGGUFTypesUnknownMagicAndVersion(t *testing.T) {
	write := func(name, magic string, version uint32) string {
		p := filepath.Join(t.TempDir(), name)
		var b [8]byte
		copy(b[:4], magic)
		binary.LittleEndian.PutUint32(b[4:], version)
		if err := os.WriteFile(p, b[:], 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	if got, err := inspectGGUF(write("ok.gguf", "GGUF", 3)); err != nil || got != FormatGGUFV3 {
		t.Fatalf("got=%q err=%v", got, err)
	}
	for _, p := range []string{write("magic.bin", "NOPE", 3), write("v9.gguf", "GGUF", 9)} {
		if _, err := inspectGGUF(p); err == nil || !strings.Contains(err.Error(), ReasonUnknownFormat) {
			t.Fatalf("err=%v", err)
		}
	}
}

func TestLiveRefusesUnpinnedRuntimeBeforeNetwork(t *testing.T) {
	_, err := compareLive("missing.gguf", "llama.cpp@latest", "http://invalid", "http://invalid")
	if err == nil || !strings.Contains(err.Error(), ReasonRuntimeNotPinned) {
		t.Fatalf("err=%v", err)
	}
}

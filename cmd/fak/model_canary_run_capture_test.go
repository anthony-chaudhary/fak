package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type modelCanaryDarwinCapture struct {
	Schema         string `json:"schema"`
	CaptureDate    string `json:"capture_date"`
	HostClass      string `json:"host_class"`
	CaptureKind    string `json:"capture_kind"`
	FreshLive      bool   `json:"fresh_live_capture"`
	ProvenanceNote string `json:"provenance_note"`
	Provenance     []struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
	} `json:"provenance_sources"`
	HashContract struct {
		TextEncoding   string `json:"text_encoding"`
		CommandSHA256  string `json:"command_sha256"`
		RawSHA256      string `json:"raw_sha256"`
		ScrubbedSHA256 string `json:"scrubbed_sha256"`
		SourceSHA256   string `json:"source_sha256"`
		BundleSHA256   string `json:"bundle_sha256"`
	} `json:"hash_contract"`
	Sources []struct {
		Name           string `json:"name"`
		Command        string `json:"command"`
		Raw            string `json:"raw_text"`
		Scrubbed       string `json:"scrubbed_text"`
		CommandSHA256  string `json:"command_sha256"`
		RawSHA256      string `json:"raw_sha256"`
		ScrubbedSHA256 string `json:"scrubbed_sha256"`
		SourceSHA256   string `json:"source_sha256"`
	} `json:"sources"`
	BundleSHA256 string `json:"bundle_sha256"`
}

func TestModelCanaryRunDarwinCaptureBindings(t *testing.T) {
	capture := readModelCanaryDarwinCapture(t)
	if err := verifyModelCanaryDarwinCapture(capture); err != nil {
		t.Fatal(err)
	}
	if capture.FreshLive || capture.CaptureKind != "scrubbed_reconstruction_from_committed_live_witness" {
		t.Fatalf("capture provenance is overstated: fresh_live=%v kind=%q", capture.FreshLive, capture.CaptureKind)
	}
	if !strings.Contains(capture.ProvenanceNote, "Original per-command bytes were not retained") ||
		!strings.Contains(capture.ProvenanceNote, "not represented as a fresh live capture") {
		t.Fatalf("capture does not disclose the reconstruction boundary: %q", capture.ProvenanceNote)
	}
	if got, want := len(capture.Sources), 8; got != want {
		t.Fatalf("capture sources=%d want %d", got, want)
	}
	for _, p := range capture.Provenance {
		body, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(p.Path)))
		if err != nil {
			t.Fatalf("read provenance %s: %v", p.Path, err)
		}
		if got := modelCanaryTestSHA(body); got != p.SHA256 {
			t.Fatalf("provenance %s sha256=%s want %s", p.Path, got, p.SHA256)
		}
	}
	body, err := os.ReadFile(filepath.Join("testdata", "model_canary_run_darwin_capture_v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"/Users/", "/private/", "token=", "password=", "secret="} {
		if strings.Contains(strings.ToLower(string(body)), strings.ToLower(forbidden)) {
			t.Fatalf("capture contains private marker %q", forbidden)
		}
	}
}

func TestModelCanaryRunDarwinCaptureTamperReadback(t *testing.T) {
	capture := readModelCanaryDarwinCapture(t)
	if err := verifyModelCanaryDarwinCapture(capture); err != nil {
		t.Fatalf("baseline capture: %v", err)
	}

	mutants := []struct {
		name   string
		mutate func(*modelCanaryDarwinCapture)
	}{
		{"command", func(v *modelCanaryDarwinCapture) { v.Sources[0].Command += " --tampered" }},
		{"raw", func(v *modelCanaryDarwinCapture) { v.Sources[1].Raw += "tampered\n" }},
		{"scrubbed", func(v *modelCanaryDarwinCapture) { v.Sources[2].Scrubbed += "tampered\n" }},
		{"source hash", func(v *modelCanaryDarwinCapture) { v.Sources[3].SourceSHA256 = strings.Repeat("0", 64) }},
		{"bundle hash", func(v *modelCanaryDarwinCapture) { v.BundleSHA256 = strings.Repeat("0", 64) }},
		{"provenance hash", func(v *modelCanaryDarwinCapture) { v.Provenance[0].SHA256 = strings.Repeat("0", 64) }},
	}
	for _, tc := range mutants {
		t.Run(tc.name, func(t *testing.T) {
			copy := capture
			copy.Sources = append([]struct {
				Name           string `json:"name"`
				Command        string `json:"command"`
				Raw            string `json:"raw_text"`
				Scrubbed       string `json:"scrubbed_text"`
				CommandSHA256  string `json:"command_sha256"`
				RawSHA256      string `json:"raw_sha256"`
				ScrubbedSHA256 string `json:"scrubbed_sha256"`
				SourceSHA256   string `json:"source_sha256"`
			}(nil), capture.Sources...)
			copy.Provenance = append([]struct {
				Path   string `json:"path"`
				SHA256 string `json:"sha256"`
			}(nil), capture.Provenance...)
			tc.mutate(&copy)
			if err := verifyModelCanaryDarwinCapture(copy); err == nil {
				t.Fatal("tampered capture verified")
			}
		})
	}
}

func TestModelCanaryRunDarwinCaptureParserInputs(t *testing.T) {
	capture := readModelCanaryDarwinCapture(t)
	if err := verifyModelCanaryDarwinCapture(capture); err != nil {
		t.Fatal(err)
	}
	raw := make(map[string][]byte, len(capture.Sources))
	for _, source := range capture.Sources {
		raw[source.Name] = []byte(source.Raw)
	}

	owner, err := parseModelCanaryLsof(raw["lsof_listener"], 8090)
	if err != nil || owner.PID != 50123 || owner.Command != "fak" {
		t.Fatalf("lsof owner=%+v err=%v", owner, err)
	}
	if pid, plist, err := parseDarwinModelCanaryLaunchctl(raw["launchctl_print"], "gui/501/com.fak.model"); err != nil || pid != 50123 || plist != filepath.Clean("/Library/LaunchAgents/com.fak.model.plist") {
		t.Fatalf("launchctl=(%d,%q,%v)", pid, plist, err)
	}
	identity, command, err := parseModelCanaryPS(raw["ps_identity"], time.UTC)
	if err != nil || identity.PID != 50123 || identity.StartedAt != "2026-08-26T07:52:02Z" || command != "/opt/fak/bin/fak model serve --port 8090" || identity.ArgvSHA256 != digestBytes([]byte(command)) {
		t.Fatalf("ps identity=%+v command=%q err=%v", identity, command, err)
	}
	if got, err := parseModelCanaryRSS(raw["ps_rss"], 50123); err != nil || got != 19_267_272_704 {
		t.Fatalf("rss=%d err=%v", got, err)
	}
	if got, err := parseModelCanaryFootprint(raw["footprint"]); err != nil || got != 35_065_430_016 {
		t.Fatalf("footprint=%d err=%v", got, err)
	}
	if got, err := parseModelCanarySwap(raw["swapusage"]); err != nil || got != 2_954_362_880 {
		t.Fatalf("swap=%d err=%v", got, err)
	}
	if got, err := parseModelCanaryMemoryPressure(raw["memory_pressure"]); err != nil || got != 42 {
		t.Fatalf("system-free=%d err=%v", got, err)
	}
	if got, err := parseModelCanaryMemorystatus(raw["memorystatus"]); err != nil || got != 42 {
		t.Fatalf("memorystatus=%d err=%v", got, err)
	}
}

func TestModelCanaryRunDarwinParsersRejectGNUOnlyAndMalformedShapes(t *testing.T) {
	tests := []struct {
		name  string
		parse func() error
	}{
		{"lsof table instead of BSD fields", func() error {
			_, err := parseModelCanaryLsof([]byte("COMMAND PID USER FD TYPE NAME\nfak 50123 user 17u IPv6 *:8090 (LISTEN)\n"), 8090)
			return err
		}},
		{"lsof multiple owners", func() error {
			_, err := parseModelCanaryLsof([]byte("p1\nca\nn*:8090\np2\ncb\nn*:8090\n"), 8090)
			return err
		}},
		{"GNU ps header", func() error {
			_, _, err := parseModelCanaryPS([]byte("PID STARTED COMMAND\n50123 Wed Aug 26 07:52:02 2026 fak\n"), time.UTC)
			return err
		}},
		{"GNU ps elapsed start", func() error {
			_, _, err := parseModelCanaryPS([]byte("50123 01:02:03 fak --serve\n"), time.UTC)
			return err
		}},
		{"ps RSS wrong PID", func() error { _, err := parseModelCanaryRSS([]byte("50124 100\n"), 50123); return err }},
		{"ps RSS extra column", func() error { _, err := parseModelCanaryRSS([]byte("50123 100 fak\n"), 50123); return err }},
		{"GNU maxrss", func() error {
			_, err := parseModelCanaryFootprint([]byte("Maximum resident set size (kbytes): 33441\n"))
			return err
		}},
		{"duplicate footprint", func() error {
			_, err := parseModelCanaryFootprint([]byte("phys_footprint: 1 MB\nphysical footprint: 1 MB\n"))
			return err
		}},
		{"GNU free swap", func() error { _, err := parseModelCanarySwap([]byte("Swap: 5120 2817 2303\n")); return err }},
		{"swap missing unit", func() error { _, err := parseModelCanarySwap([]byte("vm.swapusage: used = 2817\n")); return err }},
		{"memory pressure label drift", func() error { _, err := parseModelCanaryMemoryPressure([]byte("Memory available: 42%\n")); return err }},
		{"memorystatus sysctl label", func() error {
			_, err := parseModelCanaryMemorystatus([]byte("kern.memorystatus_level: 42\n"))
			return err
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.parse(); err == nil {
				t.Fatal("malformed/GNU-only shape was accepted")
			}
		})
	}
}

func readModelCanaryDarwinCapture(t *testing.T) modelCanaryDarwinCapture {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "model_canary_run_darwin_capture_v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var capture modelCanaryDarwinCapture
	dec := json.NewDecoder(strings.NewReader(string(body)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&capture); err != nil {
		t.Fatalf("decode capture: %v", err)
	}
	return capture
}

func verifyModelCanaryDarwinCapture(c modelCanaryDarwinCapture) error {
	if c.Schema != "fak.model-canary-run-darwin-capture/v1" || c.CaptureDate == "" || c.HostClass == "" {
		return fmt.Errorf("capture identity is incomplete")
	}
	wantHashContract := struct {
		TextEncoding   string `json:"text_encoding"`
		CommandSHA256  string `json:"command_sha256"`
		RawSHA256      string `json:"raw_sha256"`
		ScrubbedSHA256 string `json:"scrubbed_sha256"`
		SourceSHA256   string `json:"source_sha256"`
		BundleSHA256   string `json:"bundle_sha256"`
	}{
		TextEncoding:   "UTF-8",
		CommandSHA256:  "sha256(command UTF-8 bytes)",
		RawSHA256:      "sha256(raw_text UTF-8 bytes)",
		ScrubbedSHA256: "sha256(scrubbed_text UTF-8 bytes)",
		SourceSHA256:   "sha256(name + LF + command_sha256 + LF + raw_sha256 + LF + scrubbed_sha256 + LF)",
		BundleSHA256:   "sha256(schema + LF + capture_date + LF + host_class + LF + each source_sha256 + LF in listed order + each provenance path + LF + sha256 + LF in listed order)",
	}
	if c.HashContract != wantHashContract {
		return fmt.Errorf("capture hash contract drift")
	}
	if len(c.Sources) == 0 || len(c.Provenance) == 0 {
		return fmt.Errorf("capture provenance is incomplete")
	}
	var bundle strings.Builder
	fmt.Fprintf(&bundle, "%s\n%s\n%s\n", c.Schema, c.CaptureDate, c.HostClass)
	seen := make(map[string]bool, len(c.Sources))
	for _, source := range c.Sources {
		if source.Name == "" || seen[source.Name] {
			return fmt.Errorf("duplicate or empty capture source %q", source.Name)
		}
		seen[source.Name] = true
		commandSHA := modelCanaryTestSHA([]byte(source.Command))
		rawSHA := modelCanaryTestSHA([]byte(source.Raw))
		scrubbedSHA := modelCanaryTestSHA([]byte(source.Scrubbed))
		if commandSHA != source.CommandSHA256 || rawSHA != source.RawSHA256 || scrubbedSHA != source.ScrubbedSHA256 {
			return fmt.Errorf("capture source %s byte hash mismatch", source.Name)
		}
		binding := source.Name + "\n" + commandSHA + "\n" + rawSHA + "\n" + scrubbedSHA + "\n"
		if got := modelCanaryTestSHA([]byte(binding)); got != source.SourceSHA256 {
			return fmt.Errorf("capture source %s binding hash mismatch", source.Name)
		}
		fmt.Fprintf(&bundle, "%s\n", source.SourceSHA256)
	}
	for _, provenance := range c.Provenance {
		digest, err := hex.DecodeString(provenance.SHA256)
		if provenance.Path == "" || err != nil || len(digest) != sha256.Size || provenance.SHA256 != hex.EncodeToString(digest) {
			return fmt.Errorf("capture provenance binding is incomplete")
		}
		fmt.Fprintf(&bundle, "%s\n%s\n", provenance.Path, provenance.SHA256)
	}
	if got := modelCanaryTestSHA([]byte(bundle.String())); got != c.BundleSHA256 {
		return fmt.Errorf("capture bundle hash mismatch")
	}
	return nil
}

func modelCanaryTestSHA(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

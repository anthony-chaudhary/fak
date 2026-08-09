package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runQualityRunJSON plants a defect through `fak quality run --json` and returns
// the stored artifact plus the file it was written to. This is the operator-visible
// half of the #4515 round trip: everything after this point sees only the file.
func runQualityRunJSON(t *testing.T, defect string) (string, []byte) {
	t.Helper()
	var out, errOut bytes.Buffer
	code := runQualityRun(&out, &errOut, []string{"-inject", defect, "-json"})
	wantCode := 1
	if defect == "" {
		wantCode = 0
	}
	if code != wantCode {
		t.Fatalf("run --inject %q exit = %d, want %d (stderr: %s)", defect, code, wantCode, errOut.String())
	}
	path := filepath.Join(t.TempDir(), "result.json")
	if err := os.WriteFile(path, out.Bytes(), 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	return path, out.Bytes()
}

// TestQualityReplayRoundTripsAnInjectedFailure is the shipped witness for
// "one command replays an injected failure from its bundle" (#4515): a planted
// defect is run, its artifact is stored, and the replay verb reproduces the same
// failing oracle at the same first divergence from that file alone — exit 0.
func TestQualityReplayRoundTripsAnInjectedFailure(t *testing.T) {
	for _, defect := range []string{"decode", "stop", "report"} {
		t.Run(defect, func(t *testing.T) {
			path, _ := runQualityRunJSON(t, defect)

			var out, errOut bytes.Buffer
			if code := runQualityReplay(&out, &errOut, strings.NewReader(""), []string{"-bundle", path}); code != 0 {
				t.Fatalf("replay exit = %d, want 0\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
			}
			if !strings.Contains(out.String(), "REPRODUCED") {
				t.Errorf("replay output does not state REPRODUCED:\n%s", out.String())
			}

			out.Reset()
			errOut.Reset()
			if code := runQualityReplay(&out, &errOut, strings.NewReader(""), []string{"-bundle", path, "-json"}); code != 0 {
				t.Fatalf("replay --json exit = %d, want 0 (stderr: %s)", code, errOut.String())
			}
			var v struct {
				Schema     string `json:"schema"`
				CaseID     string `json:"case_id"`
				Reproduced bool   `json:"reproduced"`
				Expected   struct {
					FailingOracle string `json:"failing_oracle"`
				} `json:"expected"`
				Observed *struct {
					FailingOracle string `json:"failing_oracle"`
				} `json:"observed"`
			}
			if err := json.Unmarshal(out.Bytes(), &v); err != nil {
				t.Fatalf("parse replay verdict: %v\n%s", err, out.String())
			}
			if !v.Reproduced {
				t.Errorf("verdict JSON says not reproduced: %s", out.String())
			}
			if v.Schema != "fak-quality-replay/1" {
				t.Errorf("verdict schema = %q", v.Schema)
			}
			if v.Observed == nil || v.Observed.FailingOracle != v.Expected.FailingOracle {
				t.Errorf("observed oracle does not match the bundle's: %s", out.String())
			}
		})
	}
}

// TestQualityReplayReadsStdin proves the round trip pipes:
// `fak quality run --inject decode --json | fak quality replay --bundle -`.
func TestQualityReplayReadsStdin(t *testing.T) {
	_, blob := runQualityRunJSON(t, "decode")
	var out, errOut bytes.Buffer
	if code := runQualityReplay(&out, &errOut, bytes.NewReader(blob), []string{"-bundle", "-"}); code != 0 {
		t.Fatalf("replay from stdin exit = %d, want 0 (stderr: %s)", code, errOut.String())
	}
	if !strings.Contains(out.String(), "REPRODUCED") {
		t.Errorf("replay from stdin did not reproduce:\n%s", out.String())
	}
}

// TestQualityReplayNeverPassesWithoutEvidence keeps the verb's exit codes honest:
// a bundle that cannot be replayed exits 1 (inconclusive is never a pass), while a
// missing/unusable INPUT exits 2 — an infrastructure error is never reported as a
// quality verdict, in either direction.
func TestQualityReplayNeverPassesWithoutEvidence(t *testing.T) {
	path, blob := runQualityRunJSON(t, "decode")

	// Strip the engine trace out of the stored artifact: replay-critical evidence
	// is gone, so the verb must report it rather than pass.
	var doc map[string]any
	if err := json.Unmarshal(blob, &doc); err != nil {
		t.Fatalf("parse artifact: %v", err)
	}
	fb, ok := doc["failure_bundle"].(map[string]any)
	if !ok {
		t.Fatalf("artifact carries no failure_bundle: %s", blob)
	}
	fb["engine"] = map[string]any{"runner": "engine-decode-defect", "tokens": []any{}, "text": ""}
	stripped, err := json.Marshal(fb)
	if err != nil {
		t.Fatalf("marshal stripped bundle: %v", err)
	}
	strippedPath := filepath.Join(t.TempDir(), "stripped.json")
	if err := os.WriteFile(strippedPath, stripped, 0o600); err != nil {
		t.Fatalf("write stripped bundle: %v", err)
	}

	var out, errOut bytes.Buffer
	if code := runQualityReplay(&out, &errOut, strings.NewReader(""), []string{"-bundle", strippedPath}); code != 1 {
		t.Fatalf("stripped bundle exit = %d, want 1\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "INCONCLUSIVE") {
		t.Errorf("stripped bundle must render INCONCLUSIVE:\n%s", out.String())
	}

	for name, argv := range map[string][]string{
		"no -bundle flag": {},
		"missing file":    {"-bundle", filepath.Join(t.TempDir(), "absent.json")},
		"unknown flag":    {"-bundle", path, "-nope"},
	} {
		t.Run(name, func(t *testing.T) {
			var o, e bytes.Buffer
			if code := runQualityReplay(&o, &e, strings.NewReader(""), argv); code != 2 {
				t.Fatalf("%s exit = %d, want 2 (stdout: %s)", name, code, o.String())
			}
		})
	}

	t.Run("a passing result is not a bundle", func(t *testing.T) {
		cleanPath, _ := runQualityRunJSON(t, "")
		var o, e bytes.Buffer
		if code := runQualityReplay(&o, &e, strings.NewReader(""), []string{"-bundle", cleanPath}); code != 2 {
			t.Fatalf("clean result exit = %d, want 2 (stdout: %s stderr: %s)", code, o.String(), e.String())
		}
	})
}

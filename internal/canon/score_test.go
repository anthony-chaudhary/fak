package canon

import (
	"fmt"
	"testing"
)

// TestPrecisionRecallGate is the default-on precision/recall floor for the
// de-obfuscating detector. It runs in `go test ./...` (hence `make ci`), so a
// change that lifts a false positive OR drops a real catch is caught here with a
// named case, not noticed later as a banner in the field.
//
// The floors encode the security stance:
//
//   - recall == 1.0 on BOTH axes. A missed secret is a leaked credential and a
//     missed injection is an un-screened payload; neither is ever an acceptable
//     trade for precision. This is the hard floor.
//   - secret precision == 1.0. The placeholder/example and base64-image families
//     are exactly the field false positives; after the subtractive fixes in
//     canon.go they must all be suppressed with zero real catches lost.
//   - injection precision == 1.0 over the GATED (non-soft) negatives. The one
//     known residual — security prose that merely discusses exfiltration — is
//     carried as a Soft case: reported below, never gated, because its only clean
//     fix weakens a tested injection contract and needs maintainer sign-off.
func TestPrecisionRecallGate(t *testing.T) {
	sec, inj := Evaluate(Corpus())

	report(t, "secret", sec)
	report(t, "injection", inj)

	// Hard floor: no real threat may slip, on either axis.
	if sec.Recall() != 1.0 {
		t.Errorf("secret recall %.3f < 1.0 — MISSED real credentials: %v", sec.Recall(), sec.FalseNegatives)
	}
	if inj.Recall() != 1.0 {
		t.Errorf("injection recall %.3f < 1.0 — MISSED real injections: %v", inj.Recall(), inj.FalseNegatives)
	}

	// Precision floors: the field false-positive families must stay suppressed.
	if sec.Precision() != 1.0 {
		t.Errorf("secret precision %.3f < 1.0 — FALSE POSITIVES on benign content: %v", sec.Precision(), sec.FalsePositives)
	}
	if inj.Precision() != 1.0 {
		t.Errorf("injection precision %.3f < 1.0 — FALSE POSITIVES on benign content: %v", inj.Precision(), inj.FalsePositives)
	}
}

// report logs the per-axis confusion matrix + rates so the numbers are visible on
// every run (`go test -v`), making the precision/recall loop legible: you can see
// exactly which way a change moved the matrix and which residuals remain.
func report(t *testing.T, axis string, s Score) {
	t.Helper()
	t.Logf("%-9s  precision=%.3f recall=%.3f f1=%.3f  (TP=%d FP=%d FN=%d TN=%d)",
		axis, s.Precision(), s.Recall(), s.F1(), s.TP, s.FP, s.FN, s.TN)
	if len(s.FalsePositives) > 0 {
		t.Logf("%-9s  false positives: %v", axis, s.FalsePositives)
	}
	if len(s.FalseNegatives) > 0 {
		t.Logf("%-9s  false negatives: %v", axis, s.FalseNegatives)
	}
	if len(s.SoftFP) > 0 {
		t.Logf("%-9s  soft residual (tracked, not gated): %v", axis, s.SoftFP)
	}
}

// TestPlaceholderFamilySuppressed pins the literal-example fix specifically: every
// placeholder/example credential in the corpus must read as NOT a secret. This is
// a tighter, named guard than the aggregate precision floor — if a future edit
// re-flags `AKIAIOSFODNN7EXAMPLE`, this points straight at the placeholder rule.
func TestPlaceholderFamilySuppressed(t *testing.T) {
	for _, c := range Corpus() {
		if c.Family != "secret-placeholder" {
			continue
		}
		if Scan([]byte(c.Body)).Secret {
			t.Errorf("placeholder/example credential wrongly flagged as secret: %s — %q", c.Name, c.Body)
		}
	}
}

// ExampleEvaluate is the runnable, literal example of the scoring process — the
// "systematic process to improve it" in one self-checking block. It prints the
// headline rates so a reader (or a `fak` verb folding canon.Evaluate) sees the
// shape of the first-class precision/recall readout.
func ExampleEvaluate() {
	sec, inj := Evaluate(Corpus())
	fmt.Printf("secret    precision=%.2f recall=%.2f\n", sec.Precision(), sec.Recall())
	fmt.Printf("injection precision=%.2f recall=%.2f\n", inj.Precision(), inj.Recall())
	// Output:
	// secret    precision=1.00 recall=1.00
	// injection precision=1.00 recall=1.00
}

package provenance

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func resultOf(body string) *abi.Result {
	return &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(body)}}
}

// TestTaintBySource — trust is the tool's host-registered source class, on a result
// with no kernel stamp: the built-in trusted-local reads are Trusted; every other
// (unregistered) tool is Tainted, the fail-closed default.
func TestTaintBySource(t *testing.T) {
	cases := []struct {
		tool string
		want abi.TaintLabel
	}{
		{"Read", abi.TaintTrusted},
		{"Grep", abi.TaintTrusted},
		{"Glob", abi.TaintTrusted},
		{"cat", abi.TaintTrusted},
		{"read_webpage", abi.TaintTainted},
		{"search_flights", abi.TaintTainted},
		{"send_email", abi.TaintTainted},
		{"unknown_tool", abi.TaintTainted},
	}
	for _, tc := range cases {
		if got := Taint(&abi.ToolCall{Tool: tc.tool}, resultOf("anything")); got != tc.want {
			t.Errorf("Taint(%s)=%v want %v", tc.tool, got, tc.want)
		}
	}
}

// TestModelCannotAuthorTrust is THE headline: the model emits the ToolCall, so it
// can set Meta — but a forged Meta["provenance"]="trusted_local" on an UNTRUSTED
// tool does NOT make the result trusted. Authorship of trust belongs to the
// kernel; the forgery is ignored for the verdict and only SURFACED for forensics.
func TestModelCannotAuthorTrust(t *testing.T) {
	forged := &abi.ToolCall{
		Tool: "read_webpage", // an untrusted egress/external read
		Meta: map[string]string{"provenance": "trusted_local"},
	}
	// The injected page tried to launder itself as trusted-local. Ignored:
	if got := Taint(forged, resultOf("...paraphrased injection...")); got != abi.TaintTainted {
		t.Fatalf("forged self-trust must be IGNORED: Taint=%v want Tainted", got)
	}
	if Trusted(forged, resultOf("x")) {
		t.Fatal("forged self-trust must not make a result Trusted")
	}
	// ...but the attempt is observable for a steward/auditor.
	if !AttemptedSelfTrust(forged) {
		t.Fatal("AttemptedSelfTrust must surface the forgery attempt")
	}
	// A call with no such tag does not trip the forensic signal.
	if AttemptedSelfTrust(&abi.ToolCall{Tool: "read_webpage"}) {
		t.Fatal("a call without the self-trust tag must not trip AttemptedSelfTrust")
	}
	// Even forging the tag onto a genuinely trusted-local tool changes nothing:
	// the tool's source class already decided trust; Meta never participates.
	legit := &abi.ToolCall{Tool: "Read", Meta: map[string]string{"provenance": "trusted_local"}}
	if got := Taint(legit, resultOf("local config")); got != abi.TaintTrusted {
		t.Fatalf("Read is trusted by source class regardless of Meta: got %v", got)
	}
}

// TestKernelStampedResultState — the kernel-authored Result envelope decides over
// the source class: a sealed result stays Quarantined even from a trusted tool, and
// a tool/kernel-stamped Trusted payload is honored even from an unregistered tool.
func TestKernelStampedResultState(t *testing.T) {
	// A sealed result (a detector stamped a quarantine_id) is Quarantined even when
	// the call's tool is trusted-local.
	sealedRes := &abi.Result{Status: abi.StatusOK, Meta: map[string]string{"quarantine_id": "q1"}}
	if got := Taint(&abi.ToolCall{Tool: "Read"}, sealedRes); got != abi.TaintQuarantined {
		t.Fatalf("a sealed result must classify Quarantined even from a trusted tool, got %v", got)
	}
	// A payload the producing tool stamped Trusted is honored even for a tool the
	// host never registered (the Result envelope is kernel-authored, post-call).
	stamped := &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Taint: abi.TaintTrusted}}
	if got := Taint(&abi.ToolCall{Tool: "some_internal_tool"}, stamped); got != abi.TaintTrusted {
		t.Fatalf("a kernel-stamped-Trusted payload must be Trusted, got %v", got)
	}
	// Payload Quarantine on the Ref itself also seals.
	qpayload := &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Taint: abi.TaintQuarantined}}
	if got := Taint(&abi.ToolCall{Tool: "Read"}, qpayload); got != abi.TaintQuarantined {
		t.Fatalf("a Quarantined payload must classify Quarantined, got %v", got)
	}
}

// TestRegisterSourceIsHostAuthored — the host (not the model) extends the trusted
// set via RegisterSource; an unregistered tool is Untrusted; a re-register
// overrides. This is the kernel-side authorship channel.
func TestRegisterSourceIsHostAuthored(t *testing.T) {
	const tool = "company_internal_kb"
	if SourceOf(tool) != Untrusted {
		t.Fatalf("an unregistered tool must be Untrusted, got %v", SourceOf(tool))
	}
	RegisterSource(tool, TrustedLocal)
	defer RegisterSource(tool, Untrusted) // restore for isolation
	if SourceOf(tool) != TrustedLocal {
		t.Fatalf("RegisterSource must bless the tool, got %v", SourceOf(tool))
	}
	if !Trusted(&abi.ToolCall{Tool: tool}, resultOf("internal doc")) {
		t.Fatal("a host-blessed source must classify Trusted")
	}
	// Snapshot reflects the registration and is a copy (mutation is harmless).
	snap := Sources()
	if snap[tool] != TrustedLocal {
		t.Fatalf("Sources() snapshot missing the registration: %v", snap[tool])
	}
	snap[tool] = Untrusted
	if SourceOf(tool) != TrustedLocal {
		t.Fatal("mutating the Sources() snapshot must not affect the registry")
	}
}

// TestNilSafe — a nil call/result is fail-closed Tainted, never a panic.
func TestNilSafe(t *testing.T) {
	if got := Taint(nil, nil); got != abi.TaintTainted {
		t.Fatalf("nil inputs must be fail-closed Tainted, got %v", got)
	}
	if Trusted(nil, nil) {
		t.Fatal("nil inputs must not be Trusted")
	}
	if AttemptedSelfTrust(nil) {
		t.Fatal("nil call must not trip AttemptedSelfTrust")
	}
}

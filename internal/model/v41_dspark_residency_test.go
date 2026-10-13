// fak-test:runtime fast est=0.2s lane=default

package model

import (
	"errors"
	"math"
	"regexp"
	"strings"
	"testing"
)

// --- helpers ---------------------------------------------------------------

var v41DSparkHex40 = regexp.MustCompile(`^[0-9a-f]{40}$`)

func v41DSparkDefaultPins(t *testing.T) []V41DSparkSourcePin {
	t.Helper()
	pins := V41DSparkPins()
	if len(pins) == 0 {
		t.Fatal("V41DSparkPins() returned no pins")
	}
	return pins
}

func v41DSparkCopyPins(pins []V41DSparkSourcePin) []V41DSparkSourcePin {
	out := make([]V41DSparkSourcePin, len(pins))
	copy(out, pins)
	return out
}

func v41DSparkValidPin(t *testing.T) V41DSparkSourcePin {
	t.Helper()
	return v41DSparkDefaultPins(t)[0]
}

func v41DSparkValidClass(c V41DSparkPortClass) bool {
	switch c {
	case V41DSparkPortAdapt, V41DSparkPortDirectPort, V41DSparkPortInspireOnly:
		return true
	}
	return false
}

func v41DSparkFindAcceptance(s V41DSparkStudy, w V41DSparkWorkload) int {
	for i := range s.Acceptance {
		if s.Acceptance[i].Workload == w {
			return i
		}
	}
	return -1
}

func v41DSparkDropWorkload(s V41DSparkStudy, w V41DSparkWorkload) V41DSparkStudy {
	out := make([]V41DSparkAcceptance, 0, len(s.Acceptance))
	for _, a := range s.Acceptance {
		if a.Workload != w {
			out = append(out, a)
		}
	}
	s.Acceptance = out
	return s
}

// --- defaults & pins -------------------------------------------------------

func TestV41DSparkResidencyDefaultStudyValidates(t *testing.T) {
	if err := DefaultV41DSparkStudy().Validate(); err != nil {
		t.Fatalf("DefaultV41DSparkStudy().Validate() = %v, want nil", err)
	}
}

func TestV41DSparkResidencyPins(t *testing.T) {
	pins := V41DSparkPins()
	if len(pins) != 4 {
		t.Fatalf("V41DSparkPins() len = %d, want 4", len(pins))
	}
	for i, p := range pins {
		if p.Repo == "" {
			t.Errorf("pin[%d] (%s): empty Repo", i, p.Path)
		}
		if p.Path == "" {
			t.Errorf("pin[%d] %s: empty Path", i, p.Repo)
		}
		if p.License == "" {
			t.Errorf("pin[%d] %s: empty License", i, p.Path)
		}
		if !v41DSparkHex40.MatchString(p.Revision) {
			t.Errorf("pin[%d] %s: Revision %q is not 40-hex lowercase", i, p.Path, p.Revision)
		}
		if p.LineStart < 0 {
			t.Errorf("pin[%d] %s: LineStart = %d, want >= 0", i, p.Path, p.LineStart)
		}
		if p.LineEnd < p.LineStart {
			t.Errorf("pin[%d] %s: LineEnd %d < LineStart %d", i, p.Path, p.LineEnd, p.LineStart)
		}
		if !v41DSparkValidClass(p.Class) {
			t.Errorf("pin[%d] %s: invalid Class %q", i, p.Path, p.Class)
		}
	}
}

// --- pin validation --------------------------------------------------------

func TestV41DSparkResidencyValidatePins(t *testing.T) {
	if err := ValidateV41DSparkPins(nil); err == nil {
		t.Error("ValidateV41DSparkPins(nil) = nil, want error")
	}
	if err := ValidateV41DSparkPins([]V41DSparkSourcePin{}); err == nil {
		t.Error("ValidateV41DSparkPins(empty slice) = nil, want error")
	}
	if err := ValidateV41DSparkPins(V41DSparkPins()); err != nil {
		t.Errorf("ValidateV41DSparkPins(V41DSparkPins()) = %v, want nil", err)
	}

	base := v41DSparkValidPin(t)
	const hex39 = "0123456789abcdef0123456789abcdef0123456" // 39 chars

	cases := []struct {
		name string
		mut  func(p *V41DSparkSourcePin)
	}{
		{"empty repo", func(p *V41DSparkSourcePin) { p.Repo = "" }},
		{"empty path", func(p *V41DSparkSourcePin) { p.Path = "" }},
		{"empty revision", func(p *V41DSparkSourcePin) { p.Revision = "" }},
		{"upper-case revision", func(p *V41DSparkSourcePin) {
			p.Revision = strings.ToUpper(p.Revision)
		}},
		{"short revision", func(p *V41DSparkSourcePin) { p.Revision = "abc123" }},
		{"39-char revision", func(p *V41DSparkSourcePin) { p.Revision = hex39 }},
		{"unset class", func(p *V41DSparkSourcePin) { p.Class = "" }},
		{"wrong-class string", func(p *V41DSparkSourcePin) { p.Class = V41DSparkPortClass("BOGUS") }},
		{"unknown license", func(p *V41DSparkSourcePin) { p.License = "WTFPL-9000" }},
		{"empty license", func(p *V41DSparkSourcePin) { p.License = "" }},
		{"negative LineStart", func(p *V41DSparkSourcePin) { p.LineStart = -1 }},
		{"LineEnd < LineStart", func(p *V41DSparkSourcePin) {
			p.LineStart = 100
			p.LineEnd = 10
		}},
		{"empty blob", func(p *V41DSparkSourcePin) { p.BlobSHA = "" }},
		{"39-hex blob", func(p *V41DSparkSourcePin) { p.BlobSHA = hex39 }},
		{"upper-case blob", func(p *V41DSparkSourcePin) { p.BlobSHA = strings.ToUpper(p.BlobSHA) }},
		{"code anchor with empty blob", func(p *V41DSparkSourcePin) {
			p.LineStart = 10
			p.LineEnd = 20
			p.BlobSHA = ""
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pin := base
			tc.mut(&pin)
			pins := v41DSparkCopyPins(V41DSparkPins())
			pins[0] = pin
			if err := ValidateV41DSparkPins(pins); err == nil {
				t.Errorf("ValidateV41DSparkPins(tampered %q) = nil, want error", tc.name)
			}
		})
	}

	t.Run("valid baseline copy", func(t *testing.T) {
		if err := ValidateV41DSparkPins(v41DSparkCopyPins(V41DSparkPins())); err != nil {
			t.Errorf("ValidateV41DSparkPins(valid copy) = %v, want nil", err)
		}
	})

	t.Run("doc-only pin with empty blob", func(t *testing.T) {
		pin := base
		pin.LineStart = 0
		pin.LineEnd = 0
		pin.BlobSHA = ""
		pins := v41DSparkCopyPins(V41DSparkPins())
		pins[0] = pin
		if err := ValidateV41DSparkPins(pins); err != nil {
			t.Errorf("ValidateV41DSparkPins(doc-only pin with empty BlobSHA) = %v, want nil", err)
		}
	})
}

// --- residency -------------------------------------------------------------

func TestV41DSparkResidencyResidentBytesAndFitsHost(t *testing.T) {
	cases := []struct {
		draft, target, budget int64
		resident              int64
		fits                  bool
	}{
		{0, 0, 0, 0, true},
		{1, 2, 3, 3, true},
		{1, 2, 2, 3, false},
		{5, 5, 11, 10, true},
		{V41DSparkGiB, 152 * V41DSparkGiB, 128 * V41DSparkGiB, 153 * V41DSparkGiB, false},
	}
	for _, tc := range cases {
		r := V41DSparkResidency{DraftBytes: tc.draft, TargetBytes: tc.target, HostBudgetBytes: tc.budget}
		if got := r.ResidentBytes(); got != tc.resident {
			t.Errorf("ResidentBytes(draft=%d,target=%d) = %d, want %d", tc.draft, tc.target, got, tc.resident)
		}
		if got := r.FitsHost(); got != tc.fits {
			t.Errorf("FitsHost(resident=%d,budget=%d) = %v, want %v", tc.resident, tc.budget, got, tc.fits)
		}
	}
}

func TestV41DSparkResidencyDefaultResidency(t *testing.T) {
	s := DefaultV41DSparkStudy()
	r := s.Residency

	if r.TargetBytes != V41DSparkDefaultTargetBytes {
		t.Errorf("TargetBytes = %d, want V41DSparkDefaultTargetBytes = %d", r.TargetBytes, V41DSparkDefaultTargetBytes)
	}
	if r.TargetBytes != 152*V41DSparkGiB {
		t.Errorf("TargetBytes = %d, want 152 GiB = %d", r.TargetBytes, 152*V41DSparkGiB)
	}
	if r.DraftBytes != V41DSparkDefaultDraftBytes {
		t.Errorf("DraftBytes = %d, want V41DSparkDefaultDraftBytes = %d", r.DraftBytes, V41DSparkDefaultDraftBytes)
	}
	if r.DraftBytes <= 0 {
		t.Errorf("DraftBytes = %d, want > 0", r.DraftBytes)
	}
	if r.DraftBytes <= 7*V41DSparkGiB || r.DraftBytes >= 8*V41DSparkGiB {
		t.Errorf("DraftBytes = %d (~%.2f GiB), want ~7.4 GiB", r.DraftBytes, float64(r.DraftBytes)/float64(V41DSparkGiB))
	}
	if r.NodeCount != V41DSparkDefaultNodeCount {
		t.Errorf("NodeCount = %d, want V41DSparkDefaultNodeCount = %d", r.NodeCount, V41DSparkDefaultNodeCount)
	}
	if r.NodeCount != 2 {
		t.Errorf("NodeCount = %d, want 2", r.NodeCount)
	}
	if !r.DraftResident {
		t.Error("DraftResident = false, want true")
	}
	if !r.TargetStreamed {
		t.Error("TargetStreamed = false, want true")
	}
	if r.ResidentBytes() != r.DraftBytes+r.TargetBytes {
		t.Errorf("ResidentBytes() = %d, want %d", r.ResidentBytes(), r.DraftBytes+r.TargetBytes)
	}
	// 152 GiB target + ~7.4 GiB draft exceeds the 128 GiB host budget.
	if r.FitsHost() {
		t.Errorf("FitsHost() = true for default (resident %d > budget %d), want false",
			r.ResidentBytes(), r.HostBudgetBytes)
	}
}

// --- acceptance ------------------------------------------------------------

func TestV41DSparkResidencyAcceptance(t *testing.T) {
	s := DefaultV41DSparkStudy()

	var code, prose V41DSparkAcceptance
	for _, w := range []V41DSparkWorkload{V41DSparkWorkloadCode, V41DSparkWorkloadProse} {
		a, ok := s.AcceptanceFor(w)
		if !ok {
			t.Fatalf("AcceptanceFor(%q) ok = false, want true", w)
		}
		if a.Workload != w {
			t.Errorf("AcceptanceFor(%q).Workload = %q", w, a.Workload)
		}
		if !(a.DSparkTokPerSec > a.BaselineTokPerSec) {
			t.Errorf("AcceptanceFor(%q): DSpark %.3f <= Baseline %.3f, want DSpark > Baseline",
				w, a.DSparkTokPerSec, a.BaselineTokPerSec)
		}
		if !(a.AcceptanceRate > 0 && a.AcceptanceRate <= 1) {
			t.Errorf("AcceptanceFor(%q).AcceptanceRate = %v, want 0 < rate <= 1", w, a.AcceptanceRate)
		}
		if !(a.AcceptedTokensPerCycle > 0) {
			t.Errorf("AcceptanceFor(%q).AcceptedTokensPerCycle = %v, want > 0", w, a.AcceptedTokensPerCycle)
		}
		if !a.ByteIdentical {
			t.Errorf("AcceptanceFor(%q).ByteIdentical = false, want true", w)
		}
		if a.EvidenceClass != V41DSparkEvidenceUpstreamReported {
			t.Errorf("AcceptanceFor(%q).EvidenceClass = %q, want %q", w, a.EvidenceClass, V41DSparkEvidenceUpstreamReported)
		}
		switch w {
		case V41DSparkWorkloadCode:
			code = a
		case V41DSparkWorkloadProse:
			prose = a
		}
	}

	codeGain := code.DSparkTokPerSec - code.BaselineTokPerSec
	proseGain := prose.DSparkTokPerSec - prose.BaselineTokPerSec
	if !(codeGain > proseGain) {
		t.Errorf("code gain %.3f <= prose gain %.3f, want code gain strictly greater (core ticket finding)",
			codeGain, proseGain)
	}

	if _, ok := s.AcceptanceFor(V41DSparkWorkload("unknown")); ok {
		t.Error("AcceptanceFor(unknown) ok = true, want false")
	}
	if _, ok := s.AcceptanceFor(""); ok {
		t.Error(`AcceptanceFor("") ok = true, want false`)
	}
}

// --- Validate error table --------------------------------------------------

func TestV41DSparkResidencyValidateErrors(t *testing.T) {
	cases := []struct {
		name    string
		mut     func(s *V41DSparkStudy)
		wantErr bool
	}{
		{"unmodified default", func(s *V41DSparkStudy) {}, false},
		{"code row removed", func(s *V41DSparkStudy) {
			*s = v41DSparkDropWorkload(*s, V41DSparkWorkloadCode)
		}, true},
		{"prose row removed", func(s *V41DSparkStudy) {
			*s = v41DSparkDropWorkload(*s, V41DSparkWorkloadProse)
		}, true},
		{"baseline tok/s zero", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].BaselineTokPerSec = 0
		}, true},
		{"baseline tok/s NaN", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].BaselineTokPerSec = math.NaN()
		}, true},
		{"dspark tok/s zero", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].DSparkTokPerSec = 0
		}, true},
		{"dspark tok/s NaN", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].DSparkTokPerSec = math.NaN()
		}, true},
		{"accepted tokens/cycle zero", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].AcceptedTokensPerCycle = 0
		}, true},
		{"accepted tokens/cycle NaN", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].AcceptedTokensPerCycle = math.NaN()
		}, true},
		{"acceptance rate zero", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].AcceptanceRate = 0
		}, true},
		{"acceptance rate NaN", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].AcceptanceRate = math.NaN()
		}, true},
		{"acceptance rate negative", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].AcceptanceRate = -0.5
		}, true},
		{"acceptance rate > 1", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].AcceptanceRate = 1.01
		}, true},
		{"dspark <= baseline (equal)", func(s *V41DSparkStudy) {
			i := v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)
			s.Acceptance[i].DSparkTokPerSec = s.Acceptance[i].BaselineTokPerSec
		}, true},
		{"dspark <= baseline (below)", func(s *V41DSparkStudy) {
			i := v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)
			s.Acceptance[i].DSparkTokPerSec = s.Acceptance[i].BaselineTokPerSec / 2
		}, true},
		{"missing evidence class", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].EvidenceClass = ""
		}, true},
		{"wrong evidence class", func(s *V41DSparkStudy) {
			s.Acceptance[v41DSparkFindAcceptance(*s, V41DSparkWorkloadCode)].EvidenceClass = "fak-measured"
		}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := DefaultV41DSparkStudy()
			tc.mut(&s)
			err := s.Validate()
			if tc.wantErr && err == nil {
				t.Errorf("Validate() = nil, want error")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}

	t.Run("zero-value study", func(t *testing.T) {
		if err := (V41DSparkStudy{}).Validate(); err == nil {
			t.Error("V41DSparkStudy{}.Validate() = nil, want error (fail-closed)")
		}
	})
}

// --- resonance claim admission --------------------------------------------

func TestV41DSparkResidencyAdmitClaim(t *testing.T) {
	defaultStudy := DefaultV41DSparkStudy()
	if defaultStudy.Residency.FitsHost() {
		t.Fatalf("precondition: default study unexpectedly fits host (resident %d, budget %d)",
			defaultStudy.Residency.ResidentBytes(), defaultStudy.Residency.HostBudgetBytes)
	}

	t.Run("nodeCount zero", func(t *testing.T) {
		if err := defaultStudy.AdmitResidentClaim(0, false); err == nil {
			t.Error("AdmitResidentClaim(0, false) = nil, want error")
		}
	})
	t.Run("nodeCount negative", func(t *testing.T) {
		if err := defaultStudy.AdmitResidentClaim(-1, false); err == nil {
			t.Error("AdmitResidentClaim(-1, false) = nil, want error")
		}
	})
	t.Run("nodeCount 1, does not fit, load failed", func(t *testing.T) {
		err := defaultStudy.AdmitResidentClaim(1, false)
		if err == nil {
			t.Fatal("AdmitResidentClaim(1, false) = nil, want ErrV41DSparkResidencyOverclaim")
		}
		if !errors.Is(err, ErrV41DSparkResidencyOverclaim) {
			t.Errorf("AdmitResidentClaim(1, false) = %v, want errors.Is(_, ErrV41DSparkResidencyOverclaim)", err)
		}
	})
	t.Run("nodeCount 1, load passed", func(t *testing.T) {
		if err := defaultStudy.AdmitResidentClaim(1, true); err != nil {
			t.Errorf("AdmitResidentClaim(1, true) = %v, want nil", err)
		}
	})
	t.Run("nodeCount 2", func(t *testing.T) {
		if err := defaultStudy.AdmitResidentClaim(2, false); err != nil {
			t.Errorf("AdmitResidentClaim(2, false) = %v, want nil", err)
		}
	})
	t.Run("zero-value study at nodeCount 1", func(t *testing.T) {
		if err := (V41DSparkStudy{}).AdmitResidentClaim(1, false); err == nil {
			t.Error("V41DSparkStudy{}.AdmitResidentClaim(1, false) = nil, want error (must not silently admit)")
		}
		if err := (V41DSparkStudy{}).AdmitResidentClaim(1, true); err == nil {
			t.Error("V41DSparkStudy{}.AdmitResidentClaim(1, true) = nil, want error (must not silently admit)")
		}
	})
	t.Run("zero-value study at nodeCount 2", func(t *testing.T) {
		err := (V41DSparkStudy{}).AdmitResidentClaim(2, false)
		if err == nil {
			t.Fatal("V41DSparkStudy{}.AdmitResidentClaim(2, false) = nil, want ErrV41DSparkResidencyOverclaim")
		}
		if !errors.Is(err, ErrV41DSparkResidencyOverclaim) {
			t.Errorf("V41DSparkStudy{}.AdmitResidentClaim(2, false) = %v, want errors.Is(_, ErrV41DSparkResidencyOverclaim)", err)
		}
	})
	t.Run("zero-value study at nodeCount 5", func(t *testing.T) {
		err := (V41DSparkStudy{}).AdmitResidentClaim(5, false)
		if err == nil {
			t.Fatal("V41DSparkStudy{}.AdmitResidentClaim(5, false) = nil, want ErrV41DSparkResidencyOverclaim")
		}
		if !errors.Is(err, ErrV41DSparkResidencyOverclaim) {
			t.Errorf("V41DSparkStudy{}.AdmitResidentClaim(5, false) = %v, want errors.Is(_, ErrV41DSparkResidencyOverclaim)", err)
		}
	})
	t.Run("plan flags cleared at nodeCount 2", func(t *testing.T) {
		s := defaultStudy
		s.Residency.DraftResident = false
		if err := s.AdmitResidentClaim(2, false); err == nil {
			t.Error("AdmitResidentClaim(2, false) with DraftResident=false = nil, want error")
		}
	})
	t.Run("positive fitting 1-node case", func(t *testing.T) {
		s := V41DSparkStudy{
			Residency: V41DSparkResidency{
				DraftBytes:      1,
				TargetBytes:     2,
				HostBudgetBytes: 3,
				NodeCount:       1,
				DraftResident:   true,
				TargetStreamed:  true,
			},
		}
		if err := s.AdmitResidentClaim(1, false); err != nil {
			t.Errorf("AdmitResidentClaim(1, false) on fitting 1-node plan = %v, want nil", err)
		}
	})
}

// --- evidence class closure ------------------------------------------------

func TestV41DSparkResidencyEvidenceClassClosed(t *testing.T) {
	i := v41DSparkFindAcceptance(DefaultV41DSparkStudy(), V41DSparkWorkloadCode)
	if i < 0 {
		t.Fatal("default study has no code acceptance row")
	}

	s := DefaultV41DSparkStudy()
	s.Acceptance[i].EvidenceClass = ""
	if err := s.Validate(); err == nil {
		t.Error("Validate() with empty EvidenceClass = nil, want error")
	}

	s = DefaultV41DSparkStudy()
	s.Acceptance[i].EvidenceClass = "fak-measured"
	if err := s.Validate(); err == nil {
		t.Errorf("Validate() with EvidenceClass %q = nil, want error", "fak-measured")
	}

	s = DefaultV41DSparkStudy()
	s.Acceptance[i].EvidenceClass = V41DSparkEvidenceUpstreamReported
	if err := s.Validate(); err != nil {
		t.Errorf("Validate() with EvidenceClass %q = %v, want nil", V41DSparkEvidenceUpstreamReported, err)
	}
}

// --- exact ticket figures --------------------------------------------------

func TestV41DSparkResidencyAcceptanceConstants(t *testing.T) {
	s := DefaultV41DSparkStudy()

	code, ok := s.AcceptanceFor(V41DSparkWorkloadCode)
	if !ok {
		t.Fatal("AcceptanceFor(code) ok = false, want true")
	}
	if code.BaselineTokPerSec != 13.8 {
		t.Fatalf("code BaselineTokPerSec = %v, want 13.8", code.BaselineTokPerSec)
	}
	if code.DSparkTokPerSec != 31.7 {
		t.Fatalf("code DSparkTokPerSec = %v, want 31.7", code.DSparkTokPerSec)
	}
	if code.AcceptedTokensPerCycle != 3.6 {
		t.Fatalf("code AcceptedTokensPerCycle = %v, want 3.6", code.AcceptedTokensPerCycle)
	}
	if code.AcceptanceRate != 0.975 {
		t.Fatalf("code AcceptanceRate = %v, want 0.975", code.AcceptanceRate)
	}

	prose, ok := s.AcceptanceFor(V41DSparkWorkloadProse)
	if !ok {
		t.Fatal("AcceptanceFor(prose) ok = false, want true")
	}
	if prose.BaselineTokPerSec != 13.4 {
		t.Fatalf("prose BaselineTokPerSec = %v, want 13.4", prose.BaselineTokPerSec)
	}
	if prose.DSparkTokPerSec != 16.6 {
		t.Fatalf("prose DSparkTokPerSec = %v, want 16.6", prose.DSparkTokPerSec)
	}
	if prose.AcceptedTokensPerCycle != 1.5 {
		t.Fatalf("prose AcceptedTokensPerCycle = %v, want 1.5", prose.AcceptedTokensPerCycle)
	}
	if prose.AcceptanceRate != 0.55 {
		t.Fatalf("prose AcceptanceRate = %v, want 0.55", prose.AcceptanceRate)
	}
}

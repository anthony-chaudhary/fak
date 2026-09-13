package macbench

import "testing"

func TestGradeSLO_NoFloorsReturnsNil(t *testing.T) {
	if v := GradeSLO(nil, 0, 0); v != nil {
		t.Fatalf("want nil verdict when both floors unset, got %+v", v)
	}
}

func TestGradeSLO_BestRowGradedAgainstFloors(t *testing.T) {
	rows := []Row{
		{Name: "prefill-512", PrefillTokensPerSecond: 31.5, TokensPerSecond: 0},
		{Name: "decode-64", PrefillTokensPerSecond: 0, TokensPerSecond: 7.4},
	}
	v := GradeSLO(rows, 30, 7)
	if v == nil {
		t.Fatal("want verdict")
	}
	if !v.OK || !v.PrefillOK || !v.DecodeOK {
		t.Fatalf("want all OK, got %+v", v)
	}
	if v.BestPrefillTPS != 31.5 || v.BestDecodeTPS != 7.4 {
		t.Fatalf("best not folded: %+v", v)
	}
}

func TestGradeSLO_MissFailsAndIgnoresErroredRows(t *testing.T) {
	rows := []Row{
		{Name: "prefill-ok", PrefillTokensPerSecond: 45, TokensPerSecond: 8},
		{Name: "decode-error", TokensPerSecond: 99, Error: "boom"},
		{Name: "prefill-http500", PrefillTokensPerSecond: 99, HTTPStatus: 500},
	}
	v := GradeSLO(rows, 30, 7)
	if !v.OK {
		t.Fatalf("expected OK from the good row, got %+v", v)
	}
	if v.BestDecodeTPS != 8 {
		t.Fatalf("errored row must not count: got best decode %v", v.BestDecodeTPS)
	}
	miss := GradeSLO([]Row{{Name: "decode", TokensPerSecond: 3}}, 30, 7)
	if miss.OK {
		t.Fatalf("want SLO miss, got %+v", miss)
	}
	// A floor with no qualifying evidence fails closed: the decode-only row
	// yields no prefill observation, so BOTH floors are breached.
	if miss.DecodeOK || miss.PrefillOK {
		t.Fatalf("decode-only run must breach both floors: %+v", miss)
	}
}

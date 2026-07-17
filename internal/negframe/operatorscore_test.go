package negframe

import "testing"

func TestNegationOperatorScoreSignals(t *testing.T) {
	s := MeasureNegationOperator(2.5, 0.125)
	if s.EnumerableDomains == 0 || s.HandledDomains != s.EnumerableDomains || s.DomainCoverage != 1 || s.BenchmarkDelta != 2.5 || s.UnknownFallbackRate != 0.125 {
		t.Fatalf("signals = %+v", s)
	}
	p := BuildNegationOperatorScore(s)
	if !p.OK || p.Corpus["family"] != "Cards" || p.Corpus["sentinel"] != "deferred" {
		t.Fatalf("payload = %+v", p)
	}
}

package tokenizer

import "testing"

func TestCompareLocalUsesSameCorpusAndMatchesNaive(t *testing.T) {
	tok := loadFixtureTokenizer(t)
	report := CompareLocal(tok, ComparisonCorpus())
	if report.Schema != ComparisonSchema || report.Corpus != len(ComparisonCorpus()) || report.Complete {
		t.Fatalf("report=%+v", report)
	}
	if len(report.Arms) != 4 || len(report.Pending) == 0 || report.CorpusDigest == "" {
		t.Fatalf("report=%+v", report)
	}
	for _, arm := range report.Arms[:2] {
		if !arm.Available || arm.ExactMatches != report.Corpus || arm.RoundTripMatch != report.Corpus {
			t.Fatalf("arm=%+v", arm)
		}
	}
	if report.Arms[0].Tokens != report.Arms[1].Tokens {
		t.Fatalf("native tokens=%d naive=%d", report.Arms[0].Tokens, report.Arms[1].Tokens)
	}
	for _, arm := range report.Arms[2:] {
		if arm.Available || arm.Integration == "" {
			t.Fatalf("external arm must stay explicit and unavailable without live executable: %+v", arm)
		}
	}
}

func TestEncodeNaiveMatchesProductionOnMergeFixture(t *testing.T) {
	tok, err := ParseJSON([]byte(`{
	  "model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},"merges":["a b","ab c"]},
	  "decoder":{"type":"ByteLevel"}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	want, err := tok.Encode("abc")
	if err != nil {
		t.Fatal(err)
	}
	got, err := encodeNaive(tok, "abc")
	if err != nil {
		t.Fatal(err)
	}
	if !equalIDs(got, want) {
		t.Fatalf("naive=%v production=%v", got, want)
	}
}

package gateway

import "testing"

func TestResultLivelockObservesExecPollingAndResetsOnProgress(t *testing.T) {
	s := &Server{}
	mk := func(tool, digest string) []ResultAdmission {
		return []ResultAdmission{{Tool: tool, ResultDigest: digest, Verdict: WireVerdict{Kind: "ALLOW"}}}
	}
	var got []ResultAdmission
	for i := 0; i < 9; i++ {
		got = mk("write_stdin", "same-poll")
		s.annotateResultLivelock("codex-poll", got)
	}
	if got[0].Livelock == nil || !got[0].Livelock.Escalate {
		t.Fatalf("repeated poll did not reach abort threshold: %+v", got[0])
	}
	progress := mk("search_kb", "new-output")
	s.annotateResultLivelock("codex-poll", progress)
	again := mk("write_stdin", "same-poll")
	s.annotateResultLivelock("codex-poll", again)
	if again[0].Livelock != nil {
		t.Fatalf("progress did not reset polling run: %+v", again[0])
	}
}

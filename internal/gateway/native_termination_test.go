package gateway

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestNativeTerminationFramesHaveDistinctSafeCauses(t *testing.T) {
	cases := []struct {
		err   error
		cause string
	}{{errors.New("provider status 429: private response"), agent.TerminationRateLimited}, {errors.New("POLICY_BLOCK: private policy"), agent.TerminationRefused}, {errors.New("mystery: private detail"), agent.TerminationUnknown}}
	seen := map[string]bool{}
	for _, tc := range cases {
		frame := nativeTerminationError(tc.err)
		inner, ok := frame["error"].(map[string]any)
		if !ok || inner["type"] != "terminated_run" || inner["message"] == "upstream model error" {
			t.Fatalf("bad frame: %#v", frame)
		}
		term := agent.ClassifyTermination(tc.err)
		if inner["cause"] != term.Cause || inner["evidence"] != term.Evidence {
			t.Fatalf("frame/classifier diverged: %#v %+v", frame, term)
		}
		if term.Cause != tc.cause {
			t.Fatalf("%v => %+v", tc.err, term)
		}
		if term.Evidence == "" || strings.Contains(term.Evidence, "private") {
			t.Fatalf("unsafe evidence: %+v", term)
		}
		seen[term.Cause] = true
	}
	if len(seen) != 3 {
		t.Fatalf("causes collapsed: %v", seen)
	}
}

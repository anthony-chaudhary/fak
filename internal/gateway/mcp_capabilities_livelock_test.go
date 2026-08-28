package gateway

import "testing"

// Repeated fak_capabilities calls are served by the bounded MCP reuse entry, so
// the generic admitted-call detector must not send the model an advisory for a
// side-effect-free operation the kernel has already made cheap. Other admitted
// tools retain the generic detector and still trip on their third identical call.
func TestReusableCapabilityDiscoverySkipsGenericLivelockOnly(t *testing.T) {
	for _, tool := range []string{
		"fak_capabilities",
		"mcp__fak__fak_capabilities",
		"mcp__fak_guard__fak_capabilities",
	} {
		t.Run(tool, func(t *testing.T) {
			s := &Server{}
			discovery := ToolAdjudication{
				ToolCallID: "cap-1",
				Tool:       tool,
				ArgsDigest: "sha256:stable-capabilities-request",
				Admitted:   true,
				Verdict:    WireVerdict{Kind: "ALLOW"},
			}
			for repeat := 1; repeat <= 5; repeat++ {
				adjs := []ToolAdjudication{discovery}
				s.annotateToolLivelock("trace-capabilities", adjs)
				if adjs[0].Livelock != nil {
					t.Fatalf("repeat %d added generic livelock advisory to reusable discovery: %+v", repeat, adjs[0].Livelock)
				}
			}
		})
	}

	s := &Server{}
	stateful := ToolAdjudication{
		ToolCallID: "write-1",
		Tool:       "write_file",
		ArgsDigest: "sha256:stateful-request",
		Admitted:   true,
		Verdict:    WireVerdict{Kind: "ALLOW"},
	}
	for repeat := 1; repeat <= 3; repeat++ {
		adjs := []ToolAdjudication{stateful}
		s.annotateToolLivelock("trace-stateful", adjs)
		if repeat == 1 || repeat == 2 {
			// Reusable discovery is invisible to the generic detector; it must
			// neither increment nor clear an in-progress stateful repeat run.
			discovery := []ToolAdjudication{{
				Tool:       "mcp__fak__fak_capabilities",
				ArgsDigest: "sha256:stable-capabilities-request",
				Admitted:   true,
				Verdict:    WireVerdict{Kind: "ALLOW"},
			}}
			s.annotateToolLivelock("trace-stateful", discovery)
		}
		if repeat < 3 && adjs[0].Livelock != nil {
			t.Fatalf("stateful repeat %d fired livelock early: %+v", repeat, adjs[0].Livelock)
		}
		if repeat == 3 && adjs[0].Livelock == nil {
			t.Fatal("stateful third repeat lost generic livelock handling")
		}
	}
}

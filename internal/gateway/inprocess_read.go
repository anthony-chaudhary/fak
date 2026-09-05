package gateway

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// PromoteShellReadToInProcess attempts to execute a shell command in-process if it represents
// an effect-free read operation (cat, head, tail), avoiding subprocess shell spawning (#11035).
// Returns the structured ResultEnvelope, WireVerdict, and true if promoted; otherwise false.
func (s *Server) PromoteShellReadToInProcess(ctx context.Context, tool, rawArgs, traceID string) (*ResultEnvelope, WireVerdict, bool) {
	if !vdso.IsPromotableShellTool(tool) {
		return nil, WireVerdict{}, false
	}

	tc, err := s.buildCall(ctx, tool, rawArgs, true, "", traceID)
	if err != nil {
		return nil, WireVerdict{}, false
	}

	res, ok := vdso.PromoteInProcessRead(tc, "")
	if !ok || res == nil {
		return nil, WireVerdict{}, false
	}

	wv := renderVerdict(abi.Verdict{Kind: abi.VerdictAllow}, res.Meta)
	env := &ResultEnvelope{
		Status:  statusName(res.Status),
		Content: string(resolveBytes(ctx, res.Payload)),
		Meta:    res.Meta,
	}
	s.rememberOriginSeq(tc.TraceID, tc.Tool, string(resolveBytes(ctx, tc.Args)), tc.SeqNo)
	return env, wv, true
}

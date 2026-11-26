package agent

import (
	"context"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type inKernelPromptPanelHeadroomKey struct{}

// promptPanelHeadroomGrant carries one request's admitted prompt-panel headroom to the
// sessions it configures. It is a shared pointer so an OOM retry can revoke the grant for
// every later configure on the same request context.
type promptPanelHeadroomGrant struct{ bytes atomic.Int64 }

func withPromptPanelHeadroom(ctx context.Context, bytes int64) context.Context {
	if bytes <= 0 {
		return ctx
	}
	g := &promptPanelHeadroomGrant{}
	g.bytes.Store(bytes)
	return context.WithValue(ctx, inKernelPromptPanelHeadroomKey{}, g)
}

func promptPanelHeadroomBytes(ctx context.Context) int64 {
	if ctx == nil {
		return 0
	}
	if g, _ := ctx.Value(inKernelPromptPanelHeadroomKey{}).(*promptPanelHeadroomGrant); g != nil {
		return g.bytes.Load()
	}
	return 0
}

// revokePromptPanelHeadroom drops the grant so a retry after a device OOM prefills with the
// transient-only chunk the base plan prices.
func revokePromptPanelHeadroom(ctx context.Context) {
	if ctx == nil {
		return
	}
	if g, _ := ctx.Value(inKernelPromptPanelHeadroomKey{}).(*promptPanelHeadroomGrant); g != nil {
		g.bytes.Store(0)
	}
}

// applyPromptPanelHeadroom sets the session's panel headroom from the request's grant. It
// always writes, so a session reused across requests never keeps a previous request's grant.
func applyPromptPanelHeadroom(ctx context.Context, s *model.Session) {
	if s != nil {
		s.PromptPanelHeadroomBytes = promptPanelHeadroomBytes(ctx)
	}
}

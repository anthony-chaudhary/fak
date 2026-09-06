package mcpbroker

import "context"

type identityContextKey struct{}

// IdentityContext represents a typed request-context identity signal indicating
// that transport-level structured compression must be bypassed to preserve exact original bytes.
type IdentityContext struct {
	// Identity specifies whether exact identity bytes are requested.
	Identity bool
}

// WithIdentityContext derives a context carrying the typed request-context identity signal.
//
// Invariant: Identity can narrow only; auto never overrides inherited or operator identity.
// If the parent context already carries an identity signal, inherited identity is preserved.
func WithIdentityContext(ctx context.Context, id ...IdentityContext) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	// Monotonicity: inherited identity remains identity.
	if HasIdentityContext(ctx) {
		return ctx
	}
	signal := IdentityContext{Identity: true}
	if len(id) > 0 {
		signal = id[0]
	}
	if !signal.Identity {
		// Attempting to request non-identity on context cannot override inherited identity.
		return ctx
	}
	ctx = context.WithValue(ctx, identityContextKey{}, signal)
	ctx = context.WithValue(ctx, compressionContextKey{}, true)
	ctx = context.WithValue(ctx, compressionPolicyContextKey{}, CompressionIdentity)
	return ctx
}

// WithIdentity is an alias for WithIdentityContext for callers requesting identity output.
func WithIdentity(ctx context.Context) context.Context {
	return WithIdentityContext(ctx)
}

// WithCompressionIdentity is an alias for WithIdentityContext.
func WithCompressionIdentity(ctx context.Context) context.Context {
	return WithIdentityContext(ctx)
}

// HasIdentityContext reports whether the context carries a typed identity signal
// (or an inherited identity policy) requesting exact original bytes.
func HasIdentityContext(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	if sig, ok := ctx.Value(identityContextKey{}).(IdentityContext); ok && sig.Identity {
		return true
	}
	if v, ok := ctx.Value(compressionContextKey{}).(bool); ok && v {
		return true
	}
	if p, ok := CompressionPolicyFromContext(ctx); ok && IsCompressionOptOut(string(p)) {
		return true
	}
	return false
}

// IsIdentityContext is an alias for HasIdentityContext.
func IsIdentityContext(ctx context.Context) bool {
	return HasIdentityContext(ctx)
}

// IdentityContextFromContext extracts the IdentityContext signal from ctx, if present.
func IdentityContextFromContext(ctx context.Context) (IdentityContext, bool) {
	if ctx == nil {
		return IdentityContext{}, false
	}
	if sig, ok := ctx.Value(identityContextKey{}).(IdentityContext); ok {
		return sig, true
	}
	if HasIdentityContext(ctx) {
		return IdentityContext{Identity: true}, true
	}
	return IdentityContext{}, false
}

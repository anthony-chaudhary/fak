package agent

import (
	"context"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

type prefixCacheIdentityKey struct{}

// WithPrefixCacheIdentity binds an authenticated cache owner to a planner call.
// Gateway transports should set tenant from their isolation principal. Agent may
// be empty when the caller has tenant-level rather than worker-level identity.
func WithPrefixCacheIdentity(ctx context.Context, tenant, agent string) context.Context {
	owner := radixkv.CacheIdentity{Tenant: strings.TrimSpace(tenant), Agent: strings.TrimSpace(agent)}
	if owner.Tenant == "" {
		return ctx
	}
	return context.WithValue(ctx, prefixCacheIdentityKey{}, owner)
}

func prefixCacheIdentityFromContext(ctx context.Context) (radixkv.CacheIdentity, bool) {
	if ctx == nil {
		return radixkv.CacheIdentity{}, false
	}
	owner, ok := ctx.Value(prefixCacheIdentityKey{}).(radixkv.CacheIdentity)
	return owner, ok && owner.Tenant != ""
}

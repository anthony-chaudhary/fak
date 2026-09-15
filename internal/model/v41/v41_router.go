package v41

import "math"

// This is the build-independent home for the pure V4.1 routed+shared geometry
// constants and the overflow-safe scoring function (issue #13095, parent
// #13038). The core-resident v41_router.go keeps an identical copy until its
// consumers (v41Route / v41SharedExpertAdd, which return the core routePick and
// v4RouteError types) move here with the rest of the router hub; the two are
// kept byte-equivalent (package clause aside).

const (
	V41RouterExperts     = 384
	V41RouterTopK        = 6
	V41RouterSharedCount = 1
	V41RouterMoEWidth    = 2304
	V41RouterRouteScale  = 1.5
)

// V41RouterConfig is the admitted V4.1 routed+shared geometry.
type V41RouterConfig struct {
	Experts     int
	TopK        int
	SharedCount int
	RouteScale  float32
}

// V41DefaultRouterConfig returns the published V4.1 geometry.
func V41DefaultRouterConfig() V41RouterConfig {
	return V41RouterConfig{
		Experts:     V41RouterExperts,
		TopK:        V41RouterTopK,
		SharedCount: V41RouterSharedCount,
		RouteScale:  V41RouterRouteScale,
	}
}

// V41SqrtSoftplus is the overflow-safe sqrt(softplus(z)) scoring function.
func V41SqrtSoftplus(z float32) float32 {
	zf := float64(z)
	return float32(math.Sqrt(math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))))
}

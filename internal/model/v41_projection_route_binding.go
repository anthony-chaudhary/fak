package model

// v41ProjectionRouteBinding is an immutable pair of ordinary callbacks. A
// replacement creates a new pair, so state and scratch copies cannot observe
// another owner's callback replacement. It carries no device tensors or native
// execution capability and is omitted from continuation snapshots.
type v41ProjectionRouteBinding struct {
	projection v41DenseProjectionFunc
	router     v41RouterTopKFunc
}

func newV41ProjectionRouteBinding(projection v41DenseProjectionFunc, router v41RouterTopKFunc) *v41ProjectionRouteBinding {
	if projection == nil && router == nil {
		return nil
	}
	return &v41ProjectionRouteBinding{projection: projection, router: router}
}

func (b *v41ProjectionRouteBinding) projectionFunc() v41DenseProjectionFunc {
	if b == nil {
		return nil
	}
	return b.projection
}

func (b *v41ProjectionRouteBinding) routerFunc() v41RouterTopKFunc {
	if b == nil {
		return nil
	}
	return b.router
}

// Replacements retain only ordinary callbacks. Optional execution capabilities
// must never be inherited by an arbitrary wrapper around an old callback.
func (b *v41ProjectionRouteBinding) withProjection(projection v41DenseProjectionFunc) *v41ProjectionRouteBinding {
	return newV41ProjectionRouteBinding(projection, b.routerFunc())
}

func (b *v41ProjectionRouteBinding) withRouter(router v41RouterTopKFunc) *v41ProjectionRouteBinding {
	return newV41ProjectionRouteBinding(b.projectionFunc(), router)
}

func (st *v41ForwardState) denseProjection() v41DenseProjectionFunc {
	return st.projectionRoute.projectionFunc()
}

func (st *v41ForwardState) routerTopK() v41RouterTopKFunc {
	return st.projectionRoute.routerFunc()
}

func (st *v41ForwardState) setDenseProjection(projection v41DenseProjectionFunc) {
	st.projectionRoute = st.projectionRoute.withProjection(projection)
}

func (st *v41ForwardState) setRouterTopK(router v41RouterTopKFunc) {
	st.projectionRoute = st.projectionRoute.withRouter(router)
}

func (scratch *v41ProjScratch) denseProjection() v41DenseProjectionFunc {
	return scratch.projectionRoute.projectionFunc()
}

func (scratch *v41ProjScratch) routerTopK() v41RouterTopKFunc {
	return scratch.projectionRoute.routerFunc()
}

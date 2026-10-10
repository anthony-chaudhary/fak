package model

import (
	"errors"
	"fmt"
)

var errV41DeviceOnlyHostOp = errors.New("model: V4.1 forward op runs on host under device-only policy")

// v41HostOp is one V4.1 forward op with no device dispatch yet. site names the
// host function that computes it, so a refusal points at the next slice.
type v41HostOp struct {
	name string
	site string
}

// v41DeviceOnlyHostOpError names every V4.1 forward op still on the host, in
// forward order.
type v41DeviceOnlyHostOpError struct {
	ops []v41HostOp
}

func (e *v41DeviceOnlyHostOpError) Error() string {
	if e == nil || len(e.ops) == 0 {
		return errV41DeviceOnlyHostOp.Error()
	}
	return fmt.Sprintf("%v: %s (%s) and %d more", errV41DeviceOnlyHostOp, e.ops[0].name, e.ops[0].site, len(e.ops)-1)
}

func (e *v41DeviceOnlyHostOpError) Is(target error) bool { return target == errV41DeviceOnlyHostOp }

// v41DeviceOnlyHostOps is the closed inventory the device-only gate refuses on.
// Remove an entry only in the change that moves that op onto the device seam;
// an empty inventory is the V4.1 admission condition.
var v41DeviceOnlyHostOps = []v41HostOp{
	{name: "embedding-panel", site: "Model.v41EmbeddingPanel"},
	{name: "mhc-initial-streams", site: "v41FullInitialStreams"},
	{name: "mhc-sinkhorn-split", site: "v41MHCSplit"},
	{name: "mhc-pre-collapse", site: "v41MHCPre"},
	{name: "attention-index-list", site: "Model.v41AttentionIndexList"},
	{name: "attention-output-rotation", site: "v41AttentionOutputForProjection"},
	{name: "moe-router-topk", site: "v41RouteForGeometry"},
	{name: "routed-expert-weighted-sum", site: "v41FullRoutedSum"},
	{name: "shared-expert-add", site: "v41SharedExpertAdd"},
	{name: "mhc-post-mix", site: "v41MHCPostBF16"},
	{name: "residual-stream-carry", site: "Model.v41Layer"},
}

func v41DeviceOnlyRefusal() error {
	if len(v41DeviceOnlyHostOps) == 0 {
		return nil
	}
	return &v41DeviceOnlyHostOpError{ops: append([]v41HostOp(nil), v41DeviceOnlyHostOps...)}
}

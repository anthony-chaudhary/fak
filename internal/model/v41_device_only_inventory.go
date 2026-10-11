package model

import (
	"errors"
	"fmt"
	"strings"
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
	names := make([]string, len(e.ops))
	for i, op := range e.ops {
		names[i] = op.name + " (" + op.site + ")"
	}
	return fmt.Sprintf("%v: %s", errV41DeviceOnlyHostOp, strings.Join(names, ", "))
}

func (e *v41DeviceOnlyHostOpError) Is(target error) bool { return target == errV41DeviceOnlyHostOp }

// v41DeviceOnlyHostOps is the closed inventory the device-only gate refuses on.
// Remove an entry only in the change that moves that op onto the device seam;
// an empty inventory is the V4.1 admission condition.
var v41DeviceOnlyHostOps = []v41HostOp{
	{name: "engram-stream-injection", site: "Model.v41EngramInjectPrepared"},
	{name: "attention-input-norm", site: "Model.v41AttentionInputNormWithLeaf"},
	{name: "compressor-gated-pool", site: "V41CompressorPool.PushNormalized"},
	{name: "live-expert-shared-add", site: "Session.v4ExpertForward"},
}

func v41DeviceOnlyRefusal() error {
	if len(v41DeviceOnlyHostOps) == 0 {
		return nil
	}
	return &v41DeviceOnlyHostOpError{ops: append([]v41HostOp(nil), v41DeviceOnlyHostOps...)}
}

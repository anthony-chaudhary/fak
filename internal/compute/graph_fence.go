package compute

// fcuda_graph_end_launch return codes (FCUDA_GRAPH_* in cuda_backend.h).
const (
	graphEndCaptureFailed  = 1
	graphInstantiateFailed = 2
	graphLaunchFailed      = 3
	graphFenceFailed       = 4
)

// Graph launch sites carried by the typed error, so a caller can tell a capture or launch
// rejection from an asynchronous fault that appeared at the replay fence.
const (
	GraphLaunchSite = "cuda-graph-launch"
	GraphFenceSite  = "cuda-graph-fence"
)

// graphEndLaunchError turns a nonzero fcuda_graph_end_launch code into the typed launch
// error GraphEndLaunch raises.
func graphEndLaunchError(code int) *CUDALaunchError {
	site, msg := GraphLaunchSite, "cuda graph capture/launch failed"
	if code == graphFenceFailed {
		site, msg = GraphFenceSite, "cuda graph replay faulted at its fence synchronize"
	}
	return &CUDALaunchError{CUDAOpError: CUDAOpError{Op: "GraphEndLaunch", Site: site, Msg: msg, Code: code}}
}

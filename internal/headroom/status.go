package headroom

import "context"

// PluginStatus is the operator-facing health/capability row for one context-compression
// plugin. It uses the same small vocabulary as the cache ablation report so later cache
// roll-ups can join "what saved tokens?" with "which subcomponent was active, lossless,
// recoverable, unavailable, or a deliberate no-op?"
type PluginStatus struct {
	Name         string `json:"name"`
	Selected     bool   `json:"selected"`
	Owner        string `json:"owner"`
	Dependency   string `json:"dependency"`
	Fidelity     string `json:"fidelity"`
	Evidence     string `json:"evidence"`
	Status       string `json:"status"`
	Reachability string `json:"reachability"`
	Reason       string `json:"reason"`
}

// StatusReport is the machine-readable `fak headroom status --json` envelope.
type StatusReport struct {
	Selected          string         `json:"selected"`
	Plugins           []PluginStatus `json:"plugins"`
	HeadroomURL       string         `json:"headroom_url"`
	HeadroomReachable bool           `json:"headroom_reachable"`
	GateStats         Stats          `json:"gate_stats"`
}

// BuildStatus snapshots the registered plugins, selected plugin, external sidecar
// reachability, and process-local gate counters. It performs at most one network probe
// (the headroom sidecar reachability check) and otherwise folds local registry state.
func BuildStatus(ctx context.Context) StatusReport {
	selectedName := Selected().Name()
	names := Names()
	hasHeadroom := false
	for _, name := range names {
		if name == HeadroomName {
			hasHeadroom = true
			break
		}
	}
	headroomReachable := false
	if hasHeadroom {
		headroomReachable = Reachable(ctx)
	}
	rep := StatusReport{
		Selected:          selectedName,
		HeadroomURL:       HeadroomURL(),
		HeadroomReachable: headroomReachable,
		GateStats:         Default.Stats(),
	}
	for _, name := range names {
		rep.Plugins = append(rep.Plugins, pluginStatus(name, name == selectedName, headroomReachable))
	}
	return rep
}

// pluginDescriptors holds the STATIC half of each built-in plugin's status row — the
// ownership/fidelity/evidence vocabulary and the row it reports before any live signal
// is folded in. Only these constants differed between the four cases, so they live as
// data here and pluginStatus keeps just the live overrides.
var pluginDescriptors = map[string]PluginStatus{
	NoopName: {
		Owner:        "fak",
		Dependency:   "none",
		Fidelity:     "no-op",
		Evidence:     "configured",
		Status:       "inactive",
		Reachability: "not_applicable",
		Reason:       "identity compressor registered; selecting it leaves tool results unchanged",
	},
	NativeName: {
		Owner:        "fak",
		Dependency:   "in_process",
		Fidelity:     "recoverable",
		Evidence:     "witnessed",
		Status:       "available",
		Reachability: "not_applicable",
		Reason:       "in-process structural compressor; gate preserves the original bytes when a transform fires",
	},
	HeadroomName: {
		Owner:        "external",
		Dependency:   "external_http_sidecar",
		Fidelity:     "recoverable",
		Evidence:     "observed",
		Status:       "unavailable",
		Reachability: "unreachable",
		Reason:       "headroom sidecar is not reachable; bridge passes original bytes through unchanged",
	},
}

// unknownPluginDescriptor is the static row for a compressor registered by a third
// party: it is present and usable, but carries no first-party capability claims.
var unknownPluginDescriptor = PluginStatus{
	Owner:        "unknown",
	Dependency:   "registered_plugin",
	Fidelity:     "unknown",
	Evidence:     "configured",
	Status:       "available",
	Reachability: "unknown",
	Reason:       "registered compressor has no built-in status descriptor",
}

// pluginStatus folds the live signals (is this plugin the selected one, is the external
// sidecar answering) into the plugin's static descriptor. Only the plugins whose live
// wording differs from "selected ⇒ active" carry a case of their own.
func pluginStatus(name string, selected, headroomReachable bool) PluginStatus {
	st, ok := pluginDescriptors[name]
	if !ok {
		st = unknownPluginDescriptor
	}
	st.Name = name
	st.Selected = selected
	switch name {
	case NoopName:
		if selected {
			st.Status = "no-op"
			st.Reason = "identity compressor selected; context compression is intentionally off"
		}
	case HeadroomName:
		if headroomReachable {
			st.Status = "available"
			st.Reachability = "reachable"
			st.Reason = "headroom sidecar responded to the status probe; compression results remain per-call observed outputs with CCR handles when supplied"
			if selected {
				st.Status = "active"
			}
		}
	default:
		if selected {
			st.Status = "active"
		}
	}
	return st
}

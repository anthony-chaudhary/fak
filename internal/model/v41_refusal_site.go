package model

import "errors"

// V41RefusalSite locates a typed V4.1 forward refusal so a caller (planner
// recovery, warmup readiness) can report where it stopped without parsing prose.
// Kind is the closed error class; Stage and Layer come from the outermost typed
// error that carries them, refined by a wrapped stage error. Layer is -1 for a
// model-global or unlocated refusal.
type V41RefusalSite struct {
	Kind  string `json:"kind"`
	Stage string `json:"stage,omitempty"`
	Layer int    `json:"layer"`
}

// V41RefusalSiteOf reports the site of a typed V4.1 forward refusal anywhere in
// err's chain: a stage refusal, the prefill-latency refusal, a backend forward
// operation, or a selected V4.1 operation error. ok is false for any other error.
func V41RefusalSiteOf(err error) (site V41RefusalSite, ok bool) {
	site.Layer = -1
	located := false
	for e := err; e != nil; e = errors.Unwrap(e) {
		kind, stage, layer, typed := v41RefusalFrame(e)
		if !typed {
			continue
		}
		if !ok {
			site.Kind, ok = kind, true
		}
		if site.Stage == "" {
			site.Stage = stage
		}
		if !located && layer >= 0 {
			site.Layer, located = layer, true
		}
	}
	if !ok && errors.Is(err, ErrV41PrefillLatency) {
		return V41RefusalSite{Kind: "prefill_latency", Layer: -1}, true
	}
	return site, ok
}

func v41RefusalFrame(e error) (kind, stage string, layer int, ok bool) {
	switch t := e.(type) {
	case *V41ForwardError:
		return "stage", string(t.Stage), t.Layer, true
	case *BackendForwardOperationError:
		return "backend_forward", t.Stage, t.Layer, true
	case *V41ProjectionOperationError:
		return "projection", t.Stage, t.Layer, true
	case *V41ExpertOperationError:
		return "expert", t.Stage, t.Layer, true
	case *V41QueryPrefixOperationError:
		return "query_prefix", t.Stage, t.Layer, true
	case *V41MHCOperationError:
		return "mhc", t.Op, t.Layer, true
	case *V41BF16BoundaryOperationError:
		return "bf16_boundary", t.Site, t.Layer, true
	case *V41TailRoPEOperationError:
		return "tail_rope", "", t.Layer, true
	case *V41IndexWeightScaleOperationError:
		return "index_weight_scale", "", t.Layer, true
	case *V41IndexerScoreOperationError:
		return "indexer_score", "", t.Layer, true
	case *V41SharedAttentionOperationError:
		return "shared_attention", "", t.Layer, true
	case *V41LogitScaleOperationError:
		return "logit_scale", "", -1, true
	case *V41DecoderError:
		return "decoder", string(t.Stage), -1, true
	case *V41EngramStreamError:
		return "engram_stream", string(t.Kind), -1, true
	case *V41PrefillLatencyError:
		return "prefill_latency", "", -1, true
	}
	return "", "", -1, false
}

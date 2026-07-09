package model

import "testing"

// tensor_resolver_v4_test.go — witnesses the DeepSeek-V4 resolver family routing
// (V4 attention seam map, Missing seam #8): a V4 checkpoint routes to its OWN
// "deepseek-v4" family spec (distinct from the single-plane "deepseek-mla" of
// V2/V3), by either the arch key naming v4 or a well-formed CSA/HCA compression
// declaration. The scaffold is globals-only, exactly like deepSeekMLASpec — the
// per-layer two-plane / dual-indexer names are gated on a real checkpoint and NOT
// asserted here.
//
// v4Globals is the shared (non-layer) tensor set baseGlobals() requires; norm.bias
// and lm_head are optional, so the two mandatory globals are enough to resolve a
// NumLayers:0 scaffold.
func v4Globals() map[string]tensorMeta {
	return manifestKeys("model.embed_tokens.weight", "model.norm.weight")
}

// TestResolverDeepSeekV4RoutesByArchKey: a model_type naming deepseek_v4 routes to
// the distinct "deepseek-v4" family and its globals resolve.
func TestResolverDeepSeekV4RoutesByArchKey(t *testing.T) {
	res, err := ResolveTensorNames(Config{NumLayers: 0, ModelType: "deepseek_v4"}, v4Globals())
	if err != nil {
		t.Fatalf("deepseek-v4 scaffold globals must resolve: %v", err)
	}
	if res.Family != "deepseek-v4" {
		t.Fatalf("family = %q, want deepseek-v4", res.Family)
	}
}

// TestResolverDeepSeekV4RoutesByCompressionRates isolates the accessor-driven
// signal: a generic "deepseek" arch key (no v4 in the name) carrying a well-formed
// CSA/HCA declaration still routes to the V4 family. This ties seam #8's routing
// to seam #7's validated rates.
func TestResolverDeepSeekV4RoutesByCompressionRates(t *testing.T) {
	cfg := Config{NumLayers: 0, ModelType: "deepseek", CSACompressionRate: 4, HCACompressionRate: 128}
	res, err := ResolveTensorNames(cfg, v4Globals())
	if err != nil {
		t.Fatalf("deepseek-v4 (rate-signalled) globals must resolve: %v", err)
	}
	if res.Family != "deepseek-v4" {
		t.Fatalf("family = %q, want deepseek-v4 (routed by CSA/HCA rates)", res.Family)
	}
}

// TestResolverDeepSeekV3StaysMLASpec guards the regression boundary: a V2/V3
// checkpoint (no v4 in the name, no compression rates) must still route to the
// single-plane "deepseek-mla" family, unaffected by the new V4 branch.
func TestResolverDeepSeekV3StaysMLASpec(t *testing.T) {
	res, err := ResolveTensorNames(Config{NumLayers: 0, ModelType: "deepseek_v3"}, v4Globals())
	if err != nil {
		t.Fatalf("deepseek-mla scaffold globals must resolve: %v", err)
	}
	if res.Family != "deepseek-mla" {
		t.Fatalf("family = %q, want deepseek-mla (V3 must not capture the V4 branch)", res.Family)
	}
}

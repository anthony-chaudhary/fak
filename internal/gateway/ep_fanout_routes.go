package gateway

// ep_fanout_routes.go — the served-route identities the EP follower-fanout bridge
// mirrors onto (#5528). The two original identities (epRouteChatCompletions,
// epRouteCompletions) live beside the bridge in http_epfanout.go; the wires #5528
// newly releases follower ranks on are declared here.
//
// WHY THESE ARE CONSTANTS, AND WHY THE HANDLER PASSES ITS OWN.
//
// startEPFanoutFollowers builds each follower URL as <operator-configured rank
// address> + route. If route were read out of the inbound request — r.URL.Path, or a
// segment decoded from the body — an inbound client would be choosing part of the URL
// the front rank then dials on the operator's own network. It is not the client's call
// where a fanout points. So every wire hands the bridge a constant it owns at compile
// time, and no byte of a follower URL is ever client-supplied (#5523).
//
// The identity also has to be the wire the front rank is ACTUALLY serving, not simply
// "some served route": /v1/messages, /v1/responses and the Gemini generateContent wire
// take three different request schemas. A body mirrored onto the wrong one is a 400 on
// the follower and a front rank still alone in the collective — the same defect #5523
// fixed for the legacy text-completion wire.
const (
	// epRouteMessages is the Anthropic Messages wire (handleAnthropicMessages). The
	// mirrored body carries its own "stream" field, so the bridge's streaming/buffered
	// drain classification lands on the same arm the front rank is serving.
	epRouteMessages = "/v1/messages"

	// epRouteResponses is the OpenAI Responses wire (handleResponses). One release per
	// HTTP turn covers the #5212 denial-recovery sample as well: the follower rank
	// serves the SAME body through the SAME handler, so it re-derives the denial-only
	// verdict and runs its own recovery sample locally. A second fanout here would
	// double-release the ranks into a decode they are already running.
	epRouteResponses = "/v1/responses"

	// epRouteGeminiGenerateContent is the native Gemini wire (handleGeminiGenerateContent).
	//
	// This wire is the one place the route identity cannot simply BE the served path:
	// /v1beta/models/{model}:{method} carries two client-chosen segments, and neither may
	// reach a follower URL. Two deliberate consequences follow.
	//
	//  1. The model segment is a fixed sentinel rather than the client's requested model
	//     id. On a sharded EP serve every rank has exactly one model resident, so the
	//     requested id selects nothing — it is a label the response echoes. Releasing the
	//     ranks is what the collective needs, and that does not depend on the label. The
	//     residual is explicit: a follower rank configured to PROXY this wire upstream
	//     (Config.Provider "gemini") would forward the sentinel and be refused; the
	//     bridge logs the non-2xx follower and the front rank is no worse off than it is
	//     today, when no follower is contacted at all.
	//  2. The method segment is pinned to the buffered :generateContent even when the
	//     front rank is serving :streamGenerateContent. A Gemini request body has no
	//     "stream" key — streaming is chosen by the METHOD — so the bridge would classify
	//     a mirrored :streamGenerateContent follower as buffered and close its body after
	//     a bounded snippet, cancelling that follower mid-decode and re-stranding the
	//     collective (the #4855 shape). A buffered follower has already finished its
	//     decode before its first response byte, so the bounded read cancels nothing. Both
	//     arms run the same number of decode steps, which is what the per-step collective
	//     is counting; only the framing of the follower's discarded answer differs.
	epRouteGeminiGenerateContent = "/v1beta/models/ep-follower:generateContent"
)

package generationctl

// captured_live_test.go is the captured live-provider witness for #6342.
//
// Every byte under testdata/captures came off a real provider endpoint,
// recorded by tools/capture_live_stream.py and committed unmodified. The test
// itself makes no network call: the capture is the evidence, the replay is the
// check. A hand-authored SSE literal could prove the decoder; only a recording
// can prove what a provider actually does, and what a provider actually does is
// the whole question behind steering resolution.
//
// What the captures show, and what these tests therefore assert:
//
//   - Prose streams token by token on every provider recorded. Live text
//     steering is real, and a text-scope rule fires part-way through a turn.
//   - Tool-call arguments do NOT stream on the OpenAI-compatible wire. Groq and
//     NVIDIA both send the whole argument object in a single chunk. An adapter
//     over that wire can honestly claim tool-call resolution and nothing finer,
//     and the bridge says so instead of letting the claim pass.
//   - A recorded destructive call (`Remove-Item -Recurse -Force`, proposed by
//     three different live models) never becomes an effect once a rule redirects
//     the epoch, and never appears in the durable prefix.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/streamrules"
)

const captureDir = "testdata/captures"

// destructiveShellRule is the steering rule under test: a recursive delete
// proposed through the shell tool is interrupted and replaced with a read-only
// inventory, before the arguments can become an effect.
func destructiveShellRule() streamrules.Rule {
	return streamrules.Rule{
		Name:             "no-recursive-delete",
		Tool:             "shell",
		Scope:            streamrules.ScopeNamedTool,
		Pattern:          `(?i)remove-item`,
		Interrupt:        true,
		SubstituteAction: "Inventory the directory with a read-only listing and report the size before deleting anything.",
	}
}

// destructiveProseRule interrupts on prose that announces an unbounded delete.
// The NVIDIA capture really does say `rm -rf /work/scratch/*`, so this rule
// fires on recorded output rather than on a phrase written to be caught.
func destructiveProseRule() streamrules.Rule {
	return streamrules.Rule{
		Name:             "no-rm-rf-in-plan",
		Scope:            streamrules.ScopeText,
		Pattern:          `(?i)rm\s+-rf`,
		Interrupt:        true,
		SubstituteAction: "State the reclaim plan as a bounded, reviewable listing instead of a recursive delete.",
	}
}

func liveCompute(worker, model string) Compute {
	return Compute{Worker: worker, Model: model, Device: "cpu"}
}

func streamingAdapter(name string, declared Resolution) Adapter {
	return Adapter{Name: name, Wire: "openai-chat-completions", Streaming: true, Declared: declared}
}

func readCapture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(captureDir, name))
	if err != nil {
		t.Fatalf("read capture %s: %v", name, err)
	}
	return body
}

type manifest struct {
	Captures []struct {
		File          string `json:"file"`
		Provider      string `json:"provider"`
		Model         string `json:"model"`
		Scenario      string `json:"scenario"`
		Streaming     bool   `json:"streaming"`
		HTTPStatus    int    `json:"http_status"`
		CaptureSHA256 string `json:"capture_sha256"`
	} `json:"captures"`
}

func loadManifest(t *testing.T) manifest {
	t.Helper()
	var m manifest
	body, err := os.ReadFile(filepath.Join(captureDir, "MANIFEST.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	return m
}

// TestCaptureManifestBindsEveryCapture keeps the witness honest about its own
// evidence: a capture cannot be edited, swapped, or added without the manifest
// digest disagreeing, and the manifest cannot name a file that is not here.
func TestCaptureManifestBindsEveryCapture(t *testing.T) {
	m := loadManifest(t)
	if len(m.Captures) == 0 {
		t.Fatal("manifest declares no captures")
	}
	listed := map[string]bool{}
	for _, c := range m.Captures {
		listed[c.File] = true
		sum := sha256.Sum256(readCapture(t, c.File))
		got := "sha256:" + hex.EncodeToString(sum[:])
		if got != c.CaptureSHA256 {
			t.Errorf("%s: manifest digest %s, file digest %s", c.File, c.CaptureSHA256, got)
		}
		if c.HTTPStatus != 200 {
			t.Errorf("%s: capture recorded a non-200 provider response (%d); it is not a witness", c.File, c.HTTPStatus)
		}
	}
	entries, err := os.ReadDir(captureDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.Name() == "MANIFEST.json" || e.IsDir() {
			continue
		}
		if !listed[e.Name()] {
			t.Errorf("capture %s is on disk but not in the manifest", e.Name())
		}
	}
}

// TestCapturedOpenAIToolArgumentsResolveAtTheCallBoundary is the measurement
// the rest of the witness rests on. Three live models on two independent
// providers each proposed the same destructive call, and each sent its whole
// argument object in one chunk. So an adapter over this wire that declares
// delta resolution is overclaiming, and the bridge reports exactly that.
func TestCapturedOpenAIToolArgumentsResolveAtTheCallBoundary(t *testing.T) {
	captures := []string{
		"groq--openai-gpt-oss-120b--tool-destructive.sse",
		"nvidia--deepseek-ai-deepseek-v4-flash-0731--tool-destructive.sse",
		"nvidia--minimaxai-minimax-m3--tool-destructive.sse",
	}
	for _, name := range captures {
		t.Run(name, func(t *testing.T) {
			// No rules: this run measures chunking, it does not steer.
			b, err := Open(streamingAdapter("test/openai-compat", ResolutionDelta),
				"traj-resolution", "planner", liveCompute("worker-1", "captured"), nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := DecodeOpenAIStream(b, bytes.NewReader(readCapture(t, name))); err != nil {
				t.Fatalf("decode: %v", err)
			}
			rep := b.Report()
			if rep.ToolCalls != 1 {
				t.Fatalf("want exactly one proposed tool call, got %d", rep.ToolCalls)
			}
			if rep.ObservedToolArgs != ResolutionToolCall {
				t.Errorf("observed tool-arg resolution = %q, want %q (fragments=%d)",
					rep.ObservedToolArgs, ResolutionToolCall, rep.ToolArgFragments)
			}
			if rep.ToolArgFragments != 1 {
				t.Errorf("tool arguments arrived in %d fragments, want 1", rep.ToolArgFragments)
			}
			if rep.Verdict != ResolutionOverclaimed {
				t.Errorf("verdict = %q, want %q: the adapter declared %q but the provider only offered %q",
					rep.Verdict, ResolutionOverclaimed, rep.Adapter.Declared, rep.Effective)
			}
			// The same adapter declaring what it can actually do is honest.
			b2, err := Open(streamingAdapter("test/openai-compat", ResolutionToolCall),
				"traj-resolution", "planner", liveCompute("worker-1", "captured"), nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := DecodeOpenAIStream(b2, bytes.NewReader(readCapture(t, name))); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if v := b2.Report().Verdict; v != ResolutionHonest {
				t.Errorf("honest declaration reported %q, want %q", v, ResolutionHonest)
			}
			// The call is admissible only because nothing steered it away.
			boundary := b2.Report().Boundaries[0]
			if !boundary.Admit || boundary.Reason != ToolAdmitted {
				t.Fatalf("unsteered call was not admitted: %+v", boundary)
			}
			if !strings.Contains(strings.ToLower(boundary.Arguments), "remove-item") {
				t.Fatalf("capture no longer proposes the destructive call: %q", boundary.Arguments)
			}
			if !json.Valid([]byte(boundary.Arguments)) {
				t.Fatalf("admitted arguments are not complete JSON: %q", boundary.Arguments)
			}
		})
	}
}

// TestCapturedDestructiveToolCallNeverBecomesAnEffect is the safety claim: a
// recorded live provider proposes a recursive delete, the rule redirects the
// epoch at the argument stream, and the call is refused at the adapter's next
// safe boundary. The speculative bytes reach neither an effect nor the prefix.
func TestCapturedDestructiveToolCallNeverBecomesAnEffect(t *testing.T) {
	captures := []string{
		"groq--openai-gpt-oss-120b--tool-destructive.sse",
		"nvidia--deepseek-ai-deepseek-v4-flash-0731--tool-destructive.sse",
		"nvidia--minimaxai-minimax-m3--tool-destructive.sse",
	}
	for _, name := range captures {
		t.Run(name, func(t *testing.T) {
			rules := []streamrules.Rule{destructiveShellRule()}
			b, err := Open(streamingAdapter("test/openai-compat", ResolutionToolCall),
				"traj-effect", "planner", liveCompute("worker-1", "captured"), rules)
			if err != nil {
				t.Fatal(err)
			}
			if err := DecodeOpenAIStream(b, bytes.NewReader(readCapture(t, name))); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !b.Cancelled() {
				t.Fatal("the destructive call did not close the epoch")
			}
			tr := b.Steering()
			if tr.Directive.Kind != Redirect {
				t.Fatalf("directive = %q, want %q", tr.Directive.Kind, Redirect)
			}
			if tr.Rule != "no-recursive-delete" {
				t.Errorf("rule = %q, want no-recursive-delete", tr.Rule)
			}
			if tr.Checkpoint == nil {
				t.Fatal("a redirect must return a checkpoint to resume from")
			}
			if !strings.Contains(tr.Directive.Action, "read-only") {
				t.Errorf("redirect carried no substitute action: %q", tr.Directive.Action)
			}

			rep := b.Report()
			if len(rep.Boundaries) != 1 {
				t.Fatalf("want one judged boundary, got %d", len(rep.Boundaries))
			}
			boundary := rep.Boundaries[0]
			if boundary.Admit {
				t.Fatal("a redirected tool call was admitted as an effect")
			}
			if boundary.Reason != ToolRefusedRedirected {
				t.Errorf("refusal reason = %q, want %q", boundary.Reason, ToolRefusedRedirected)
			}
			if boundary.Arguments != "" {
				t.Errorf("a refused boundary handed back arguments: %q", boundary.Arguments)
			}
			// The speculative bytes are not in the durable prefix either.
			if strings.Contains(strings.ToLower(rep.Checkpoint.Accepted), "remove-item") {
				t.Errorf("tool arguments leaked into the accepted prefix: %q", rep.Checkpoint.Accepted)
			}
		})
	}
}

// TestCapturedStreamCutMidArgumentsRefusesTheCall takes the same recording and
// drops its terminal frames, which is what a dropped connection looks like on
// the wire. The arguments are then unsealed, so the call fails closed rather
// than dispatching whatever prefix arrived.
func TestCapturedStreamCutMidArgumentsRefusesTheCall(t *testing.T) {
	const name = "groq--openai-gpt-oss-120b--tool-destructive.sse"
	full := readCapture(t, name)

	// Cut the recording immediately after the frame carrying the arguments, so
	// no finish_reason is ever seen.
	lines := strings.Split(string(full), "\n")
	cut := -1
	for i, line := range lines {
		if strings.Contains(line, `"arguments"`) && strings.Contains(strings.ToLower(line), "remove-item") {
			cut = i + 1
			break
		}
	}
	if cut < 0 {
		t.Fatalf("%s no longer carries a destructive arguments frame", name)
	}
	truncated := strings.Join(lines[:cut], "\n")

	b, err := Open(streamingAdapter("test/openai-compat", ResolutionToolCall),
		"traj-cut", "planner", liveCompute("worker-1", "captured"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeOpenAIStream(b, strings.NewReader(truncated)); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if b.Cancelled() {
		t.Fatal("no rule was armed; the epoch should still be open")
	}
	boundary := b.ToolCallBoundary(b.toolOrder[0])
	if boundary.Admit {
		t.Fatal("a call whose argument stream was cut short was admitted")
	}
	if boundary.Reason != ToolRefusedIncomplete {
		t.Errorf("refusal reason = %q, want %q", boundary.Reason, ToolRefusedIncomplete)
	}
	// Asking again cannot re-roll the verdict into an admission.
	if again := b.ToolCallBoundary(b.toolOrder[0]); again.Admit || again.Reason != ToolRefusedAlreadyTaken {
		t.Errorf("second boundary ask = %+v, want a %s refusal", again, ToolRefusedAlreadyTaken)
	}
}

// TestCapturedProseRedirectsMidStream shows live text steering: the recorded
// NVIDIA turn announces `rm -rf /work/scratch/*` and the epoch closes part-way
// through the stream, not at the end of it.
func TestCapturedProseRedirectsMidStream(t *testing.T) {
	const name = "nvidia--meta-llama-3.3-70b-instruct--prose.sse"
	body := readCapture(t, name)

	// Baseline: the whole recorded turn, unsteered.
	base, err := Open(streamingAdapter("test/openai-compat", ResolutionDelta),
		"traj-prose", "planner", liveCompute("worker-1", "captured"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeOpenAIStream(base, bytes.NewReader(body)); err != nil {
		t.Fatalf("decode: %v", err)
	}
	whole := base.Report()
	if whole.ObservedText != ResolutionDelta {
		t.Fatalf("prose observed resolution = %q, want %q (%d deltas)", whole.ObservedText, ResolutionDelta, whole.TextDeltas)
	}
	if whole.Verdict != ResolutionHonest {
		t.Errorf("delta declaration over a per-token prose stream reported %q", whole.Verdict)
	}
	if !strings.Contains(strings.ToLower(whole.Checkpoint.Accepted), "rm -rf") {
		t.Skipf("recorded turn no longer proposes rm -rf; recapture needed: %q", whole.Checkpoint.Accepted)
	}

	steered, err := Open(streamingAdapter("test/openai-compat", ResolutionDelta),
		"traj-prose", "planner", liveCompute("worker-1", "captured"),
		[]streamrules.Rule{destructiveProseRule()})
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeOpenAIStream(steered, bytes.NewReader(body)); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !steered.Cancelled() {
		t.Fatal("the rm -rf plan did not close the epoch")
	}
	rep := steered.Report()
	if rep.TextDeltas >= whole.TextDeltas {
		t.Errorf("steered run consumed %d deltas of %d: it did not stop mid-stream",
			rep.TextDeltas, whole.TextDeltas)
	}
	// The prefix committed before the redirect is a real prefix of the turn.
	if !strings.HasPrefix(whole.Checkpoint.Accepted, rep.Checkpoint.Accepted) {
		t.Errorf("checkpoint %q is not a prefix of the full turn %q",
			rep.Checkpoint.Accepted, whole.Checkpoint.Accepted)
	}
	if !strings.Contains(strings.ToLower(rep.Checkpoint.Accepted), "rm -rf") {
		t.Errorf("the redirect fired before the matched text was committed: %q", rep.Checkpoint.Accepted)
	}
}

// TestCapturedTrajectorySurvivesRestart is the end-to-end done condition: one
// trajectory across two live providers and three epochs. Epoch 1 accepts a real
// streamed turn, epoch 2 is redirected while a real model proposes a recursive
// delete, and epoch 3 resumes on different compute under a different owner with
// the accepted prefix intact and the speculative bytes gone.
func TestCapturedTrajectorySurvivesRestart(t *testing.T) {
	rules := []streamrules.Rule{destructiveShellRule()}
	const trajectory = "traj-restart-6342"

	// --- epoch 1: Groq, prose. Every byte here is committed output. ---
	e1, err := Open(streamingAdapter("groq/openai-compat", ResolutionToolCall),
		trajectory, "planner-micro-agent", liveCompute("worker-cpu-1", "openai/gpt-oss-120b"), rules)
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeOpenAIStream(e1, bytes.NewReader(readCapture(t, "groq--openai-gpt-oss-120b--prose.sse"))); err != nil {
		t.Fatalf("epoch 1 decode: %v", err)
	}
	if e1.Cancelled() {
		t.Fatal("epoch 1 should not have been steered")
	}
	if e1.Epoch().Number != 1 {
		t.Fatalf("epoch 1 number = %d", e1.Epoch().Number)
	}
	prose := e1.Checkpoint().Accepted
	if len(prose) == 0 {
		t.Fatal("epoch 1 accepted nothing")
	}
	if e1.Report().TextDeltas < 2 {
		t.Fatalf("epoch 1 saw %d text deltas; the capture is not a live stream", e1.Report().TextDeltas)
	}
	// Hand the trajectory on at a clean boundary.
	yielded, err := e1.Steer(Directive{Kind: Yield, Reason: "handoff to the tool-capable worker"})
	if err != nil {
		t.Fatal(err)
	}
	if yielded.Checkpoint == nil {
		t.Fatal("yield returned no checkpoint")
	}

	// --- epoch 2: NVIDIA, the destructive proposal. Different provider, same
	// trajectory, different compute. ---
	e2, err := Reopen(streamingAdapter("nvidia/openai-compat", ResolutionToolCall),
		*yielded.Checkpoint, "safety-micro-agent", liveCompute("worker-gpu-7", "deepseek-ai/deepseek-v4-flash-0731"), rules)
	if err != nil {
		t.Fatal(err)
	}
	if e2.Epoch().Number != 2 {
		t.Fatalf("epoch 2 number = %d, want 2", e2.Epoch().Number)
	}
	if got := e2.Checkpoint().Accepted; got != prose {
		t.Fatalf("epoch 2 did not inherit the prefix:\n got %q\nwant %q", got, prose)
	}
	if err := DecodeOpenAIStream(e2, bytes.NewReader(readCapture(t, "nvidia--deepseek-ai-deepseek-v4-flash-0731--tool-destructive.sse"))); err != nil {
		t.Fatalf("epoch 2 decode: %v", err)
	}
	if !e2.Cancelled() {
		t.Fatal("epoch 2 was not redirected by the destructive call")
	}
	redirect := e2.Steering()
	if redirect.Checkpoint == nil {
		t.Fatal("the redirect returned no checkpoint to resume from")
	}
	rep2 := e2.Report()
	if rep2.Boundaries[0].Admit {
		t.Fatal("the destructive call became an effect")
	}
	if !strings.HasPrefix(redirect.Checkpoint.Accepted, prose) {
		t.Errorf("the redirect checkpoint dropped epoch 1's prefix: %q", redirect.Checkpoint.Accepted)
	}
	if strings.Contains(strings.ToLower(redirect.Checkpoint.Accepted), "remove-item") {
		t.Errorf("speculative arguments survived into the checkpoint: %q", redirect.Checkpoint.Accepted)
	}

	// --- epoch 3: restart from the returned checkpoint under the substitute
	// action. This is the "accepted prefix survives restart" claim. ---
	e3, err := Reopen(streamingAdapter("groq/openai-compat", ResolutionToolCall),
		*redirect.Checkpoint, "inventory-micro-agent", liveCompute("worker-cpu-2", "openai/gpt-oss-120b"), rules)
	if err != nil {
		t.Fatal(err)
	}
	if e3.Epoch().Number != 3 {
		t.Fatalf("epoch 3 number = %d, want 3", e3.Epoch().Number)
	}
	if e3.Epoch().TrajectoryID != trajectory {
		t.Fatalf("trajectory identity changed across the handoff: %q", e3.Epoch().TrajectoryID)
	}
	restored := e3.Checkpoint().Accepted
	if !strings.HasPrefix(restored, prose) {
		t.Fatalf("epoch 3 lost epoch 1's accepted prefix:\n got %q\nwant prefix %q", restored, prose)
	}
	if strings.Contains(strings.ToLower(restored), "remove-item") {
		t.Fatalf("epoch 3 resumed carrying speculative tool arguments: %q", restored)
	}
	if _, err := e3.Text(" Running the read-only inventory instead."); err != nil {
		t.Fatalf("epoch 3 could not continue the trajectory: %v", err)
	}
	final := e3.Checkpoint()
	if final.TrajectoryID != trajectory || final.AfterEpoch != 3 {
		t.Fatalf("final checkpoint = %+v", final)
	}
	if !strings.HasPrefix(final.Accepted, prose) || !strings.HasSuffix(final.Accepted, "inventory instead.") {
		t.Fatalf("final accepted prefix is not continuous: %q", final.Accepted)
	}
}

// unboundedGlobRule interrupts a recursive scan proposed through the Glob tool.
// The captured turn really does propose `**/*.o`, so this rule fires on recorded
// arguments rather than on a phrase written to be caught.
func unboundedGlobRule() streamrules.Rule {
	return streamrules.Rule{
		Name:             "no-unbounded-recursive-scan",
		Tool:             "Glob",
		Scope:            streamrules.ScopeNamedTool,
		Pattern:          `\*\*/`,
		Interrupt:        true,
		SubstituteAction: "Scan one directory level and report the count before widening to a recursive glob.",
	}
}

// TestCapturedAnthropicWireResolvesAtTheCallBoundary records something about
// fak's OWN surface rather than a third party. The Anthropic `/v1/messages`
// wire is the one that CAN carry a tool call's input in fragments, so it is the
// obvious place to claim delta resolution — but a live `fak serve` holds each
// tool_use block for adjudication and re-emits the adjudicated input as a
// single input_json_delta. So a downstream adapter over the gateway gets
// tool-call resolution, and saying otherwise would be an overclaim.
func TestCapturedAnthropicWireResolvesAtTheCallBoundary(t *testing.T) {
	const name = "fak-gateway-anthropic--claude-sonnet-4-5--anthropic-unbounded-glob.sse"
	body := readCapture(t, name)
	anthropic := func(declared Resolution) Adapter {
		return Adapter{
			Name:      "gateway/anthropic-messages-passthrough",
			Wire:      "anthropic-messages-sse",
			Streaming: true,
			Declared:  declared,
		}
	}

	measure, err := Open(anthropic(ResolutionDelta), "traj-anthropic", "planner",
		liveCompute("worker-1", "captured"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeAnthropicStream(measure, bytes.NewReader(body)); err != nil {
		t.Fatalf("decode: %v", err)
	}
	rep := measure.Report()
	if rep.ToolCalls != 1 {
		t.Fatalf("want one tool_use block, got %d", rep.ToolCalls)
	}
	if rep.ToolArgFragments != 1 || rep.ObservedToolArgs != ResolutionToolCall {
		t.Errorf("observed %q over %d fragments; the gateway re-emits adjudicated input whole, so want %q over 1",
			rep.ObservedToolArgs, rep.ToolArgFragments, ResolutionToolCall)
	}
	if rep.Verdict != ResolutionOverclaimed {
		t.Errorf("verdict = %q, want %q: this turn gave no sub-call boundary to steer at", rep.Verdict, ResolutionOverclaimed)
	}
	if b := rep.Boundaries[0]; !b.Admit || b.Tool != "Glob" {
		t.Fatalf("unsteered Glob call was not admitted: %+v", b)
	}

	// With the rule armed the same recorded call is redirected instead, and the
	// arguments never reach the boundary as an effect.
	steered, err := Open(anthropic(ResolutionToolCall), "traj-anthropic", "planner",
		liveCompute("worker-1", "captured"), []streamrules.Rule{unboundedGlobRule()})
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeAnthropicStream(steered, bytes.NewReader(body)); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !steered.Cancelled() {
		t.Fatal("the unbounded glob did not close the epoch")
	}
	tr := steered.Steering()
	if tr.Directive.Kind != Redirect || tr.Checkpoint == nil {
		t.Fatalf("steering = %+v, want a redirect with a checkpoint", tr.Directive)
	}
	sr := steered.Report()
	if sr.Verdict != ResolutionHonest {
		t.Errorf("verdict = %q, want %q", sr.Verdict, ResolutionHonest)
	}
	if b := sr.Boundaries[0]; b.Admit || b.Reason != ToolRefusedRedirected {
		t.Fatalf("redirected Glob call was not refused: %+v", b)
	}
	if strings.Contains(sr.Checkpoint.Accepted, "**/") {
		t.Errorf("tool arguments leaked into the accepted prefix: %q", sr.Checkpoint.Accepted)
	}

	// Resuming from that checkpoint keeps the trajectory, drops the proposal.
	next, err := Reopen(anthropic(ResolutionToolCall), *tr.Checkpoint, "inventory-micro-agent",
		liveCompute("worker-2", "captured"), []streamrules.Rule{unboundedGlobRule()})
	if err != nil {
		t.Fatal(err)
	}
	if next.Epoch().Number != 2 || next.Epoch().TrajectoryID != "traj-anthropic" {
		t.Fatalf("resumed epoch = %+v", next.Epoch())
	}
	if next.Report().ToolCalls != 0 {
		t.Error("the refused call was carried into the resumed epoch")
	}
}

// TestCapturedNonStreamingAdapterReportsRequestBoundary covers the last clause
// of the done condition. The same provider, asked with stream:false, produces
// the same destructive call — and the adapter that consumes it must say its
// only steering boundary is the request, rather than passing for live.
func TestCapturedNonStreamingAdapterReportsRequestBoundary(t *testing.T) {
	body := readCapture(t, "groq--openai-gpt-oss-120b--tool-destructive-nonstream.json")
	buffered := Adapter{
		Name:      "test/openai-compat-buffered",
		Wire:      "openai-chat-completions",
		Streaming: false,
		Declared:  ResolutionRequest,
	}

	b, err := Open(buffered, "traj-buffered", "planner", liveCompute("worker-1", "captured"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeOpenAIResponse(b, body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	rep := b.Report()
	if rep.ToolCalls != 1 {
		t.Fatalf("want one proposed call, got %d", rep.ToolCalls)
	}
	if rep.Effective != ResolutionRequest {
		t.Errorf("effective resolution = %q, want %q", rep.Effective, ResolutionRequest)
	}
	if rep.ObservedText != ResolutionRequest || rep.ObservedToolArgs != ResolutionRequest {
		t.Errorf("buffered adapter reported text=%q tool=%q; both must be %q",
			rep.ObservedText, rep.ObservedToolArgs, ResolutionRequest)
	}
	if rep.Verdict != ResolutionHonest {
		t.Errorf("verdict = %q, want %q", rep.Verdict, ResolutionHonest)
	}

	// A buffered adapter cannot claim live steering at all: the claim is
	// refused when the epoch opens, not quietly accepted and never checked.
	for _, overclaim := range []Resolution{ResolutionToolCall, ResolutionDelta} {
		bad := buffered
		bad.Declared = overclaim
		if _, err := Open(bad, "traj-buffered", "planner", liveCompute("worker-1", "captured"), nil); err == nil {
			t.Errorf("a non-streaming adapter was allowed to declare %q", overclaim)
		}
	}

	// Routing a buffered body through a streaming adapter is refused too, so a
	// caller cannot launder a whole-turn response into a live-looking report.
	streaming, err := Open(streamingAdapter("test/openai-compat", ResolutionToolCall),
		"traj-buffered", "planner", liveCompute("worker-1", "captured"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeOpenAIResponse(streaming, body); err == nil {
		t.Error("a streaming adapter was allowed to decode a whole-response body")
	}

	// Even at request resolution the rule still refuses the effect; what the
	// buffered adapter loses is earliness, not enforcement.
	steered, err := Open(buffered, "traj-buffered", "planner", liveCompute("worker-1", "captured"),
		[]streamrules.Rule{destructiveShellRule()})
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeOpenAIResponse(steered, body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !steered.Cancelled() {
		t.Fatal("the buffered destructive call did not close the epoch")
	}
	if boundary := steered.Report().Boundaries[0]; boundary.Admit {
		t.Fatal("the buffered destructive call became an effect")
	}
}

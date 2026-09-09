package llamacppinterop

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

const (
	StrixLlamaC1Seed = 424242

	strixLlamaC1CreditReason              = "pair-and-five-measured-trials-required"
	strixLlamaC1MaxResponseBytes          = 4 << 20
	strixLlamaC1ExchangeBindTimeout       = 500 * time.Millisecond
	strixLlamaC1TimingRelativeTolerance   = 0.005
	strixLlamaC1TimingAbsoluteMSTolerance = 0.01
)

var ErrInvalidStrixLlamaC1Observation = errors.New("invalid Strix llama.cpp c1 observation")

// StrixLlamaC1CaptureOptions identifies one approved c=1 exchange. Endpoint
// must be the literal IPv4 loopback peer accepted by Authority's exact child.
type StrixLlamaC1CaptureOptions struct {
	Authority          *StrixComparatorAuthority
	Cell               qwen38quantrun.StrixComparisonCellManifest
	ApprovedCellDigest string
	Packet             qwen38quantrun.PromptTokenPacket
	Endpoint           string
}

type strixLlamaC1CompletionRequest struct {
	Prompt            []int   `json:"prompt"`
	NPredict          int     `json:"n_predict"`
	Temperature       float64 `json:"temperature"`
	TopK              int     `json:"top_k"`
	TopP              float64 `json:"top_p"`
	Seed              int     `json:"seed"`
	Stream            bool    `json:"stream"`
	IgnoreEOS         bool    `json:"ignore_eos"`
	CachePrompt       bool    `json:"cache_prompt"`
	ReturnTokens      bool    `json:"return_tokens"`
	NProbs            int     `json:"n_probs"`
	PostSamplingProbs bool    `json:"post_sampling_probs"`
}

type StrixLlamaC1Timings struct {
	PromptTokens        int
	PromptMS            float64
	PromptPerTokenMS    float64
	PromptPerSecond     float64
	PredictedTokens     int
	PredictedMS         float64
	PredictedPerTokenMS float64
	PredictedPerSecond  float64
	DecodeSteps         int
	CacheTokens         int
}

type strixLlamaC1Probability struct {
	ID      *int     `json:"id"`
	Token   *string  `json:"token"`
	Bytes   []int    `json:"bytes"`
	Logprob *float64 `json:"logprob"`
}

type strixLlamaC1TokenProbability struct {
	ID          *int                      `json:"id"`
	Token       *string                   `json:"token"`
	Bytes       []int                     `json:"bytes"`
	Logprob     *float64                  `json:"logprob"`
	TopLogprobs []strixLlamaC1Probability `json:"top_logprobs"`
}

type strixLlamaC1Response struct {
	Index                   *int                           `json:"index"`
	Content                 string                         `json:"content"`
	Tokens                  []int                          `json:"tokens"`
	IDSlot                  *int                           `json:"id_slot"`
	Stop                    *bool                          `json:"stop"`
	Model                   json.RawMessage                `json:"model"`
	TokensPredicted         *int                           `json:"tokens_predicted"`
	TokensEvaluated         *int                           `json:"tokens_evaluated"`
	GenerationSettings      map[string]json.RawMessage     `json:"generation_settings"`
	Prompt                  json.RawMessage                `json:"prompt"`
	HasNewLine              *bool                          `json:"has_new_line"`
	Truncated               *bool                          `json:"truncated"`
	StoppedEOS              *bool                          `json:"stopped_eos"`
	StoppedWord             *bool                          `json:"stopped_word"`
	StoppedLimit            *bool                          `json:"stopped_limit"`
	StopType                *string                        `json:"stop_type"`
	StoppingWord            *string                        `json:"stopping_word"`
	TokensCached            *int                           `json:"tokens_cached"`
	CompletionProbabilities []strixLlamaC1TokenProbability `json:"completion_probabilities"`
	Timings                 struct {
		PromptN             int     `json:"prompt_n"`
		PromptMS            float64 `json:"prompt_ms"`
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
		PromptPerSecond     float64 `json:"prompt_per_second"`
		PredictedN          int     `json:"predicted_n"`
		PredictedMS         float64 `json:"predicted_ms"`
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
		PredictedPerSecond  float64 `json:"predicted_per_second"`
		CacheN              *int    `json:"cache_n"`
	} `json:"timings"`
}

// StrixLlamaC1Observation is opaque, copy-resistant, and connection-bound.
// Close releases only the client socket created by its capture operation.
type StrixLlamaC1Observation struct {
	mu           *sync.Mutex
	self         *StrixLlamaC1Observation
	authority    *StrixComparatorAuthority
	exchange     *StrixComparatorExchangeAuthority
	resource     *StrixComparatorResourceAuthority
	resources    StrixComparatorResourceObservation
	connection   *net.TCPConn
	closed       bool
	pid          int
	endpoint     string
	cellDigest   string
	packetDigest string
	request      []byte
	response     []byte
	requestSHA   string
	responseSHA  string
	bindingSHA   string
	tokenIDs     []int
	tokenTexts   []string
	logprobs     []float64
	timings      StrixLlamaC1Timings
}

// CaptureStrixLlamaC1Observation sends exactly one deterministic POST over a
// connection first bound to the approved reference and exact accepted socket.
func CaptureStrixLlamaC1Observation(ctx context.Context, options StrixLlamaC1CaptureOptions) (_ *StrixLlamaC1Observation, retErr error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidStrixLlamaC1Observation)
	}
	pid, err := validStrixLlamaC1Authority(options.Authority)
	if err != nil {
		return nil, err
	}
	if err := validateStrixLlamaC1CellAndPacket(options); err != nil {
		return nil, err
	}
	endpoint, address, err := parseStrixLlamaC1Endpoint(options.Endpoint)
	if err != nil {
		return nil, err
	}
	requestBody, err := buildStrixLlamaC1RequestBody(options.Packet)
	if err != nil {
		return nil, err
	}
	requestWire := buildStrixLlamaC1HTTPRequest(address, requestBody)

	rawConnection, err := (&net.Dialer{Timeout: 5 * time.Second}).DialContext(ctx, "tcp4", address)
	if err != nil {
		return nil, fmt.Errorf("%w: dial completion endpoint: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	connection, ok := rawConnection.(*net.TCPConn)
	if !ok {
		_ = rawConnection.Close()
		return nil, fmt.Errorf("%w: endpoint did not yield a TCP connection", ErrInvalidStrixLlamaC1Observation)
	}
	keepConnection := false
	defer func() {
		if !keepConnection {
			_ = connection.Close()
		}
	}()
	exchange, err := bindStrixLlamaC1Exchange(ctx, options, connection)
	if err != nil {
		return nil, err
	}
	if !exchange.Valid() {
		return nil, fmt.Errorf("%w: accepted connection authority is invalid", ErrInvalidStrixLlamaC1Observation)
	}
	resourceWindow, err := exchange.BeginResourceWindow(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w: begin full request resource window: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	resourceWindowOpen := true
	defer func() {
		if !resourceWindowOpen {
			return
		}
		if cleanupErr := resourceWindow.Close(); cleanupErr != nil {
			cleanupErr = fmt.Errorf("%w: cancel failed request resource window: %v", ErrInvalidStrixLlamaC1Observation, cleanupErr)
			if retErr == nil {
				retErr = cleanupErr
			} else {
				retErr = errors.Join(retErr, cleanupErr)
			}
		}
	}()
	deadline := time.Now().Add(30 * time.Second)
	if contextDeadline, ok := ctx.Deadline(); ok && contextDeadline.Before(deadline) {
		deadline = contextDeadline
	}
	if err := connection.SetDeadline(deadline); err != nil {
		return nil, fmt.Errorf("%w: set exchange deadline: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	if written, err := io.Copy(connection, bytes.NewReader(requestWire)); err != nil || written != int64(len(requestWire)) {
		return nil, fmt.Errorf("%w: write completion request: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	req, err := http.NewRequest(http.MethodPost, endpoint+"/completion", nil)
	if err != nil {
		return nil, fmt.Errorf("%w: reconstruct request: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	response, err := http.ReadResponse(bufio.NewReader(connection), req)
	if err != nil {
		return nil, fmt.Errorf("%w: read completion response: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, strixLlamaC1MaxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("%w: read response body: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	if len(responseBody) > strixLlamaC1MaxResponseBytes {
		return nil, fmt.Errorf("%w: response exceeds %d bytes", ErrInvalidStrixLlamaC1Observation, strixLlamaC1MaxResponseBytes)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: completion HTTP status %d", ErrInvalidStrixLlamaC1Observation, response.StatusCode)
	}
	if mediaType := strings.TrimSpace(strings.Split(response.Header.Get("Content-Type"), ";")[0]); mediaType != "application/json" {
		return nil, fmt.Errorf("%w: response Content-Type %q", ErrInvalidStrixLlamaC1Observation, response.Header.Get("Content-Type"))
	}
	parsed, tokenTexts, logprobs, timings, err := parseStrixLlamaC1Response(responseBody, len(options.Packet.PromptTokenIDs))
	if err != nil {
		return nil, err
	}
	if err := connection.SetDeadline(time.Time{}); err != nil {
		return nil, fmt.Errorf("%w: clear exchange deadline: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	if pidAfter, err := validStrixLlamaC1Authority(options.Authority); err != nil || pidAfter != pid || !exchange.Valid() {
		return nil, fmt.Errorf("%w: comparator exchange changed during request", ErrInvalidStrixLlamaC1Observation)
	}
	resource, err := resourceWindow.Finalize()
	resourceWindowOpen = false
	if err != nil || resource == nil || !resource.Valid() {
		return nil, fmt.Errorf("%w: finalize full request resource window: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	resources, ok := resource.Observation()
	if !ok {
		return nil, fmt.Errorf("%w: finalized resource authority withheld its observation", ErrInvalidStrixLlamaC1Observation)
	}

	requestSum, responseSum := sha256.Sum256(requestWire), sha256.Sum256(responseBody)
	o := &StrixLlamaC1Observation{
		mu: new(sync.Mutex), authority: options.Authority, exchange: exchange, resource: resource, resources: resources, connection: connection,
		pid: pid, endpoint: endpoint, cellDigest: options.Cell.Digest, packetDigest: options.Packet.PacketDigest,
		request: slices.Clone(requestWire), response: slices.Clone(responseBody),
		requestSHA: hex.EncodeToString(requestSum[:]), responseSHA: hex.EncodeToString(responseSum[:]),
		tokenIDs: slices.Clone(parsed.Tokens), tokenTexts: tokenTexts, logprobs: logprobs, timings: timings,
	}
	o.bindingSHA = o.computeBindingSHA()
	o.self = o
	if !o.Valid() {
		return nil, fmt.Errorf("%w: final exchange binding failed", ErrInvalidStrixLlamaC1Observation)
	}
	keepConnection = true
	return o, nil
}

func buildStrixLlamaC1RequestBody(packet qwen38quantrun.PromptTokenPacket) ([]byte, error) {
	controls := packet.GenerationControls
	if len(packet.PromptTokenIDs) != 26 || controls.MaxOutputTokens != qwen38quantrun.StrixComparisonOutputTokens || controls.Temperature != 0 || controls.TopK != 1 || controls.TopP != 1 || !controls.IgnoreEOS || len(packet.StopTokens) != 0 || len(packet.StopTokenIDs) != 0 || len(controls.StopTokens) != 0 || len(controls.StopTokenIDs) != 0 {
		return nil, fmt.Errorf("%w: request packet is not the exact deterministic 26-ID c1 envelope", ErrInvalidStrixLlamaC1Observation)
	}
	body, err := json.Marshal(strixLlamaC1CompletionRequest{
		Prompt: slices.Clone(packet.PromptTokenIDs), NPredict: qwen38quantrun.StrixComparisonOutputTokens,
		Temperature: 0, TopK: 1, TopP: 1, Seed: StrixLlamaC1Seed, Stream: false,
		IgnoreEOS: true, CachePrompt: false, ReturnTokens: true, NProbs: 1, PostSamplingProbs: false,
	})
	if err != nil {
		return nil, fmt.Errorf("%w: encode request: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	return body, nil
}

func validateStrixLlamaC1CellAndPacket(options StrixLlamaC1CaptureOptions) error {
	if err := qwen38quantrun.VerifyStrixComparisonCellManifest(options.Cell, options.ApprovedCellDigest); err != nil {
		return fmt.Errorf("%w: cell: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	if options.Cell.Challenge.Concurrency != 1 || options.Cell.Workload.AcceptedOutputTokens != qwen38quantrun.StrixComparisonOutputTokens || options.Cell.Workload.SpeculationEnabled || options.Cell.Reference.CampaignClass != "llama.cpp-comparator-only" || options.Cell.Reference.SourceRevision != qwen38quantrun.StrixComparisonLlamaSourceRevision {
		return fmt.Errorf("%w: cell is not the approved non-speculative comparator c1 cell", ErrInvalidStrixLlamaC1Observation)
	}
	if err := qwen38quantrun.VerifyPromptPacket(options.Packet); err != nil {
		return fmt.Errorf("%w: packet: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	packetBytes, err := qwen38quantrun.ExportPromptPacket(options.Packet)
	if err != nil {
		return fmt.Errorf("%w: packet: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	if options.Packet.PacketDigest != options.Cell.Workload.PromptPacketDigest || !bytes.Equal(packetBytes, options.Cell.Workload.PromptPacketBytes) || !slices.Equal(options.Packet.PromptTokenIDs, options.Cell.Workload.PromptTokenIDs) {
		return fmt.Errorf("%w: packet does not match approved cell", ErrInvalidStrixLlamaC1Observation)
	}
	return nil
}

func bindStrixLlamaC1Exchange(ctx context.Context, options StrixLlamaC1CaptureOptions, connection *net.TCPConn) (*StrixComparatorExchangeAuthority, error) {
	deadline := time.Now().Add(strixLlamaC1ExchangeBindTimeout)
	if contextDeadline, ok := ctx.Deadline(); ok && contextDeadline.Before(deadline) {
		deadline = contextDeadline
	}
	var lastErr error
	for {
		exchange, err := options.Authority.BindApprovedReferenceAndAcceptedConnection(options.Cell, options.ApprovedCellDigest, connection)
		if err == nil {
			return exchange, nil
		}
		lastErr = err
		if !time.Now().Before(deadline) {
			return nil, fmt.Errorf("%w: bind accepted connection: %v", ErrInvalidStrixLlamaC1Observation, lastErr)
		}
		timer := time.NewTimer(5 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, fmt.Errorf("%w: bind accepted connection: %v", ErrInvalidStrixLlamaC1Observation, ctx.Err())
		case <-timer.C:
		}
	}
}

func buildStrixLlamaC1HTTPRequest(address string, body []byte) []byte {
	var request bytes.Buffer
	fmt.Fprintf(&request, "POST /completion HTTP/1.1\r\nHost: %s\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: keep-alive\r\n\r\n", address, len(body))
	request.Write(body)
	return request.Bytes()
}

func validStrixLlamaC1Authority(authority *StrixComparatorAuthority) (int, error) {
	if authority == nil || !authority.Valid() {
		return 0, fmt.Errorf("%w: comparator authority is invalid", ErrInvalidStrixLlamaC1Observation)
	}
	pid := authority.PID()
	if pid <= 0 || !authority.Valid() || authority.PID() != pid {
		return 0, fmt.Errorf("%w: comparator authority is unstable", ErrInvalidStrixLlamaC1Observation)
	}
	return pid, nil
}

func parseStrixLlamaC1Endpoint(raw string) (endpoint, address string, err error) {
	u, parseErr := url.Parse(raw)
	if parseErr != nil || u.Scheme != "http" || u.Hostname() != "127.0.0.1" || u.Port() == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.Opaque != "" {
		return "", "", fmt.Errorf("%w: endpoint must be literal http://127.0.0.1:<port>", ErrInvalidStrixLlamaC1Observation)
	}
	port, parseErr := strconv.ParseUint(u.Port(), 10, 16)
	if parseErr != nil || port == 0 {
		return "", "", fmt.Errorf("%w: endpoint port is invalid", ErrInvalidStrixLlamaC1Observation)
	}
	address = net.JoinHostPort("127.0.0.1", strconv.FormatUint(port, 10))
	endpoint = "http://" + address
	if raw != endpoint {
		return "", "", fmt.Errorf("%w: endpoint is not canonical", ErrInvalidStrixLlamaC1Observation)
	}
	return endpoint, address, nil
}

func parseStrixLlamaC1Response(raw []byte, promptTokens int) (strixLlamaC1Response, []string, []float64, StrixLlamaC1Timings, error) {
	if err := rejectStrixLlamaC1AmbiguousJSON(raw); err != nil {
		return strixLlamaC1Response{}, nil, nil, StrixLlamaC1Timings{}, err
	}
	var response strixLlamaC1Response
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&response); err != nil {
		return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: malformed completion JSON: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	if err := requireStrixLlamaC1JSONEOF(decoder); err != nil {
		return response, nil, nil, StrixLlamaC1Timings{}, err
	}
	if promptTokens != 26 || response.Index == nil || *response.Index != 0 || response.IDSlot == nil || *response.IDSlot != 0 ||
		response.Stop == nil || !*response.Stop || response.TokensPredicted == nil || *response.TokensPredicted != qwen38quantrun.StrixComparisonOutputTokens ||
		response.TokensEvaluated == nil || *response.TokensEvaluated != promptTokens || response.Truncated == nil || *response.Truncated ||
		response.StoppedEOS != nil && *response.StoppedEOS || response.StoppedWord != nil && *response.StoppedWord || response.StoppedLimit != nil && !*response.StoppedLimit ||
		response.StopType == nil || *response.StopType != "limit" || response.StoppingWord == nil || *response.StoppingWord != "" ||
		response.TokensCached == nil || *response.TokensCached != 153 {
		return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: completion envelope is not the exact pinned c1 limit response", ErrInvalidStrixLlamaC1Observation)
	}
	speculationRaw, ok := response.GenerationSettings["speculative.types"]
	if !ok {
		return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: response omitted speculative.types", ErrInvalidStrixLlamaC1Observation)
	}
	var speculation string
	if err := json.Unmarshal(speculationRaw, &speculation); err != nil || speculation != "none" {
		return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: response did not prove speculation disabled", ErrInvalidStrixLlamaC1Observation)
	}
	if len(response.Tokens) != qwen38quantrun.StrixComparisonOutputTokens || len(response.CompletionProbabilities) != len(response.Tokens) {
		return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: completion must contain exactly %d tokens and probability rows", ErrInvalidStrixLlamaC1Observation, qwen38quantrun.StrixComparisonOutputTokens)
	}
	texts := make([]string, len(response.Tokens))
	logprobs := make([]float64, len(response.Tokens))
	contentBytes := make([]byte, 0, len(response.Content))
	for i, row := range response.CompletionProbabilities {
		if response.Tokens[i] < 0 || row.ID == nil || *row.ID != response.Tokens[i] || row.Token == nil || row.Logprob == nil || len(row.Bytes) == 0 || len(row.TopLogprobs) != 1 || !finiteNonpositiveStrixLlamaC1Logprob(*row.Logprob) {
			return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: token %d lacks the exact selected ID/bytes/logprob row", ErrInvalidStrixLlamaC1Observation, i)
		}
		for _, value := range row.Bytes {
			if value < 0 || value > 255 {
				return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: token %d contains an invalid byte", ErrInvalidStrixLlamaC1Observation, i)
			}
			contentBytes = append(contentBytes, byte(value))
		}
		top := row.TopLogprobs[0]
		if top.ID == nil || top.Token == nil || top.Logprob == nil || *top.ID != *row.ID || *top.Token != *row.Token || !slices.Equal(top.Bytes, row.Bytes) || *top.Logprob != *row.Logprob || !finiteNonpositiveStrixLlamaC1Logprob(*top.Logprob) {
			return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: token %d top logprob does not exactly match the selected row", ErrInvalidStrixLlamaC1Observation, i)
		}
		texts[i], logprobs[i] = *row.Token, *row.Logprob
	}
	if !utf8.ValidString(response.Content) || !bytes.Equal(contentBytes, []byte(response.Content)) {
		return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: concatenated selected token bytes do not equal UTF-8 content", ErrInvalidStrixLlamaC1Observation)
	}
	t := response.Timings
	values := []float64{t.PromptMS, t.PromptPerTokenMS, t.PromptPerSecond, t.PredictedMS, t.PredictedPerTokenMS, t.PredictedPerSecond}
	for _, value := range values {
		if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: prompt/decode timings are incomplete", ErrInvalidStrixLlamaC1Observation)
		}
	}
	const decodeSteps = qwen38quantrun.StrixComparisonOutputTokens - 1
	if t.CacheN == nil || *t.CacheN != 0 || t.PromptN != promptTokens || t.PredictedN != len(response.Tokens) ||
		!consistentStrixLlamaC1Timing(t.PromptN, t.PromptMS, t.PromptPerTokenMS, t.PromptPerSecond) ||
		!consistentStrixLlamaC1Timing(decodeSteps, t.PredictedMS, t.PredictedPerTokenMS, t.PredictedPerSecond) {
		return response, nil, nil, StrixLlamaC1Timings{}, fmt.Errorf("%w: prompt/decode timing fields contradict token counts or each other", ErrInvalidStrixLlamaC1Observation)
	}
	timings := StrixLlamaC1Timings{PromptTokens: t.PromptN, PromptMS: t.PromptMS, PromptPerTokenMS: t.PromptPerTokenMS, PromptPerSecond: t.PromptPerSecond, PredictedTokens: t.PredictedN, PredictedMS: t.PredictedMS, PredictedPerTokenMS: t.PredictedPerTokenMS, PredictedPerSecond: t.PredictedPerSecond, DecodeSteps: decodeSteps, CacheTokens: *t.CacheN}
	return response, texts, logprobs, timings, nil
}

func finiteNonpositiveStrixLlamaC1Logprob(value float64) bool {
	return value <= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func consistentStrixLlamaC1Timing(tokens int, milliseconds, perTokenMS, perSecond float64) bool {
	if tokens <= 0 {
		return false
	}
	wantPerToken := milliseconds / float64(tokens)
	wantPerSecond := float64(tokens) * 1000 / milliseconds
	return withinStrixLlamaC1TimingTolerance(perTokenMS, wantPerToken, strixLlamaC1TimingAbsoluteMSTolerance) && withinStrixLlamaC1TimingTolerance(perSecond, wantPerSecond, 0)
}

func withinStrixLlamaC1TimingTolerance(got, want, absolute float64) bool {
	tolerance := math.Max(absolute, math.Abs(want)*strixLlamaC1TimingRelativeTolerance)
	return math.Abs(got-want) <= tolerance
}

func rejectStrixLlamaC1AmbiguousJSON(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	var walk func(bool) error
	walk = func(allowDottedKeys bool) error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make([]string, 0)
			for decoder.More() {
				keyToken, err := decoder.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok || !(isLowerASCIIJSONKey(key) || allowDottedKeys && isLowerASCIIDottedJSONKey(key)) {
					return fmt.Errorf("ambiguous JSON field %q", key)
				}
				for _, prior := range seen {
					if strings.EqualFold(prior, key) {
						return fmt.Errorf("duplicate or case-folded JSON field %q", key)
					}
				}
				seen = append(seen, key)
				if err := walk(key == "generation_settings"); err != nil {
					return err
				}
			}
			_, err = decoder.Token()
			return err
		case '[':
			for decoder.More() {
				if err := walk(false); err != nil {
					return err
				}
			}
			_, err = decoder.Token()
			return err
		default:
			return fmt.Errorf("unexpected JSON delimiter %q", delim)
		}
	}
	if err := walk(false); err != nil {
		return fmt.Errorf("%w: malformed or ambiguous completion JSON: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	return nil
}

func isLowerASCIIDottedJSONKey(key string) bool {
	parts := strings.Split(key, ".")
	if len(parts) < 2 {
		return false
	}
	for _, part := range parts {
		if !isLowerASCIIJSONKey(part) {
			return false
		}
	}
	return true
}

func isLowerASCIIJSONKey(key string) bool {
	if key == "" {
		return false
	}
	for _, c := range key {
		if c > 0x7f || c >= 'A' && c <= 'Z' || !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_') {
			return false
		}
	}
	return true
}

func requireStrixLlamaC1JSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("%w: multiple completion JSON values", ErrInvalidStrixLlamaC1Observation)
		}
		return fmt.Errorf("%w: malformed completion JSON: %v", ErrInvalidStrixLlamaC1Observation, err)
	}
	return nil
}

func (o *StrixLlamaC1Observation) computeBindingSHA() string {
	if o == nil {
		return ""
	}
	var binding bytes.Buffer
	for _, value := range []string{
		strconv.Itoa(o.pid), o.endpoint, o.cellDigest, o.packetDigest, o.requestSHA, o.responseSHA,
		strconv.FormatUint(o.resources.ProcessPeakBytes, 10), strconv.FormatUint(o.resources.DevicePeakBytes, 10),
		strconv.FormatUint(o.resources.GPUEngineActiveNS, 10), o.resources.RADVDeviceIdentity,
	} {
		fmt.Fprintf(&binding, "%d:%s", len(value), value)
	}
	sum := sha256.Sum256(binding.Bytes())
	return hex.EncodeToString(sum[:])
}

func (o *StrixLlamaC1Observation) validLocked() bool {
	if o.self != o || o.closed || o.authority == nil || o.exchange == nil || o.resource == nil || o.connection == nil || o.pid <= 0 || o.computeBindingSHA() != o.bindingSHA {
		return false
	}
	resourceObservation, ok := o.resource.Observation()
	if !ok || !o.resource.Valid() || resourceObservation != o.resources {
		return false
	}
	requestSum, responseSum := sha256.Sum256(o.request), sha256.Sum256(o.response)
	if hex.EncodeToString(requestSum[:]) != o.requestSHA || hex.EncodeToString(responseSum[:]) != o.responseSHA || !o.exchange.Valid() {
		return false
	}
	parsed, texts, logs, timings, err := parseStrixLlamaC1Response(o.response, 26)
	if err != nil || !slices.Equal(parsed.Tokens, o.tokenIDs) || !slices.Equal(texts, o.tokenTexts) || !slices.Equal(logs, o.logprobs) || timings != o.timings {
		return false
	}
	pid, err := validStrixLlamaC1Authority(o.authority)
	return err == nil && pid == o.pid && o.exchange.Valid()
}

func (o *StrixLlamaC1Observation) Valid() bool {
	if o == nil || o.mu == nil {
		return false
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.validLocked()
}

// Close releases only the client connection created by Capture.
func (o *StrixLlamaC1Observation) Close() error {
	if o == nil || o.mu == nil {
		return ErrInvalidStrixLlamaC1Observation
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.self != o || o.closed || o.connection == nil {
		return ErrInvalidStrixLlamaC1Observation
	}
	o.closed = true
	return o.connection.Close()
}

func (o *StrixLlamaC1Observation) PID() int {
	if !o.Valid() {
		return 0
	}
	return o.pid
}
func (o *StrixLlamaC1Observation) Endpoint() string {
	if !o.Valid() {
		return ""
	}
	return o.endpoint
}
func (o *StrixLlamaC1Observation) RequestSHA256() string {
	if !o.Valid() {
		return ""
	}
	return o.requestSHA
}
func (o *StrixLlamaC1Observation) ResponseSHA256() string {
	if !o.Valid() {
		return ""
	}
	return o.responseSHA
}
func (o *StrixLlamaC1Observation) AcceptedTokenIDs() []int {
	if !o.Valid() {
		return nil
	}
	return slices.Clone(o.tokenIDs)
}
func (o *StrixLlamaC1Observation) AcceptedTokenTexts() []string {
	if !o.Valid() {
		return nil
	}
	return slices.Clone(o.tokenTexts)
}
func (o *StrixLlamaC1Observation) AcceptedTokenLogprobs() []float64 {
	if !o.Valid() {
		return nil
	}
	return slices.Clone(o.logprobs)
}
func (o *StrixLlamaC1Observation) Timings() StrixLlamaC1Timings {
	if !o.Valid() {
		return StrixLlamaC1Timings{}
	}
	return o.timings
}
func (o *StrixLlamaC1Observation) ResourceObservation() (StrixComparatorResourceObservation, bool) {
	if !o.Valid() {
		return StrixComparatorResourceObservation{}, false
	}
	return o.resources, true
}
func (*StrixLlamaC1Observation) PerformanceCredit() bool { return false }
func (*StrixLlamaC1Observation) CreditReason() string    { return strixLlamaC1CreditReason }

func (*StrixLlamaC1Observation) MarshalJSON() ([]byte, error) {
	return nil, fmt.Errorf("%w: observation is not serializable", ErrInvalidStrixLlamaC1Observation)
}
func (*StrixLlamaC1Observation) UnmarshalJSON([]byte) error {
	return fmt.Errorf("%w: observation is not deserializable", ErrInvalidStrixLlamaC1Observation)
}

var _ json.Marshaler = (*StrixLlamaC1Observation)(nil)
var _ json.Unmarshaler = (*StrixLlamaC1Observation)(nil)

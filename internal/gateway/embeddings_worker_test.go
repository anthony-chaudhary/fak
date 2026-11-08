package gateway

// embeddings_worker_test.go — pins TestEmbeddingsEncoderWorkerContract: the
// model-worker-backed /v1/embeddings seam (embedding_worker.go) and its wired
// 200/refusal matrix over handleEmbeddings. Software witness only: fake
// encoders, no throughput or hardware claim.

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// encoderContractFake is the deterministic fake encoder behind the contract. It
// writes every admitted row to the end even when ctx is done (borrowed storage
// survives cancellation to completion), records what it observed, and can be
// told to leave trailing rows unwritten or to report generated tokens.
type encoderContractFake struct {
	calls     int
	finish    EmbeddingFinish
	generated bool
	unwritten int // leave this many trailing rows unwritten (NaN)
	sawCancel bool
	wroteAll  bool
}

func (f *encoderContractFake) encode(ctx context.Context, rows [][]int, out []float64) (EmbeddingFinish, bool) {
	f.calls++
	if ctx.Err() != nil {
		f.sawCancel = true
	}
	dim := 0
	if len(rows) > 0 {
		dim = len(out) / len(rows)
	}
	for i, row := range rows {
		if f.unwritten > 0 && i >= len(rows)-f.unwritten {
			continue
		}
		sum := 0
		for _, id := range row {
			sum += id
		}
		base := float64(sum%97) / 97.0
		for d := 0; d < dim; d++ {
			out[i*dim+d] = base + float64(d)/float64(dim)
		}
	}
	f.wroteAll = f.unwritten == 0
	return f.finish, f.generated
}

// fak-test:runtime fast est=1ms lane=default
func TestEmbeddingsEncoderFiniteOutput(t *testing.T) {
	for _, format := range []string{"float", "base64"} {
		for _, tc := range []struct {
			name  string
			value float64
		}{
			{"positive infinity", math.Inf(1)},
			{"negative infinity", math.Inf(-1)},
			{"float32 maximum", math.MaxFloat32},
			{"float64 maximum", math.MaxFloat64},
		} {
			t.Run(format+"/"+tc.name, func(t *testing.T) {
				wk := encoderWorkerFixture(t, &encoderContractFake{})
				wk.Encode = func(_ context.Context, _ [][]int, out []float64) (EmbeddingFinish, bool) {
					for i := range out {
						out[i] = 1
					}
					out[len(out)-1] = tc.value // A bad final component refuses the whole batch.
					return EmbeddingFinishLength, false
				}
				w := httptest.NewRecorder()
				r := httptest.NewRequest(http.MethodPost, "/v1/embeddings", nil)
				(&Server{}).handleEncoderEmbeddings(w, r, EmbeddingsRequest{Input: json.RawMessage(`[[1],[2]]`), EncodingFormat: format}, wk)
				bad := math.IsInf(tc.value, 0) || (format == "base64" && tc.value == math.MaxFloat64)
				if bad {
					if w.Code != http.StatusUnprocessableEntity || strings.Contains(w.Body.String(), `"data"`) || !strings.Contains(w.Body.String(), ErrEmbeddingWorkerNonFinite.Error()) {
						t.Fatalf("non-finite batch response: %d %s", w.Code, w.Body.String())
					}
				} else {
					var response EmbeddingsResponse
					if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || w.Code != http.StatusOK || len(response.Data) != 2 {
						t.Fatalf("finite batch response: %d %s error=%v", w.Code, w.Body.String(), err)
					}
				}
				if wk.outstandingVectors() != 0 {
					t.Fatal("output refusal or success leaked borrowed storage")
				}
			})
		}
	}
	err := refuseEmbedding(ErrEmbeddingWorkerNonFinite)
	if !errors.Is(err, ErrEmbeddingWorkerNonFinite) || !errors.Is(err, ErrEmbeddingWorker) {
		t.Fatal("non-finite refusal lost typed classification")
	}
}

func encoderWorkerFixture(t *testing.T, fake *encoderContractFake) *EmbeddingEncoderWorker {
	t.Helper()
	wk, err := NewEmbeddingEncoderWorker(EmbeddingEncoderConfig{
		ModelName: "tiny-encoder",
		Pooling:   "mean",
		HiddenDim: 8,
		Formats:   []string{"float", "base64"},
		VocabSize: 64,
		Encode:    fake.encode,
	})
	if err != nil {
		t.Fatal(err)
	}
	return wk
}

func installEncoderWorker(t *testing.T, wk *EmbeddingEncoderWorker) {
	t.Helper()
	SetEmbeddingEncoderWorker(wk)
	t.Cleanup(func() { SetEmbeddingEncoderWorker(nil) })
}

func postEmbeddings(t *testing.T, s *Server, body string, ctx context.Context) (int, string, http.Header) {
	t.Helper()
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	if ctx != nil {
		r = r.WithContext(ctx)
	}
	s.handleEmbeddings(w, r)
	return w.Code, w.Body.String(), w.Header()
}

func mustUnmarshalRequest(t *testing.T, body string) EmbeddingsRequest {
	t.Helper()
	var req EmbeddingsRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

// joinIDRow renders n token IDs cycled through the fixture vocabulary as a
// JSON array literal, so oversized shapes trip the shape bound, not the
// vocabulary guard.
func joinIDRow(n, vocab int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = strconv.Itoa(i % vocab)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// fak-test:runtime fast est=100ms lane=default
func TestEmbeddingsEncoderWorkerContract(t *testing.T) {
	t.Run("constructor validates the declared capability set", func(t *testing.T) {
		base := EmbeddingEncoderConfig{
			ModelName: "m", Pooling: "mean", HiddenDim: 8,
			Formats: []string{"float"}, VocabSize: 4,
			Encode: func(context.Context, [][]int, []float64) (EmbeddingFinish, bool) {
				return EmbeddingFinishLength, false
			},
		}
		mutations := map[string]func(*EmbeddingEncoderConfig){
			"empty model name": func(c *EmbeddingEncoderConfig) { c.ModelName = "" },
			"unknown pooling":  func(c *EmbeddingEncoderConfig) { c.Pooling = "center" },
			"zero hidden":      func(c *EmbeddingEncoderConfig) { c.HiddenDim = 0 },
			"zero vocabulary":  func(c *EmbeddingEncoderConfig) { c.VocabSize = 0 },
			"unknown format":   func(c *EmbeddingEncoderConfig) { c.Formats = []string{"hex"} },
			"no formats":       func(c *EmbeddingEncoderConfig) { c.Formats = nil },
			"nil encode seam":  func(c *EmbeddingEncoderConfig) { c.Encode = nil },
		}
		for name, mutate := range mutations {
			cfg := base
			mutate(&cfg)
			if _, err := NewEmbeddingEncoderWorker(cfg); err == nil {
				t.Fatalf("%s accepted", name)
			}
		}
	})

	t.Run("capability guards refuse before encoder work", func(t *testing.T) {
		fake := &encoderContractFake{finish: EmbeddingFinishLength}
		wk := encoderWorkerFixture(t, fake)
		installEncoderWorker(t, wk)
		s := &Server{model: "tiny-encoder", metrics: newGatewayMetrics(time.Now())}

		code, bodyOut, _ := postEmbeddings(t, s, `{"input":"ab","dimensions":4}`, nil)
		if code != http.StatusBadRequest || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("dimension mismatch %d %s", code, bodyOut)
		}
		if _, err := wk.admit(mustUnmarshalRequest(t, `{"input":"ab","dimensions":4}`)); !errors.Is(err, ErrEmbeddingWorkerDimensions) || !errors.Is(err, ErrEmbeddingWorker) {
			t.Fatalf("dimensions sentinel: %v", err)
		}

		code, _, _ = postEmbeddings(t, s, `{"input":"ab","encoding_format":"hex"}`, nil)
		if code != http.StatusBadRequest {
			t.Fatalf("undeclared format %d", code)
		}
		if _, err := wk.admit(mustUnmarshalRequest(t, `{"input":"ab","encoding_format":"hex"}`)); !errors.Is(err, ErrEmbeddingWorkerFormat) {
			t.Fatalf("format sentinel: %v", err)
		}

		code, _, _ = postEmbeddings(t, s, `{"input":"ab","model":"other"}`, nil)
		if code != http.StatusBadRequest {
			t.Fatalf("model mismatch %d", code)
		}
		if _, err := wk.admit(mustUnmarshalRequest(t, `{"input":"ab","model":"other"}`)); !errors.Is(err, ErrEmbeddingWorkerModel) {
			t.Fatalf("model sentinel: %v", err)
		}

		if fake.calls != 0 {
			t.Fatalf("guards ran encoder work: calls=%d", fake.calls)
		}
	})

	t.Run("undeclared encoding form refuses on a narrow worker", func(t *testing.T) {
		fake := &encoderContractFake{finish: EmbeddingFinishLength}
		wk, err := NewEmbeddingEncoderWorker(EmbeddingEncoderConfig{
			ModelName: "tiny-encoder", Pooling: "last", HiddenDim: 8,
			Formats: []string{"float"}, VocabSize: 64, Encode: fake.encode,
		})
		if err != nil {
			t.Fatal(err)
		}
		installEncoderWorker(t, wk)
		s := &Server{model: "tiny-encoder", metrics: newGatewayMetrics(time.Now())}
		code, _, _ := postEmbeddings(t, s, `{"input":"ab","encoding_format":"base64"}`, nil)
		if code != http.StatusBadRequest {
			t.Fatalf("narrow worker answered base64: %d", code)
		}
		if _, err := wk.admit(mustUnmarshalRequest(t, `{"input":"ab","encoding_format":"base64"}`)); !errors.Is(err, ErrEmbeddingWorkerFormat) {
			t.Fatalf("narrow-format sentinel: %v", err)
		}
		if fake.calls != 0 {
			t.Fatalf("narrow guard ran encoder work: calls=%d", fake.calls)
		}
	})

	t.Run("bounded batches preserve sequence boundaries", func(t *testing.T) {
		fake := &encoderContractFake{finish: EmbeddingFinishLength}
		wk := encoderWorkerFixture(t, fake)
		installEncoderWorker(t, wk)
		s := &Server{model: "tiny-encoder", metrics: newGatewayMetrics(time.Now())}

		code, bodyOut, header := postEmbeddings(t, s, `{"input":["alpha beta","gamma","delta epsilon zeta"]}`, nil)
		if code != http.StatusOK {
			t.Fatalf("text batch %d %s", code, bodyOut)
		}
		var resp EmbeddingsResponse
		if err := json.Unmarshal([]byte(bodyOut), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Object != "list" || len(resp.Data) != 3 || resp.Model != "tiny-encoder" {
			t.Fatalf("envelope %+v", resp)
		}
		for i, d := range resp.Data {
			if d.Object != "embedding" || d.Index != i {
				t.Fatalf("row %d %+v", i, d)
			}
			if _, ok := d.Embedding.([]any); !ok {
				t.Fatalf("float form is not a JSON number array: %T", d.Embedding)
			}
		}
		wantTokens := 0
		for _, txt := range []string{"alpha beta", "gamma", "delta epsilon zeta"} {
			wantTokens += len(tokenizeForEmbedding(txt))
		}
		if resp.Usage.PromptTokens != wantTokens || resp.Usage.TotalTokens != wantTokens {
			t.Fatalf("usage %+v, want %d fully-written input tokens", resp.Usage, wantTokens)
		}
		if header.Get("X-Fak-Embedding-Mode") != "encoder-mean" {
			t.Fatalf("pooling identity header %q", header.Get("X-Fak-Embedding-Mode"))
		}

		code, bodyOut, _ = postEmbeddings(t, s, `{"input":[[1,2],[3]]}`, nil)
		if code != http.StatusOK {
			t.Fatalf("id batch %d %s", code, bodyOut)
		}
		var idsResp EmbeddingsResponse
		if err := json.Unmarshal([]byte(bodyOut), &idsResp); err != nil {
			t.Fatal(err)
		}
		if len(idsResp.Data) != 2 || idsResp.Usage.PromptTokens != 3 {
			t.Fatalf("id usage %+v", idsResp.Usage)
		}

		refusals := []struct {
			body     string
			sentinel error
		}{
			{`{"input":["a","b","c","d","e","f","g","h","i"]}`, ErrEmbeddingWorkerInput},
			{`{"input":["` + strings.Repeat("x", 4097) + `"]}`, ErrEmbeddingWorkerInput},
			{`{"input":` + joinIDRow(257, 64) + `}`, ErrEmbeddingWorkerInput},
			{`{"input":[` + joinIDRow(205, 64) + "," + joinIDRow(205, 64) + "," + joinIDRow(205, 64) + "," + joinIDRow(205, 64) + "," + joinIDRow(205, 64) + `]}`, ErrEmbeddingWorkerInput},
			{`{"input":["a",[1]]}`, ErrEmbeddingWorkerInput},
			{`{"input":[]}`, ErrEmbeddingWorkerInput},
			{`{"input":[null]}`, ErrEmbeddingWorkerInput},
		}
		for _, rc := range refusals {
			code, bodyOut, _ := postEmbeddings(t, s, rc.body, nil)
			if code != http.StatusBadRequest || strings.Contains(bodyOut, `"data"`) {
				t.Fatalf("refusal %s -> %d %s", rc.body[:min(40, len(rc.body))], code, bodyOut)
			}
			if _, err := wk.admit(mustUnmarshalRequest(t, rc.body)); !errors.Is(err, rc.sentinel) {
				t.Fatalf("sentinel for %s: %v", rc.body[:min(40, len(rc.body))], err)
			}
		}

		// The 64 KiB raw-input admission bound, isolated from the text-length cap.
		big := append(append([]byte{'"'}, strings.Repeat("z", embedWorkerMaxInputBytes+1)...), '"')
		if _, _, err := workerSequences(json.RawMessage(big)); !errors.Is(err, ErrEmbeddingWorkerInput) {
			t.Fatalf("raw-size bound: %v", err)
		}
	})

	t.Run("unexpected tokens or non-length finish fail", func(t *testing.T) {
		fake := &encoderContractFake{finish: EmbeddingFinishLength}
		wk := encoderWorkerFixture(t, fake)
		installEncoderWorker(t, wk)
		s := &Server{model: "tiny-encoder", metrics: newGatewayMetrics(time.Now())}

		code, bodyOut, _ := postEmbeddings(t, s, `{"input":[[64]]}`, nil)
		if code != http.StatusBadRequest || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("out-of-vocabulary %d %s", code, bodyOut)
		}
		if _, err := wk.admit(mustUnmarshalRequest(t, `{"input":[[64]]}`)); !errors.Is(err, ErrEmbeddingWorkerVocab) {
			t.Fatalf("vocab sentinel: %v", err)
		}
		if fake.calls != 0 {
			t.Fatalf("out-of-vocabulary ID reached the encoder: calls=%d", fake.calls)
		}

		genFake := &encoderContractFake{finish: EmbeddingFinishLength, generated: true}
		genWk := encoderWorkerFixture(t, genFake)
		installEncoderWorker(t, genWk)
		code, bodyOut, _ = postEmbeddings(t, s, `{"input":"ab"}`, nil)
		if code != http.StatusUnprocessableEntity || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("generated tokens %d %s", code, bodyOut)
		}
		errBad := completionRefusal(EmbeddingFinishError, false)
		if !errors.Is(errBad, ErrEmbeddingWorkerCompletion) || !errors.Is(errBad, ErrEmbeddingWorker) {
			t.Fatalf("completion sentinel: %v", errBad)
		}
		if errGen := completionRefusal(EmbeddingFinishLength, true); !errors.Is(errGen, ErrEmbeddingWorkerCompletion) {
			t.Fatalf("generated-token sentinel: %v", errGen)
		}
		if errEOS := completionRefusal(EmbeddingFinishEOS, false); errEOS != nil {
			t.Fatalf("eos terminal refused: %v", errEOS)
		}
		if errLen := completionRefusal(EmbeddingFinishLength, false); errLen != nil {
			t.Fatalf("length terminal refused: %v", errLen)
		}
		if errCanceled := completionRefusal(EmbeddingFinishCanceled, false); !errors.Is(errCanceled, ErrEmbeddingWorkerCompletion) {
			t.Fatalf("canceled finish sentinel: %v", errCanceled)
		}
	})

	t.Run("borrowed storage survives cancellation to completion", func(t *testing.T) {
		fake := &encoderContractFake{finish: EmbeddingFinishCanceled}
		wk := encoderWorkerFixture(t, fake)
		installEncoderWorker(t, wk)
		s := &Server{model: "tiny-encoder", metrics: newGatewayMetrics(time.Now())}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		code, bodyOut, _ := postEmbeddings(t, s, `{"input":["ab","cd"]}`, ctx)
		if code != http.StatusRequestTimeout || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("cancelled request %d %s", code, bodyOut)
		}
		if fake.calls != 1 || !fake.sawCancel {
			t.Fatalf("encoder did not observe the cancellation: calls=%d sawCancel=%v", fake.calls, fake.sawCancel)
		}
		if !fake.wroteAll {
			t.Fatal("borrowed storage was not written through to completion")
		}
		if got := wk.outstandingVectors(); got != 0 {
			t.Fatalf("buffer did not return to its pool exactly once: outstanding=%d", got)
		}
	})

	t.Run("unwritten vectors cannot become successful embeddings", func(t *testing.T) {
		fake := &encoderContractFake{finish: EmbeddingFinishLength, unwritten: 1}
		wk := encoderWorkerFixture(t, fake)
		installEncoderWorker(t, wk)
		s := &Server{model: "tiny-encoder", metrics: newGatewayMetrics(time.Now())}

		code, bodyOut, _ := postEmbeddings(t, s, `{"input":[[1,2],[3]]}`, nil)
		if code != http.StatusUnprocessableEntity || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("unwritten row %d %s", code, bodyOut)
		}
		if got := firstUnwrittenRow([]float64{0, 0, math.NaN()}, 1, 3); got != 0 {
			t.Fatalf("firstUnwrittenRow NaN index %d", got)
		}
		if got := firstUnwrittenRow([]float64{0, 0, 0}, 1, 3); got != -1 {
			t.Fatalf("firstUnwrittenRow clean %d", got)
		}
		if got := wk.outstandingVectors(); got != 0 {
			t.Fatalf("failed call leaked a buffer: outstanding=%d", got)
		}
	})

	t.Run("mounted HTTP wiring", func(t *testing.T) {
		fake := &encoderContractFake{finish: EmbeddingFinishLength}
		wk := encoderWorkerFixture(t, fake)
		installEncoderWorker(t, wk)
		s := &Server{model: "tiny-encoder", metrics: newGatewayMetrics(time.Now())}

		// 200 matrix: one EmbeddingData per input in request order, pooling
		// identity header, exact usage.
		code, bodyOut, header := postEmbeddings(t, s, `{"input":["one two","three"]}`, nil)
		if code != http.StatusOK {
			t.Fatalf("wired 200 %d %s", code, bodyOut)
		}
		var resp EmbeddingsResponse
		if err := json.Unmarshal([]byte(bodyOut), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Object != "list" || len(resp.Data) != 2 || resp.Model != "tiny-encoder" {
			t.Fatalf("envelope %+v", resp)
		}
		if resp.Data[0].Index != 0 || resp.Data[1].Index != 1 || resp.Data[0].Object != "embedding" || resp.Data[1].Object != "embedding" {
			t.Fatalf("ordering %+v", resp.Data)
		}
		if header.Get("X-Fak-Embedding-Mode") != "encoder-mean" {
			t.Fatalf("pooling identity header %q", header.Get("X-Fak-Embedding-Mode"))
		}
		wantTokens := len(tokenizeForEmbedding("one two")) + len(tokenizeForEmbedding("three"))
		if resp.Usage.PromptTokens != wantTokens || resp.Usage.TotalTokens != wantTokens {
			t.Fatalf("usage %+v, want %d", resp.Usage, wantTokens)
		}

		// Base64 form round-trips through little-endian float32 std-base64 to the
		// same values as the float form.
		floats, ok := resp.Data[0].Embedding.([]any)
		if !ok {
			t.Fatalf("float form %T", resp.Data[0].Embedding)
		}
		code, bodyOut, _ = postEmbeddings(t, s, `{"input":["one two"],"encoding_format":"base64"}`, nil)
		if code != http.StatusOK {
			t.Fatalf("base64 200 %d %s", code, bodyOut)
		}
		var b64Resp EmbeddingsResponse
		if err := json.Unmarshal([]byte(bodyOut), &b64Resp); err != nil {
			t.Fatal(err)
		}
		raw, ok := b64Resp.Data[0].Embedding.(string)
		if !ok {
			t.Fatalf("base64 form %T", b64Resp.Data[0].Embedding)
		}
		buf, err := base64.StdEncoding.DecodeString(raw)
		if err != nil || len(buf) != 4*wk.HiddenDim {
			t.Fatalf("base64 vector %d bytes: %v", len(buf), err)
		}
		for i, v := range floats {
			f64, ok := v.(float64)
			if !ok {
				t.Fatalf("float component %T", v)
			}
			if math.Float32bits(float32(f64)) != binary.LittleEndian.Uint32(buf[i*4:]) {
				t.Fatalf("component %d: float32(%v) != packed little-endian bits", i, f64)
			}
		}

		// Refusal matrix over the wire: capability 400, cancel 408 with zero data
		// rows, terminal-completion refusal 422.
		if code, bodyOut, _ := postEmbeddings(t, s, `{"input":"ab","dimensions":16}`, nil); code != http.StatusBadRequest || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("capability 400 %d %s", code, bodyOut)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if code, bodyOut, _ := postEmbeddings(t, s, `{"input":"ab"}`, ctx); code != http.StatusRequestTimeout || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("cancel 408 %d %s", code, bodyOut)
		}
		genFake := &encoderContractFake{finish: EmbeddingFinishLength, generated: true}
		installEncoderWorker(t, encoderWorkerFixture(t, genFake))
		if code, bodyOut, _ := postEmbeddings(t, s, `{"input":"ab"}`, nil); code != http.StatusUnprocessableEntity || strings.Contains(bodyOut, `"data"`) {
			t.Fatalf("terminal refusal 422 %d %s", code, bodyOut)
		}

		// Without a worker: byte-for-byte historical hashing backend.
		SetEmbeddingEncoderWorker(nil)
		code, bodyOut, header = postEmbeddings(t, s, `{"input":"a b","dimensions":3}`, nil)
		if code != http.StatusOK || header.Get("X-Fak-Embedding-Mode") != "" {
			t.Fatalf("hash default changed %d %q", code, header.Get("X-Fak-Embedding-Mode"))
		}
		var hashResp EmbeddingsResponse
		if err := json.Unmarshal([]byte(bodyOut), &hashResp); err != nil {
			t.Fatal(err)
		}
		want, n := embedText("a b", 3)
		gotJSON, _ := json.Marshal(hashResp.Data[0].Embedding)
		wantJSON, _ := json.Marshal(want)
		if string(gotJSON) != string(wantJSON) || hashResp.Usage.TotalTokens != n {
			t.Fatal("hash output changed")
		}
	})

	t.Run("worker precedes causal fallback", func(t *testing.T) {
		cfg, p := causalGatewayFixture(t)
		fake := &encoderContractFake{finish: EmbeddingFinishLength}
		wk := encoderWorkerFixture(t, fake)
		installEncoderWorker(t, wk)
		s := &Server{model: cfg.Model, causalEmbeddings: p, metrics: newGatewayMetrics(time.Now())}
		code, _, header := postEmbeddings(t, s, `{"input":"ab"}`, nil)
		if code != http.StatusOK || header.Get("X-Fak-Embedding-Mode") != "encoder-mean" {
			t.Fatalf("precedence %d %q", code, header.Get("X-Fak-Embedding-Mode"))
		}
		if fake.calls != 1 {
			t.Fatalf("causal provider answered ahead of the encoder worker: calls=%d", fake.calls)
		}
	})
}

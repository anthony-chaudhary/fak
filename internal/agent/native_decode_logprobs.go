package agent

import (
	"fmt"
	"math"
)

const (
	NativeDecodeTopLogprobsSchema = "fak.native-decode-top-logprobs/1"
	NativeDecodeTopLogprobsEngine = "fak-native"
	// MaxNativeDecodeTopLogprobs bounds the per-token candidate list. It matches
	// the OpenAI chat wire's top_logprobs ceiling; the legacy wire caps at 5.
	MaxNativeDecodeTopLogprobs = 20
)

// NativeTokenLogprob is one candidate token at a generated position: its id,
// its decoded piece, and its log-softmax over the full vocabulary of the
// logits the sampler read.
type NativeTokenLogprob struct {
	TokenID int     `json:"token_id"`
	Token   string  `json:"token,omitempty"`
	Logprob float64 `json:"logprob"`
}

// NativeDecodeTopLogprobs is the ordered per-generated-token top-k receipt.
// Rows[i] belongs to TokenIDs[i]; each row is sorted by descending logprob
// (ties by ascending id) and Chosen[i] carries the selected token's own entry,
// which under greedy decoding equals Rows[i][0].
type NativeDecodeTopLogprobs struct {
	Schema   string                 `json:"schema"`
	Engine   string                 `json:"engine"`
	K        int                    `json:"k"`
	TokenIDs []int                  `json:"token_ids"`
	Chosen   []NativeTokenLogprob   `json:"chosen"`
	Rows     [][]NativeTokenLogprob `json:"rows"`
}

// WithNativeDecodeTopLogprobs asks native decode to record, for every generated
// token, the chosen token's logprob and the top-k candidates of the same logits.
// k <= 0 leaves it off (the default): no logits scan, no allocation, and device
// greedy selection stays available. Requesting it forces host logits, as the
// strict receipt does, so the sampler and the receipt read the same row.
func WithNativeDecodeTopLogprobs(k int) SampleOpt {
	return func(sp *SampleParams) {
		if k < 0 {
			k = 0
		}
		if k > MaxNativeDecodeTopLogprobs {
			k = MaxNativeDecodeTopLogprobs
		}
		sp.NativeDecodeTopLogprobs = k
	}
}

// nativeTopLogprobs returns the chosen token's log-softmax and the k most
// probable candidates of logits, computed in float64 over the full vocabulary.
func nativeTopLogprobs(logits []float32, token, k int) (NativeTokenLogprob, []NativeTokenLogprob, error) {
	if token < 0 || token >= len(logits) {
		return NativeTokenLogprob{}, nil, fmt.Errorf("chosen token %d outside logits size %d", token, len(logits))
	}
	if k > len(logits) {
		k = len(logits)
	}
	maxLogit := math.Inf(-1)
	for _, raw := range logits {
		v := float64(raw)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return NativeTokenLogprob{}, nil, fmt.Errorf("non-finite model logit")
		}
		if v > maxLogit {
			maxLogit = v
		}
	}
	// top holds candidate ids ordered by (logit desc, id asc); insertion keeps
	// the scan O(V*k) with k <= 20 and no full-vocab sort or allocation.
	top := make([]int, 0, k)
	var expSum float64
	for id, raw := range logits {
		expSum += math.Exp(float64(raw) - maxLogit)
		if k == 0 {
			continue
		}
		if len(top) == k && raw <= logits[top[k-1]] {
			continue
		}
		pos := len(top)
		for pos > 0 && raw > logits[top[pos-1]] {
			pos--
		}
		if len(top) < k {
			top = append(top, 0)
		}
		copy(top[pos+1:], top[pos:len(top)-1])
		top[pos] = id
	}
	logZ := maxLogit + math.Log(expSum)
	lp := func(id int) (float64, error) {
		v := float64(logits[id]) - logZ
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, fmt.Errorf("non-finite token log probability")
		}
		return min(v, 0), nil
	}
	chosenLP, err := lp(token)
	if err != nil {
		return NativeTokenLogprob{}, nil, err
	}
	row := make([]NativeTokenLogprob, len(top))
	for i, id := range top {
		v, err := lp(id)
		if err != nil {
			return NativeTokenLogprob{}, nil, err
		}
		row[i] = NativeTokenLogprob{TokenID: id, Logprob: v}
	}
	return NativeTokenLogprob{TokenID: token, Logprob: chosenLP}, row, nil
}

// buildNativeDecodeTopLogprobs copies the measurement's rows into the response
// receipt and attaches decoded pieces. decode may be nil.
func buildNativeDecodeTopLogprobs(m *nativeInferenceMeasurement, decode func([]int) (string, error)) *NativeDecodeTopLogprobs {
	if m == nil || m.topLogprobsK <= 0 {
		return nil
	}
	piece := func(id int) string {
		if decode == nil {
			return ""
		}
		s, err := decode([]int{id})
		if err != nil {
			return ""
		}
		return s
	}
	out := &NativeDecodeTopLogprobs{
		Schema:   NativeDecodeTopLogprobsSchema,
		Engine:   NativeDecodeTopLogprobsEngine,
		K:        m.topLogprobsK,
		TokenIDs: make([]int, len(m.topChosen)),
		Chosen:   make([]NativeTokenLogprob, len(m.topChosen)),
		Rows:     make([][]NativeTokenLogprob, len(m.topRows)),
	}
	for i, c := range m.topChosen {
		c.Token = piece(c.TokenID)
		out.TokenIDs[i] = c.TokenID
		out.Chosen[i] = c
	}
	for i, row := range m.topRows {
		cp := make([]NativeTokenLogprob, len(row))
		for j, c := range row {
			c.Token = piece(c.TokenID)
			cp[j] = c
		}
		out.Rows[i] = cp
	}
	return out
}

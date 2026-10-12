package computetrace

// llamacpp_dump.go -- convert a llama.cpp tensor-dump directory into a bounded
// activation-sample reference artifact.
//
// The reference runtime for DeepSeek V4.1 Flash bring-up is a llama.cpp fork
// that writes named intermediate tensors to disk. ConvertLlamaCppDump reads that
// dump and samples it with the exact rule fak's own V4.1 producer uses
// (ActivationSampleIndices, 8 points per token row), so the result is directly
// comparable with ActivationErrorProfile.
//
// Dump layout (<dir>):
//
//   - index.jsonl  one JSON object per line:
//     {"seq","name","op","type","ne":[4],"dumped","file","sum","sumsq","maxabs","nonfinite"}
//   - <file>       raw little-endian f32 tensor data, relative to <dir>
//   - tokens.json  {"prompt","tokens":[...]}
//
// Names are "<base>-<layer>". Mapped bases (unknown bases are ignored):
//
//	qr       -> q_latent             width ne0
//	kv       -> kv_latent            width ne0; the fork captures kv AFTER RoPE
//	                                 while fak samples it before, so sample
//	                                 indices in the last 64 dims (the RoPE tail)
//	                                 are listed in ExcludedSampleIndices
//	attn_out -> attention_projected  width ne0
//	ffn_out  -> moe_sum              width ne0
//	l_last   -> residual_streams     width ne0*ne1 (ne1 = hc streams)
//
// ggml stores ne0 fastest, so a [ne0=H, ne1=hc, ne2=tokens] tensor is [t][hc][i]
// in memory -- the same order as fak's streams[t][s][i] flattened. Every
// remaining dimension counts tokens. A name that repeats (a later ubatch)
// continues its token numbering where the previous occurrence ended.
//
// The converted events carry a placeholder input digest derived from the dump's
// tokens and NO weight digest, so the strict comparators refuse them as-is. Run
// AdoptActivationDigests against the fak trace (which proves the token ids
// match) before profiling.

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// LlamaCppActivationSamples is the per-token sample count the converter takes,
// matching fak's V4.1 activation producer.
const LlamaCppActivationSamples = 8

// llamaCppRoPETail is the trailing kv dimension span the fork has rotated.
const llamaCppRoPETail = 64

type llamaCppStage struct {
	stage     string
	widthDims int
	ropeTail  int
}

var llamaCppStages = map[string]llamaCppStage{
	"qr":       {stage: "q_latent", widthDims: 1},
	"kv":       {stage: "kv_latent", widthDims: 1, ropeTail: llamaCppRoPETail},
	"attn_out": {stage: "attention_projected", widthDims: 1},
	"ffn_out":  {stage: "moe_sum", widthDims: 1},
	"l_last":   {stage: "residual_streams", widthDims: 2},
}

// ErrLlamaCppDump wraps every malformed-dump refusal.
var ErrLlamaCppDump = errors.New("computetrace: malformed llama.cpp dump")

type llamaCppIndexEntry struct {
	Seq    int     `json:"seq"`
	Name   string  `json:"name"`
	Op     string  `json:"op"`
	Type   string  `json:"type"`
	Ne     []int64 `json:"ne"`
	Dumped bool    `json:"dumped"`
	File   string  `json:"file"`
}

type llamaCppTokens struct {
	Prompt string `json:"prompt"`
	Tokens []int  `json:"tokens"`
}

func dumpErr(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrLlamaCppDump}, args...)...)
}

// ConvertLlamaCppDump reads a llama.cpp dump directory and returns a sampled
// reference Artifact whose InputTokens are the dump's tokens.
func ConvertLlamaCppDump(dir string) (Artifact, error) {
	var toks llamaCppTokens
	raw, err := os.ReadFile(filepath.Join(dir, "tokens.json"))
	if err != nil {
		return Artifact{}, err
	}
	if err := json.Unmarshal(raw, &toks); err != nil {
		return Artifact{}, dumpErr("tokens.json: %v", err)
	}
	if len(toks.Tokens) == 0 {
		return Artifact{}, dumpErr("tokens.json lists no tokens")
	}
	entries, err := readLlamaCppIndex(filepath.Join(dir, "index.jsonl"))
	if err != nil {
		return Artifact{}, err
	}

	var tokenText strings.Builder
	for i, id := range toks.Tokens {
		if i != 0 {
			tokenText.WriteByte(',')
		}
		tokenText.WriteString(strconv.Itoa(id))
	}
	inputDigest := Digest("llamacpp-tokens", tokenText.String())

	type rowKey struct {
		stage string
		layer int
	}
	nextToken := map[rowKey]int{}
	art := Artifact{Schema: Schema, InputTokens: append([]int(nil), toks.Tokens...)}
	for _, e := range entries {
		base, layer, ok := splitLlamaCppName(e.Name)
		if !ok {
			continue
		}
		spec, ok := llamaCppStages[base]
		if !ok || !e.Dumped {
			continue
		}
		if e.Type != "f32" {
			return Artifact{}, dumpErr("%s: tensor type %q, want f32", e.Name, e.Type)
		}
		if len(e.Ne) != 4 {
			return Artifact{}, dumpErr("%s: ne has %d dims, want 4", e.Name, len(e.Ne))
		}
		width, tokens, byteSize := int64(1), int64(1), int64(4)
		maxBytes := int64(^uint(0) >> 1)
		for d, n := range e.Ne {
			if n <= 0 {
				return Artifact{}, dumpErr("%s: ne[%d]=%d", e.Name, d, n)
			}
			// A representable byte size also bounds width, tokens and sample offsets.
			if n > maxBytes/byteSize {
				return Artifact{}, dumpErr("%s: f32 byte size exceeds int for ne=%v", e.Name, e.Ne)
			}
			byteSize *= n
			if d < spec.widthDims {
				width *= n
			} else {
				tokens *= n
			}
		}
		if !filepath.IsLocal(e.File) {
			return Artifact{}, dumpErr("%s: file %q is not local to the dump", e.Name, e.File)
		}
		data, err := os.ReadFile(filepath.Join(dir, e.File))
		if err != nil {
			return Artifact{}, err
		}
		if int64(len(data)) != byteSize {
			return Artifact{}, dumpErr("%s: %d bytes, want %d for ne=%v", e.Name, len(data), byteSize, e.Ne)
		}
		w := int(width)
		idx := ActivationSampleIndices(w, LlamaCppActivationSamples)
		var excluded []int
		if spec.ropeTail > 0 {
			for _, j := range idx {
				if j >= w-spec.ropeTail {
					excluded = append(excluded, j)
				}
			}
		}
		key := rowKey{stage: spec.stage, layer: layer}
		for t := 0; t < int(tokens); t++ {
			vals := make([]float32, len(idx))
			for i, j := range idx {
				off := 4 * (t*w + j)
				vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))
			}
			art.Events = append(art.Events, Event{
				Sequence:              int64(len(art.Events) + 1),
				Operation:             "llamacpp.dump",
				Phase:                 spec.stage,
				Backend:               "llamacpp",
				Device:                "reference",
				Layer:                 layer,
				Token:                 nextToken[key] + t,
				InputDigest:           inputDigest,
				Shapes:                [][]int{{w}},
				SampleIndices:         append([]int(nil), idx...),
				SampleValues:          vals,
				ExcludedSampleIndices: append([]int(nil), excluded...),
				ProvenanceDigest:      Digest("llamacpp-dump", e.Name, strconv.Itoa(e.Seq)),
				Status:                "ok",
			})
			art.RetainedSampleValues += uint64(len(vals))
		}
		nextToken[key] += int(tokens)
	}
	return art, nil
}

func readLlamaCppIndex(path string) ([]llamaCppIndexEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []llamaCppIndexEntry
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var e llamaCppIndexEntry
		if err := json.Unmarshal([]byte(text), &e); err != nil {
			return nil, dumpErr("index.jsonl line %d: %v", line, err)
		}
		out = append(out, e)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out, nil
}

// splitLlamaCppName splits "<base>-<layer>" at the last '-'.
func splitLlamaCppName(name string) (string, int, bool) {
	cut := strings.LastIndexByte(name, '-')
	if cut <= 0 {
		return "", 0, false
	}
	layer, err := strconv.Atoi(name[cut+1:])
	if err != nil || layer < 0 {
		return "", 0, false
	}
	return name[:cut], layer, true
}

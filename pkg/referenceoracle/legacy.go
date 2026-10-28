package referenceoracle

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// legacyDoc is the per-model receipt shape (fak.v41_reference_oracle.v1 and the
// same field set under another model's schema id). The adapter reads:
//
//	schema, evidence_label, host, captured_utc
//	reference_engine.{repo, commit, license, binary_sha256, sha256{<binary>: <digest>}}
//	model.{repo_id, revision, quant, shard1, shard1_sha256, shards: <count> | [{name, sha256, size_bytes}],
//	      rebind: {from_revision, to_revision, witnessed_utc, matched}}
//	prompts{<text>: {input_token_ids, greedy_token_ids, top5[[{id, piece, logprob}]]}}
//	determinism.{method, results[{greedy_tokens_identical, top5_logprobs_identical_1e-6}]}
//	trace_refs (fak.reference_oracle.v1 shape)
//
// When binary_sha256 is absent the digest of llama-server, then llama-cli, then
// a lone sha256 entry is used. Prompts keep their document order.
type legacyDoc struct {
	Schema          string `json:"schema"`
	EvidenceLabel   string `json:"evidence_label"`
	Host            string `json:"host"`
	CapturedUTC     string `json:"captured_utc"`
	ReferenceEngine struct {
		Repo         string            `json:"repo"`
		Commit       string            `json:"commit"`
		License      string            `json:"license"`
		BinarySHA256 string            `json:"binary_sha256"`
		SHA256       map[string]string `json:"sha256"`
	} `json:"reference_engine"`
	Model struct {
		RepoID       string          `json:"repo_id"`
		Revision     string          `json:"revision"`
		Quant        string          `json:"quant"`
		Shard1       string          `json:"shard1"`
		Shard1SHA256 string          `json:"shard1_sha256"`
		Shards       json.RawMessage `json:"shards"`
		Rebind       *struct {
			FromRevision string   `json:"from_revision"`
			ToRevision   string   `json:"to_revision"`
			WitnessedUTC string   `json:"witnessed_utc"`
			Matched      []string `json:"matched"`
		} `json:"rebind"`
	} `json:"model"`
	Prompts     json.RawMessage `json:"prompts"`
	Determinism *struct {
		Method  string `json:"method"`
		Results []struct {
			Prompt string `json:"prompt"`
			Greedy bool   `json:"greedy_tokens_identical"`
			Top5   bool   `json:"top5_logprobs_identical_1e-6"`
		} `json:"results"`
	} `json:"determinism"`
	TraceRefs []TraceRef `json:"trace_refs"`
}

type legacyPrompt struct {
	InputTokenIDs  []int         `json:"input_token_ids"`
	GreedyTokenIDs []int         `json:"greedy_token_ids"`
	Top5           [][]TokenProb `json:"top5"`
}

func adaptLegacy(data []byte) (Oracle, error) {
	var d legacyDoc
	if err := json.Unmarshal(data, &d); err != nil {
		return Oracle{}, err
	}
	o := Oracle{
		Schema:        Schema,
		SourceSchema:  d.Schema,
		EvidenceLabel: d.EvidenceLabel,
		Host:          d.Host,
		CapturedUTC:   d.CapturedUTC,
		Reference: Reference{
			Repo:         d.ReferenceEngine.Repo,
			SHA:          d.ReferenceEngine.Commit,
			License:      d.ReferenceEngine.License,
			BinarySHA256: d.ReferenceEngine.BinarySHA256,
		},
		Artifact: Artifact{
			Repo:  d.Model.RepoID,
			Rev:   d.Model.Revision,
			Quant: d.Model.Quant,
		},
		TraceRefs: d.TraceRefs,
	}
	if o.Reference.BinarySHA256 == "" {
		o.Reference.BinarySHA256 = pickBinaryDigest(d.ReferenceEngine.SHA256)
	}
	if err := decodeShards(d.Model.Shards, &o.Artifact); err != nil {
		return Oracle{}, err
	}
	if len(o.Artifact.Shards) == 0 && d.Model.Shard1 != "" {
		o.Artifact.Shards = []Shard{{Name: d.Model.Shard1, SHA256: d.Model.Shard1SHA256}}
	}
	if d.Model.Rebind != nil {
		o.Artifact.Rebind = &Rebind{
			FromRev:      d.Model.Rebind.FromRevision,
			ToRev:        d.Model.Rebind.ToRevision,
			WitnessedUTC: d.Model.Rebind.WitnessedUTC,
			Matched:      d.Model.Rebind.Matched,
		}
	}
	prompts, err := orderedPrompts(d.Prompts)
	if err != nil {
		return Oracle{}, err
	}
	o.Prompts = prompts
	if d.Determinism != nil {
		o.SelfParity = legacySelfParity(d.Determinism.Method, d.Determinism.Results, prompts)
	}
	return o, nil
}

// legacySelfParity is Checked only when the second run covered every prompt
// exactly once (by prompt text when results name one, else by count), and
// Identical only when it is Checked and every prompt reproduced.
func legacySelfParity(method string, results []struct {
	Prompt string `json:"prompt"`
	Greedy bool   `json:"greedy_tokens_identical"`
	Top5   bool   `json:"top5_logprobs_identical_1e-6"`
}, prompts []Prompt) SelfParity {
	sp := SelfParity{Method: method}
	if len(results) != len(prompts) || len(prompts) == 0 {
		return sp
	}
	want := map[string]bool{}
	for _, p := range prompts {
		want[p.Text] = true
	}
	identical := true
	for _, r := range results {
		if r.Prompt != "" {
			if !want[r.Prompt] {
				return sp
			}
			delete(want, r.Prompt)
		}
		identical = identical && r.Greedy && r.Top5
	}
	if results[0].Prompt != "" && len(want) != 0 {
		return sp
	}
	sp.Checked, sp.Identical = true, identical
	return sp
}

func pickBinaryDigest(m map[string]string) string {
	for _, k := range []string{"llama-server", "llama-cli"} {
		if v := m[k]; v != "" {
			return v
		}
	}
	if len(m) == 1 {
		for _, v := range m {
			return v
		}
	}
	return ""
}

// orderedPrompts decodes a JSON object keyed by prompt text, keeping key order.
func orderedPrompts(raw json.RawMessage) ([]Prompt, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return nil, fmt.Errorf("prompts: want an object keyed by prompt text")
	}
	var out []Prompt
	seen := map[string]bool{}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		text, _ := keyTok.(string)
		if seen[text] {
			return nil, fmt.Errorf("prompts: duplicate prompt %q", text)
		}
		seen[text] = true
		var p legacyPrompt
		if err := dec.Decode(&p); err != nil {
			return nil, fmt.Errorf("prompts[%q]: %v", text, err)
		}
		out = append(out, Prompt{Text: text, InputTokenIDs: p.InputTokenIDs, GreedyIDs: p.GreedyTokenIDs, Top5: p.Top5})
	}
	return out, nil
}

// decodeShards reads model.shards as either a shard count or a shard list.
func decodeShards(raw json.RawMessage, a *Artifact) error {
	raw = bytes.TrimSpace(raw)
	switch {
	case len(raw) == 0 || string(raw) == "null":
		return nil
	case raw[0] == '[':
		if err := json.Unmarshal(raw, &a.Shards); err != nil {
			return fmt.Errorf("model.shards: %v", err)
		}
		a.ShardCount = len(a.Shards)
		return nil
	default:
		if err := json.Unmarshal(raw, &a.ShardCount); err != nil {
			return fmt.Errorf("model.shards: want a count or a list: %v", err)
		}
		return nil
	}
}

package amoprofpub

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type analysisFact struct{ Source, Key, Value string }

func analysisBody(title, stage string, files []File) string {
	facts := collectAnalysisFacts(stage, files)
	var b strings.Builder
	b.WriteString("<h1>" + html.EscapeString(title) + "</h1>")
	b.WriteString("<p>This is a deterministic reading of the machine-readable AMOProf outputs. It follows the original report rather than replacing it.</p>")
	b.WriteString("<h2>Extracted findings</h2>")
	if len(facts) == 0 {
		b.WriteString("<p>No supported machine-readable summary was present. Use the original report and source files.</p>")
	} else {
		b.WriteString("<table><tbody><tr><th>Source</th><th>Field</th><th>Value</th></tr>")
		for _, f := range facts {
			b.WriteString("<tr><td><code>" + html.EscapeString(f.Source) + "</code></td><td>" + html.EscapeString(f.Key) + "</td><td>" + html.EscapeString(f.Value) + "</td></tr>")
		}
		b.WriteString("</tbody></table>")
	}
	b.WriteString("<h2>Interpretation boundary</h2><p>These values describe this captured run and its collection coverage. They do not by themselves establish generated-code correctness, causal attribution, or performance on another workload or machine.</p>")
	return b.String()
}

func collectAnalysisFacts(stage string, files []File) []analysisFact {
	wanted := map[string]bool{"summary.json": true, "collection_validation.json": true, "analysis_manifest.json": true, "metrics_by_ai_operation.csv": true, "summary.csv": true}
	var out []analysisFact
	for _, f := range files {
		base := strings.ToLower(filepath.Base(f.Path))
		if !wanted[base] {
			continue
		}
		data, err := os.ReadFile(filepath.Join(stage, filepath.FromSlash(f.Path)))
		if err != nil {
			continue
		}
		if strings.HasSuffix(base, ".json") {
			var v any
			if json.Unmarshal(data, &v) == nil {
				flattenFacts(&out, f.Path, "", v, 0)
			}
		} else {
			out = append(out, csvFacts(f.Path, data)...)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Source != out[j].Source {
			return out[i].Source < out[j].Source
		}
		return out[i].Key < out[j].Key
	})
	if len(out) > 160 {
		out = out[:160]
	}
	return out
}
func flattenFacts(out *[]analysisFact, source, key string, v any, depth int) {
	if depth > 4 || len(*out) >= 200 {
		return
	}
	switch x := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			nk := k
			if key != "" {
				nk = key + "." + k
			}
			flattenFacts(out, source, nk, x[k], depth+1)
		}
	case []any:
		if len(x) <= 12 {
			for i, e := range x {
				flattenFacts(out, source, key+"["+strconv.Itoa(i)+"]", e, depth+1)
			}
		} else {
			*out = append(*out, analysisFact{source, key, fmt.Sprintf("%d items", len(x))})
		}
	case string:
		if x != "" && len(x) <= 500 {
			*out = append(*out, analysisFact{source, key, x})
		}
	case float64, bool, nil:
		*out = append(*out, analysisFact{source, key, fmt.Sprint(x)})
	}
}
func csvFacts(source string, data []byte) []analysisFact {
	rows, err := csv.NewReader(strings.NewReader(string(data))).ReadAll()
	if err != nil || len(rows) < 2 {
		return nil
	}
	limit := len(rows)
	if limit > 13 {
		limit = 13
	}
	var out []analysisFact
	for i := 1; i < limit; i++ {
		for j, v := range rows[i] {
			if v == "" {
				continue
			}
			key := fmt.Sprintf("row %d", i)
			if j < len(rows[0]) {
				key += " / " + rows[0][j]
			}
			out = append(out, analysisFact{source, key, v})
		}
	}
	return out
}

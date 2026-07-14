package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const schema = "fak-qwen36-coherence-gate/1"

var wordRE = regexp.MustCompile(`[\pL\pN]+(?:['’-][\pL\pN]+)*`)

type Manifest struct {
	Schema    string   `json:"schema,omitempty"`
	Model     string   `json:"model"`
	MaxTokens int      `json:"max_tokens,omitempty"`
	Prompts   []Prompt `json:"prompts"`
}

type Prompt struct {
	Bucket         string   `json:"bucket"`
	Path           string   `json:"path"`
	RequiredLabels []string `json:"required_labels,omitempty"`
}

type Score struct {
	Words            int      `json:"words"`
	TrigramWindows   int      `json:"trigram_windows"`
	RepeatedTrigrams int      `json:"repeated_trigrams"`
	TrigramRepeat    float64  `json:"trigram_repeat"`
	RequiredLabels   []string `json:"required_labels,omitempty"`
	MissingLabels    []string `json:"missing_labels,omitempty"`
	RequiredLabelsOK bool     `json:"required_labels_ok"`
}

type Run struct {
	Bucket         string   `json:"bucket"`
	Mode           string   `json:"mode"`
	Decode         string   `json:"decode"`
	PromptPath     string   `json:"prompt_path"`
	PromptBytes    int      `json:"prompt_bytes"`
	OutputPath     string   `json:"output_path"`
	StderrPath     string   `json:"stderr_path"`
	Command        []string `json:"command"`
	Environment    []string `json:"environment"`
	ElapsedMS      int64    `json:"elapsed_ms"`
	Score          Score    `json:"score"`
	ExecutionError string   `json:"execution_error,omitempty"`
}

type Comparison struct {
	Bucket     string  `json:"bucket"`
	Mode       string  `json:"mode"`
	Q8Repeat   float64 `json:"q8_trigram_repeat"`
	Int8Repeat float64 `json:"int8_trigram_repeat"`
	Delta      float64 `json:"delta"`
	LabelsOK   bool    `json:"required_labels_ok"`
	Pass       bool    `json:"pass"`
	Reason     string  `json:"reason,omitempty"`
}

type Artifact struct {
	Schema      string       `json:"schema"`
	GeneratedAt string       `json:"generated_at"`
	Model       string       `json:"model"`
	MaxTokens   int          `json:"max_tokens"`
	Seed        int64        `json:"seed"`
	Runs        []Run        `json:"runs"`
	Comparisons []Comparison `json:"comparisons"`
	Pass        bool         `json:"pass"`
	Reason      string       `json:"reason,omitempty"`
}

func scoreText(text string, labels []string) Score {
	words := wordRE.FindAllString(strings.ToLower(text), -1)
	total := len(words) - 2
	if total < 0 {
		total = 0
	}
	seen := make(map[string]struct{}, total)
	repeated := 0
	for i := 0; i+2 < len(words); i++ {
		tri := words[i] + "\x00" + words[i+1] + "\x00" + words[i+2]
		if _, ok := seen[tri]; ok {
			repeated++
		} else {
			seen[tri] = struct{}{}
		}
	}
	missing := make([]string, 0)
	folded := strings.ToLower(text)
	for _, label := range labels {
		if !strings.Contains(folded, strings.ToLower(label)) {
			missing = append(missing, label)
		}
	}
	ratio := 0.0
	if total > 0 {
		ratio = float64(repeated) / float64(total)
	}
	return Score{Words: len(words), TrigramWindows: total, RepeatedTrigrams: repeated, TrigramRepeat: ratio, RequiredLabels: labels, MissingLabels: missing, RequiredLabelsOK: len(missing) == 0}
}

func compareRuns(runs []Run, required []Prompt) ([]Comparison, bool, string) {
	type pair struct{ q8, int8 *Run }
	pairs := make(map[string]*pair)
	requiredKeys := make(map[string]struct{})
	for _, prompt := range required {
		for _, mode := range modes() {
			requiredKeys[prompt.Bucket+"\x00"+mode.name] = struct{}{}
		}
	}
	for i := range runs {
		r := &runs[i]
		key := r.Bucket + "\x00" + r.Mode
		p := pairs[key]
		if p == nil {
			p = &pair{}
			pairs[key] = p
		}
		switch r.Decode {
		case "q8":
			p.q8 = r
		case "int8-q4k":
			p.int8 = r
		}
	}
	keys := make([]string, 0, len(requiredKeys))
	for key := range requiredKeys {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]Comparison, 0, len(keys))
	all := len(keys) > 0
	reason := ""
	for _, key := range keys {
		p := pairs[key]
		if p == nil {
			p = &pair{}
		}
		parts := strings.Split(key, "\x00")
		c := Comparison{Bucket: parts[0], Mode: parts[1]}
		if p.q8 == nil || p.int8 == nil {
			c.Reason = "missing q8 or int8-q4k arm"
		} else if p.q8.ExecutionError != "" || p.int8.ExecutionError != "" {
			c.Reason = "one or more generation arms failed"
		} else {
			c.Q8Repeat = p.q8.Score.TrigramRepeat
			c.Int8Repeat = p.int8.Score.TrigramRepeat
			c.Delta = c.Int8Repeat - c.Q8Repeat
			c.LabelsOK = p.q8.Score.RequiredLabelsOK && p.int8.Score.RequiredLabelsOK
			c.Pass = c.Delta <= 0
			if c.Delta > 0 {
				c.Reason = "int8 Q4_K worsened trigram repetition"
			}
		}
		all = all && c.Pass
		out = append(out, c)
	}
	if len(keys) == 0 {
		reason = "manifest contains no required prompt/profile comparisons"
	}
	return out, all, reason
}

func runSelfcheck() error {
	clean := "Summary alpha beta gamma delta epsilon. Risks are bounded. Next steps are explicit."
	loop := "Summary alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma."
	labels := []string{"Summary", "Risks", "Next steps"}
	q8 := Run{Bucket: "long", Mode: "greedy", Decode: "q8", Score: scoreText(clean, labels)}
	bad := Run{Bucket: "long", Mode: "greedy", Decode: "int8-q4k", Score: scoreText(loop, labels)}
	if c, pass, _ := compareRuns([]Run{q8, bad}, []Prompt{{Bucket: "long"}}); pass || len(c) != 2 || c[0].Reason == "" {
		return errors.New("red fixture did not fail closed")
	}
	good := q8
	good.Decode = "int8-q4k"
	q8Recommended := q8
	q8Recommended.Mode = "recommended"
	goodRecommended := good
	goodRecommended.Mode = "recommended"
	if _, pass, _ := compareRuns([]Run{q8, good, q8Recommended, goodRecommended}, []Prompt{{Bucket: "long"}}); !pass {
		return errors.New("equal-repeat fixture did not pass")
	}
	fmt.Printf("SELF_CHECK PASS clean_repeat=%.6f loop_repeat=%.6f red_fixture=REJECT green_fixture=PASS\n", q8.Score.TrigramRepeat, bad.Score.TrigramRepeat)
	return nil
}

func modes() []struct {
	name string
	args []string
} {
	return []struct {
		name string
		args []string
	}{
		{name: "greedy"},
		{name: "recommended", args: []string{"--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--presence-penalty", "1.5"}},
	}
}

func execute(manifestPath, fakPath, outDir string, seed int64) (Artifact, error) {
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return Artifact{}, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return Artifact{}, err
	}
	if m.Model == "" || len(m.Prompts) == 0 {
		return Artifact{}, errors.New("manifest needs model and prompts")
	}
	if m.MaxTokens <= 0 {
		m.MaxTokens = 512
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return Artifact{}, err
	}
	base := filepath.Dir(manifestPath)
	a := Artifact{Schema: schema, GeneratedAt: time.Now().UTC().Format(time.RFC3339), Model: m.Model, MaxTokens: m.MaxTokens, Seed: seed}
	for _, p := range m.Prompts {
		if p.Bucket == "" || p.Path == "" {
			return Artifact{}, errors.New("each prompt needs bucket and path")
		}
		promptPath := p.Path
		if !filepath.IsAbs(promptPath) {
			promptPath = filepath.Join(base, promptPath)
		}
		prompt, readErr := os.ReadFile(promptPath)
		for _, mode := range modes() {
			for _, decode := range []struct{ name, env string }{{"q8", "0"}, {"int8-q4k", "1"}} {
				name := safeName(p.Bucket + "-" + mode.name + "-" + decode.name)
				outPath := filepath.Join(outDir, name+".txt")
				stderrPath := filepath.Join(outDir, name+".stderr.txt")
				args := []string{"run", m.Model, "--quiet", "--max-tokens", fmt.Sprint(m.MaxTokens)}
				args = append(args, mode.args...)
				args = append(args, string(prompt))
				r := Run{Bucket: p.Bucket, Mode: mode.name, Decode: decode.name, PromptPath: promptPath, PromptBytes: len(prompt), OutputPath: outPath, StderrPath: stderrPath, Command: append([]string{fakPath}, args...), Environment: []string{"FAK_INKERNEL_ENABLE_THINKING=0", "FAK_INKERNEL_SEED=" + fmt.Sprint(seed), "FAK_KQ_INT8=" + decode.env}}
				if readErr != nil {
					r.ExecutionError = readErr.Error()
					a.Runs = append(a.Runs, r)
					continue
				}
				cmd := exec.Command(fakPath, args...)
				cmd.Env = append(os.Environ(), r.Environment...)
				start := time.Now()
				var stdout, stderr strings.Builder
				cmd.Stdout = &stdout
				cmd.Stderr = &stderr
				cmdErr := cmd.Run()
				r.ElapsedMS = time.Since(start).Milliseconds()
				if writeErr := os.WriteFile(outPath, []byte(stdout.String()), 0o644); writeErr != nil && cmdErr == nil {
					cmdErr = writeErr
				}
				if writeErr := os.WriteFile(stderrPath, []byte(stderr.String()), 0o644); writeErr != nil && cmdErr == nil {
					cmdErr = writeErr
				}
				if cmdErr != nil {
					r.ExecutionError = cmdErr.Error()
				}
				r.Score = scoreText(stdout.String(), p.RequiredLabels)
				a.Runs = append(a.Runs, r)
			}
		}
	}
	a.Comparisons, a.Pass, a.Reason = compareRuns(a.Runs, m.Prompts)
	return a, nil
}

func safeName(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

func main() {
	manifest := flag.String("manifest", "", "JSON manifest containing model and prompt buckets")
	fakPath := flag.String("fak", "fak", "fak binary to execute")
	outDir := flag.String("out", "", "artifact directory")
	seed := flag.Int64("seed", 0, "deterministic in-kernel sampling seed")
	selfcheck := flag.Bool("selfcheck", false, "run scorer and fail-closed fixtures without a model")
	flag.Parse()
	if *selfcheck {
		if err := runSelfcheck(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *manifest == "" || *outDir == "" {
		flag.Usage()
		os.Exit(2)
	}
	a, err := execute(*manifest, *fakPath, *outDir, *seed)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	data, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	artifactPath := filepath.Join(*outDir, "coherence-gate.json")
	if err := os.WriteFile(artifactPath, append(data, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Printf("artifact=%s pass=%t comparisons=%d\n", artifactPath, a.Pass, len(a.Comparisons))
	if !a.Pass {
		os.Exit(1)
	}
}

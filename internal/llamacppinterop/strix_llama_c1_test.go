//go:build linux

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
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

func TestCaptureStrixLlamaC1ObservationBindsExactCompletionRequest(t *testing.T) {
	for marker, arg := range os.Args {
		if arg != "strix-c1-http-helper" || len(os.Args) != marker+7 {
			continue
		}
		for _, rawFD := range os.Args[marker+1 : marker+4] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(130)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(131)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper lifetime owns the maps
		}
		addressPath, acceptedPath, requestPath := os.Args[marker+4], os.Args[marker+5], os.Args[marker+6]
		listener, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
		if err != nil {
			os.Exit(132)
		}
		defer listener.Close() //nolint:errcheck // helper exits with authority
		if err := os.WriteFile(addressPath, []byte(listener.Addr().String()), 0o600); err != nil {
			os.Exit(133)
		}
		connection, err := listener.AcceptTCP()
		if err != nil {
			os.Exit(134)
		}
		defer connection.Close() //nolint:errcheck // helper exits with authority
		if err := os.WriteFile(acceptedPath, []byte("accepted"), 0o600); err != nil {
			os.Exit(135)
		}
		request, err := readStrixLlamaC1TestRequest(connection)
		if err != nil {
			for {
				time.Sleep(time.Second)
			}
		}
		if os.WriteFile(requestPath, request, 0o600) != nil {
			os.Exit(136)
		}
		response := testStrixLlamaC1Response(128)
		if _, err := fmt.Fprintf(connection, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: keep-alive\r\n\r\n", len(response)); err != nil {
			os.Exit(137)
		}
		if _, err := connection.Write(response); err != nil {
			os.Exit(138)
		}
		for {
			time.Sleep(time.Second)
		}
	}
	if os.Getenv("FAK_STRIX_LLAMA_C1_PIDNS") != "1" {
		command := exec.Command("unshare", "--user", "--map-current-user", "--pid", "--fork", "--mount-proc", os.Args[0], "-test.run=^TestCaptureStrixLlamaC1ObservationBindsExactCompletionRequest$")
		command.Env = append(os.Environ(), "FAK_STRIX_LLAMA_C1_PIDNS=1")
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("run c1 witness in isolated pid namespace: %v\n%s", err, output)
		}
		return
	}

	dir := t.TempDir()
	write := func(name string, data []byte, mode os.FileMode) string {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, data, mode); err != nil {
			t.Fatal(err)
		}
		return path
	}
	digest := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:])
	}

	server, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	server, err = filepath.EvalSymlinks(server)
	if err != nil {
		t.Fatal(err)
	}
	source := write("source.tar", []byte("c1 source"), 0o600)
	build := write("build.json", []byte(`{"build":"b10588","vulkan":true}`), 0o600)
	model := write("model.gguf", append([]byte("model"), make([]byte, 4091)...), 0o600)
	loader := write("loader.icd", append([]byte{1}, make([]byte, 4095)...), 0o600)
	dependency := write("dependency.so", append([]byte{2}, make([]byte, 4095)...), 0o600)
	serverInfo, err := os.Stat(server)
	if err != nil {
		t.Fatal(err)
	}
	dependencies := []StrixComparatorPinnedFile{{Path: dependency, SHA256: digest(dependency)}}
	maps, err := os.Open("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]struct{})
	scanner := bufio.NewScanner(maps)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 6 || !strings.HasPrefix(fields[5], "/") {
			continue
		}
		path := strings.TrimSuffix(strings.Join(fields[5:], " "), " (deleted)")
		path, err = filepath.EvalSymlinks(path)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if os.SameFile(info, serverInfo) {
			continue
		}
		if _, ok := seen[path]; ok {
			continue
		}
		seen[path] = struct{}{}
		dependencies = append(dependencies, StrixComparatorPinnedFile{Path: path, SHA256: digest(path)})
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if err := maps.Close(); err != nil {
		t.Fatal(err)
	}

	addressPath, acceptedPath, requestPath := filepath.Join(dir, "address"), filepath.Join(dir, "accepted"), filepath.Join(dir, "request")
	manifest := validStrixComparatorManifest()
	manifest.SourceArchiveSHA256, manifest.BuildManifestSHA256, manifest.ServerBinarySHA256 = digest(source), digest(build), digest(server)
	options := StrixComparatorAuthorityOptions{
		Manifest: manifest, SourceArchivePath: source, BuildManifestPath: build, ServerBinaryPath: server, ModelPath: model,
		LoaderICD: []StrixComparatorPinnedFile{{Path: loader, SHA256: digest(loader)}}, Dependencies: dependencies,
		Arguments:       []string{"-test.run=^TestCaptureStrixLlamaC1ObservationBindsExactCompletionRequest$", "--", "strix-c1-http-helper", "4", "5", "6", addressPath, acceptedPath, requestPath},
		testModelSHA256: digest(model),
	}
	authority, err := OpenStrixComparatorAuthority(options)
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close() //nolint:errcheck // test owns the real helper child
	address := waitStrixLlamaC1TestFile(t, addressPath)
	loaderSet, err := strixComparatorCompleteSetDigest("loader/icd", options.LoaderICD)
	if err != nil {
		t.Fatal(err)
	}
	dependencySet, err := strixComparatorCompleteSetDigest("dependency", options.Dependencies)
	if err != nil {
		t.Fatal(err)
	}
	cell := testStrixExchangeCell(t, manifest, loaderSet, dependencySet)
	packet, err := qwen38quantrun.ImportPromptPacket(cell.Workload.PromptPacketBytes)
	if err != nil {
		t.Fatal(err)
	}
	requestBody, err := buildStrixLlamaC1RequestBody(packet)
	if err != nil {
		t.Fatal(err)
	}
	var request strixLlamaC1CompletionRequest
	if err := json.Unmarshal(requestBody, &request); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(request.Prompt, packet.PromptTokenIDs) || len(request.Prompt) != 26 || request.NPredict != 128 || request.Temperature != 0 || request.TopK != 1 || request.TopP != 1 || request.Seed != 424242 || request.Stream || !request.IgnoreEOS || request.CachePrompt || !request.ReturnTokens || request.NProbs != 1 || request.PostSamplingProbs {
		t.Fatalf("completion request is not the exact pinned c1 envelope: %+v", request)
	}
	requestWire := buildStrixLlamaC1HTTPRequest(address, requestBody)
	if !bytes.HasSuffix(requestWire, requestBody) || !bytes.HasPrefix(requestWire, []byte("POST /completion HTTP/1.1\r\n")) {
		t.Fatal("wire request did not bind the exact completion path and body")
	}
	observation, err := CaptureStrixLlamaC1Observation(context.Background(), StrixLlamaC1CaptureOptions{
		Authority: authority, Cell: cell, ApprovedCellDigest: cell.Digest, Packet: packet, Endpoint: "http://" + address,
	})
	if err == nil || observation != nil || !strings.Contains(err.Error(), "resource window") {
		t.Fatalf("device-free capture bypassed owned RADV resource authority: observation=%v err=%v", observation, err)
	}
	if _, statErr := os.Stat(requestPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("failed resource authority still sent a comparator request: %v", statErr)
	}
	if !authority.Valid() {
		t.Fatal("failed observation capture damaged the comparator authority")
	}

	validResponse := testStrixLlamaC1Response(128)
	parsed, tokenTexts, tokenLogprobs, timings, err := parseStrixLlamaC1Response(validResponse, 26)
	if err != nil {
		t.Fatal(err)
	}
	if len(parsed.Tokens) != 128 || parsed.Tokens[0] != 1000 || parsed.Tokens[127] != 1127 || len(tokenTexts) != 128 || tokenTexts[0] != "t1000" || len(tokenLogprobs) != 128 || tokenLogprobs[0] != -0.25 {
		t.Fatal("strict response parser did not preserve all 128 selected token rows")
	}
	if timings.PromptTokens != 26 || timings.PredictedTokens != 128 || timings.DecodeSteps != 127 || timings.PredictedMS != 635 || timings.PredictedPerTokenMS != 5 || timings.CacheTokens != 0 {
		t.Fatalf("timings = %+v", timings)
	}
	zero := new(StrixLlamaC1Observation)
	if zero.Valid() || zero.PID() != 0 || zero.Endpoint() != "" || zero.RequestSHA256() != "" || zero.ResponseSHA256() != "" || zero.AcceptedTokenIDs() != nil || zero.AcceptedTokenTexts() != nil || zero.AcceptedTokenLogprobs() != nil || zero.Timings() != (StrixLlamaC1Timings{}) {
		t.Fatal("zero observation disclosed data")
	}
	if resources, ok := zero.ResourceObservation(); ok || resources != (StrixComparatorResourceObservation{}) {
		t.Fatal("zero observation disclosed resource data")
	}
	if zero.PerformanceCredit() || zero.CreditReason() != "pair-and-five-measured-trials-required" {
		t.Fatal("single comparator observation received performance credit")
	}
	if _, err := json.Marshal(zero); err == nil {
		t.Fatal("opaque observation was JSON serializable")
	}
	if !errors.Is(zero.Close(), ErrInvalidStrixLlamaC1Observation) {
		t.Fatal("zero observation closed")
	}

	mutations := map[string][]byte{
		"case-folded recognized field": bytes.Replace(validResponse, []byte(`"content"`), []byte(`"Content"`), 1),
		"case-folded dotted duplicate": bytes.Replace(validResponse, []byte(`"speculative.types":"none"`), []byte(`"speculative.Types":"none","speculative.types":"none"`), 1),
		"exact dotted duplicate":       bytes.Replace(validResponse, []byte(`"speculative.types":"none"`), []byte(`"speculative.types":"none","speculative.types":"none"`), 1),
		"unknown top field":            bytes.Replace(validResponse, []byte(`{"index":0`), []byte(`{"unknown":0,"index":0`), 1),
		"unknown probability field":    bytes.Replace(validResponse, []byte(`"id":1000`), []byte(`"unknown":0,"id":1000`), 1),
		"negative accepted ID":         bytes.Replace(validResponse, []byte(`"tokens":[1000`), []byte(`"tokens":[-1`), 1),
		"chosen ID mismatch":           bytes.Replace(validResponse, []byte(`"id":1000`), []byte(`"id":999`), 1),
		"missing chosen token":         bytes.Replace(validResponse, []byte(`"token":"t1000",`), nil, 1),
		"chosen bytes mismatch":        bytes.Replace(validResponse, []byte(`"bytes":[97]`), []byte(`"bytes":[98]`), 1),
		"content bytes mismatch":       bytes.Replace(validResponse, []byte(`"bytes":[97]`), []byte(`"bytes":[98]`), 2),
		"positive logprob":             bytes.Replace(validResponse, []byte(`"logprob":-0.25`), []byte(`"logprob":0.25`), 1),
		"speculation enabled":          bytes.Replace(validResponse, []byte(`"speculative.types":"none"`), []byte(`"speculative.types":"draft"`), 1),
		"cached response mismatch":     bytes.Replace(validResponse, []byte(`"tokens_cached":153`), []byte(`"tokens_cached":0`), 1),
		"contradictory timing":         bytes.Replace(validResponse, []byte(`"predicted_per_token_ms":5`), []byte(`"predicted_per_token_ms":7`), 1),
	}
	for name, mutation := range mutations {
		t.Run(name, func(t *testing.T) {
			if bytes.Equal(mutation, validResponse) {
				t.Fatal("test mutation did not alter response")
			}
			if _, _, _, _, err := parseStrixLlamaC1Response(mutation, 26); err == nil {
				t.Fatal("ambiguous or contradictory response was accepted")
			}
		})
	}
	var missingTop strixLlamaC1Response
	if err := json.Unmarshal(validResponse, &missingTop); err != nil {
		t.Fatal(err)
	}
	missingTop.CompletionProbabilities[0].TopLogprobs = nil
	missingTopRaw, err := json.Marshal(missingTop)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := parseStrixLlamaC1Response(missingTopRaw, 26); err == nil {
		t.Fatal("selected probability without exactly one matching top row was accepted")
	}
}

func readStrixLlamaC1TestRequest(connection *net.TCPConn) ([]byte, error) {
	reader := bufio.NewReader(connection)
	var wire bytes.Buffer
	contentLength := -1
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil, err
		}
		wire.WriteString(line)
		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			contentLength, err = strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(strings.ToLower(line), "content-length:")))
			if err != nil {
				return nil, err
			}
		}
		if line == "\r\n" {
			break
		}
	}
	if contentLength < 0 {
		return nil, errors.New("missing content length")
	}
	if _, err := io.CopyN(&wire, reader, int64(contentLength)); err != nil {
		return nil, err
	}
	return wire.Bytes(), nil
}

func testStrixLlamaC1Response(tokenCount int) []byte {
	tokens := make([]int, tokenCount)
	probabilities := make([]strixLlamaC1TokenProbability, tokenCount)
	texts := make([]string, tokenCount)
	for i := range tokenCount {
		tokens[i] = 1000 + i
		texts[i] = fmt.Sprintf("t%d", tokens[i])
		tokenID, tokenText, logprob := tokens[i], texts[i], -0.25
		bytes := []int{int('a' + byte(i%26))}
		probabilities[i] = strixLlamaC1TokenProbability{
			ID: &tokenID, Token: &tokenText, Bytes: bytes, Logprob: &logprob,
			TopLogprobs: []strixLlamaC1Probability{{ID: &tokenID, Token: &tokenText, Bytes: bytes, Logprob: &logprob}},
		}
		texts[i] = string([]byte{byte(bytes[0])})
	}
	response := struct {
		Index                   int                            `json:"index"`
		Content                 string                         `json:"content"`
		Tokens                  []int                          `json:"tokens"`
		IDSlot                  int                            `json:"id_slot"`
		Stop                    bool                           `json:"stop"`
		TokensPredicted         int                            `json:"tokens_predicted"`
		TokensEvaluated         int                            `json:"tokens_evaluated"`
		GenerationSettings      map[string]any                 `json:"generation_settings"`
		Truncated               bool                           `json:"truncated"`
		StopType                string                         `json:"stop_type"`
		StoppingWord            string                         `json:"stopping_word"`
		TokensCached            int                            `json:"tokens_cached"`
		CompletionProbabilities []strixLlamaC1TokenProbability `json:"completion_probabilities"`
		Timings                 map[string]any                 `json:"timings"`
	}{
		Index: 0, Content: strings.Join(texts, ""), Tokens: tokens, IDSlot: 0, Stop: true,
		TokensPredicted: 128, TokensEvaluated: 26, GenerationSettings: map[string]any{"speculative.types": "none"},
		Truncated: false, StopType: "limit", StoppingWord: "", TokensCached: 153, CompletionProbabilities: probabilities,
		Timings: map[string]any{
			"prompt_n": 26, "prompt_ms": 26.0, "prompt_per_token_ms": 1.0, "prompt_per_second": 1000.0,
			"predicted_n": 128, "predicted_ms": 635.0, "predicted_per_token_ms": 5.0, "predicted_per_second": 200.0,
			"cache_n": 0,
		},
	}
	raw, err := json.Marshal(response)
	if err != nil {
		panic(err)
	}
	return raw
}

func waitStrixLlamaC1TestFile(t *testing.T, path string) string {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil {
			return string(data)
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", path)
	return ""
}

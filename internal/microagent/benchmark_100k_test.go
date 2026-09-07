package microagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The mock exercises Host, descriptor contracts and actual hibernation I/O. It
// proves shared prompt bytes; it deliberately makes no GPU or prefill claim.
type scaleGateway struct {
	prefix                          string
	calls, active, peak, goroutines atomic.Int64
}

func (g *scaleGateway) Model() string { return "synthetic-descriptor-scale" }

func (g *scaleGateway) Complete(_ context.Context, messages []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	n := g.active.Add(1)
	defer g.active.Add(-1)
	for old := g.peak.Load(); n > old && !g.peak.CompareAndSwap(old, n); old = g.peak.Load() {
	}
	n = int64(runtime.NumGoroutine())
	for old := g.goroutines.Load(); n > old && !g.goroutines.CompareAndSwap(old, n); old = g.goroutines.Load() {
	}
	g.calls.Add(1)
	if len(messages) < 2 || messages[0].Role != agent.RoleSystem || messages[0].Content != g.prefix || messages[1].Role != agent.RoleUser {
		return nil, fmt.Errorf("immutable prefix changed")
	}
	b, _ := json.Marshal(tools)
	if string(b) != `[{"type":"function","function":{"name":"read_record","description":"","parameters":{"type":"object"}}}]` {
		return nil, fmt.Errorf("tool prefix changed: %s", b)
	}
	id := messages[1].Content
	output := "state:" + id
	if len(messages) == 3 {
		if messages[2].Role != agent.RoleAssistant || messages[2].Content != output {
			return nil, fmt.Errorf("restored continuation changed for %s", id)
		}
		output = "DONE:" + id + ":" + messages[2].Content
	} else if len(messages) != 2 {
		return nil, fmt.Errorf("unexpected continuation length %d", len(messages))
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: output}}, nil
}

func TestDescriptorStreamSourceFailureDrainsAcceptedWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sourceFailure := errors.New("source failed")
	gateway := &scaleGateway{prefix: "shared"}
	calls, emitted := 0, 0
	err := RunDescriptorStream(ctx, gateway, Config{Workers: 1, Queue: 2}, map[string][]agent.Message{"base": {{Role: agent.RoleSystem, Content: "shared"}}},
		func(context.Context) (Descriptor, error) {
			calls++
			if calls == 2 {
				cancel()
				return Descriptor{}, sourceFailure
			}
			return Descriptor{Schema: DescriptorSchema, ID: "accepted", BaseID: "base", TaskDelta: "accepted", Tools: []string{"read_record"},
				Budget: DescriptorBudget{MaxTurns: 1, MaxOutputTokens: 8}, OutputContract: OutputContract{Kind: "exact", Expected: "state:accepted"}}, nil
		}, func(result Result) error {
			emitted++
			if result.ID != "accepted" || (!result.Done && !errors.Is(result.Err, context.Canceled)) {
				return fmt.Errorf("unexpected terminal result: %+v", result)
			}
			return nil
		})
	if !errors.Is(err, sourceFailure) || emitted != 1 || calls != 2 {
		t.Fatalf("source error must drain accepted work: err=%v emitted=%d calls=%d", err, emitted, calls)
	}
}

func TestDescriptorStream100K(t *testing.T) {
	if testing.Short() {
		t.Skip("100k disk lifecycle witness")
	}
	if runtime.GOOS != "linux" {
		t.Skip("absolute VmHWM witness requires Linux; run through test.ps1/WSL")
	}
	if os.Getenv("FAK_DESCRIPTOR_SCALE_CHILD") != "1" {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestDescriptorStream100K$", "-test.v", "-test.timeout=4m")
		cmd.Env = append(os.Environ(), "FAK_DESCRIPTOR_SCALE_CHILD=1")
		output, err := cmd.CombinedOutput()
		t.Log(string(output))
		if err != nil {
			t.Fatalf("fresh process scaling witness: %v", err)
		}
		return
	}
	const count = 100000
	band, err := NewWarmBand(WarmBandConfig{Dir: t.TempDir(), High: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer band.Close()
	gateway := &scaleGateway{prefix: strings.Repeat("shared immutable prefix ", 256)}
	bases := map[string][]agent.Message{"shared": {{Role: agent.RoleSystem, Content: gateway.prefix}}}
	enrolled, completed := 0, 0
	seen := make([]uint64, (count+63)/64)
	err = RunDescriptorStream(context.Background(), gateway, Config{Workers: 64, Queue: 64, Warm: band, MaxTurns: 2}, bases,
		func(context.Context) (Descriptor, error) {
			if enrolled == count {
				return Descriptor{}, io.EOF
			}
			id := strconv.Itoa(enrolled)
			enrolled++
			return Descriptor{Schema: DescriptorSchemaV2, ID: id, BaseID: "shared", TaskDelta: id, ContinuationToken: id,
				Tools: []string{"read_record"}, Budget: DescriptorBudget{MaxTurns: 2, MaxOutputTokens: 32},
				OutputContract: OutputContract{Kind: "exact", Expected: "DONE:" + id + ":state:" + id}}, nil
		}, func(result Result) error {
			if !result.Done || result.Err != nil || result.Steps != 2 {
				return fmt.Errorf("unverified result: %+v", result)
			}
			id, err := strconv.Atoi(result.ID)
			if err != nil || id < 0 || id >= count {
				return fmt.Errorf("unknown result %q", result.ID)
			}
			mask := uint64(1) << uint(id%64)
			if seen[id/64]&mask != 0 {
				return fmt.Errorf("duplicate result %d", id)
			}
			seen[id/64] |= mask
			completed++
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	stats := band.Stats()
	if completed != count || gateway.calls.Load() != 2*count || stats.Thaws != 2*count || stats.Peak > 64 || gateway.peak.Load() > 64 || len(band.ColdStates()) != 0 || stats.Resident != 0 || stats.Parked != 0 || stats.Warm != 0 {
		t.Fatalf("lifecycle: completed=%d calls=%d stats=%+v cold=%d", completed, gateway.calls.Load(), stats, len(band.ColdStates()))
	}
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Fatal(err)
	}
	var rss uint64
	for _, line := range strings.Split(string(status), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			fields := strings.Fields(line)
			rss, err = strconv.ParseUint(fields[1], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			rss *= 1024
		}
	}
	if rss == 0 || rss > 100000000 {
		t.Fatalf("absolute process peak RSS %d exceeds 100000000-byte envelope", rss)
	}
	t.Logf("engine=synthetic-descriptor-scale contexts=%d completed=%d contract_verified=%d cold_thaws=%d state_mismatches=0 worker_limit=64 peak_active_calls=%d peak_observed_goroutines=%d resident_peak=%d absolute_peak_rss_bytes=%d", count, completed, completed, stats.Thaws, gateway.peak.Load(), gateway.goroutines.Load(), stats.Peak, rss)
}

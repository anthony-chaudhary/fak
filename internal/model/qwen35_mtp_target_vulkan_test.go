//go:build vulkan && (windows || linux) && cgo

package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestQwen35VulkanMTPRawHiddenPrefillAndStepMatchesCPU(t *testing.T) {
	backend, _ := requiredVulkanSequenceBackend(t)
	raw, ok := backend.(compute.Qwen35SequenceRawHiddenBackend)
	if !ok || raw.Qwen35SequenceRawHiddenPath() != compute.Qwen35SequenceRawHiddenPath {
		t.Fatalf("physical Vulkan backend %T lacks exact raw-hidden sequence capture", backend)
	}

	m := NewSynthetic(qwen35HybridTestCfg())
	device, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	device.captureTargetHidden = true
	oracle := m.NewSession()
	oracle.captureTargetHidden = true
	defer oracle.Close()

	prefix := []int{3, 7, 11, 5}
	device.Prefill(prefix)
	oracle.Prefill(prefix)
	for pos := range prefix {
		got, err := device.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("device prompt raw hidden %d: %v", pos, err)
		}
		want, err := oracle.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("CPU prompt raw hidden %d: %v", pos, err)
		}
		compareVulkanSequenceVector(t, "prompt raw hidden "+itoa(pos), got, want)
		if reflect.DeepEqual(got, m.finalNorm(want)) {
			t.Fatalf("prompt row %d captured post-final-norm output instead of raw residual", pos)
		}
	}

	lineage := append([]int(nil), prefix...)
	for step, next := range []int{17, 19, 23} {
		gotLogits := device.Step(next)
		wantLogits := oracle.Step(next)
		compareVulkanSequenceVector(t, "scalar-decode logits "+itoa(step), gotLogits, wantLogits)
		pos := len(prefix) + step
		got, err := device.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("device scalar-decode raw hidden %d: %v", pos, err)
		}
		want, err := oracle.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("CPU scalar-decode raw hidden %d: %v", pos, err)
		}
		compareVulkanSequenceVector(t, "scalar-decode raw hidden "+itoa(step), got, want)
		lineage = append(lineage, next)
		if _, err := device.VerifyTokenLineage(lineage); err != nil {
			t.Fatalf("raw-hidden token lineage after step %d: %v", step, err)
		}
	}
	if _, err := device.TargetHiddenAt(len(lineage)); err == nil {
		t.Fatal("unevaluated raw-hidden row became visible")
	}
}

func TestQwen35MTPVulkanTargetUsesResidentDraftAndDeviceVerification(t *testing.T) {
	backend, _ := requiredVulkanSequenceBackend(t)
	if raw, ok := backend.(compute.Qwen35SequenceRawHiddenBackend); !ok || raw.Qwen35SequenceRawHiddenPath() != compute.Qwen35SequenceRawHiddenPath {
		t.Fatalf("physical Vulkan backend %T lacks exact raw-hidden capture", backend)
	}

	m := NewSyntheticQwen38MTP()
	m.Cfg.TieWordEmbeddings = false
	appendQwen38MTPF32Tensor(m, "lm_head.weight", []int{m.Cfg.VocabSize, m.Cfg.HiddenSize}, make([]float32, m.Cfg.VocabSize*m.Cfg.HiddenSize))
	target, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	draft, err := NewQwen35MTPDraftSession(target, 2)
	if err != nil {
		t.Fatalf("production MTP draft constructor: %v", err)
	}
	prompt := []int{2, 1}
	target.Prefill(prompt)
	if draft.backend != target.Backend || draft.forward == nil || draft.forward.resident == nil {
		t.Fatalf("production draft did not retain target Vulkan backend: draft=%p target=%p resident=%t", draft.backend, target.Backend, draft.forward != nil && draft.forward.resident != nil)
	}
	forwardReceipt := draft.forward.Receipt()
	if forwardReceipt.Path != compute.Qwen35MTPDraftPath || forwardReceipt.Backend != backend.Name() {
		t.Fatalf("resident draft receipt=%+v", forwardReceipt)
	}
	proposal := draft.Propose(prompt)
	if err := draft.Err(); err != nil || len(proposal) == 0 {
		t.Fatalf("resident Vulkan proposal=%v error=%v", proposal, err)
	}
	draft.Close()
	target.Close()

	productionTarget, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	defer productionTarget.Close()
	ordinary := m.NewSession()
	wantOutput := ordinary.Generate(prompt, 4)
	ordinary.Close()
	run, receipt, err := SpecDecodeGreedyQwen35MTPDepthNWithReceipt(productionTarget, prompt, 4, 2)
	if err != nil {
		t.Fatalf("production Vulkan MTP depth-N: %v; receipt=%+v", err, receipt)
	}
	if receipt.Engine != "fak-native" || receipt.Error != "" || run.Rounds == 0 || len(receipt.Transactions) != run.Rounds {
		t.Fatalf("production run/receipt=%+v/%+v", run, receipt)
	}
	if !reflect.DeepEqual(run.Output, wantOutput) {
		t.Fatalf("production Vulkan MTP output=%v, ordinary greedy=%v", run.Output, wantOutput)
	}
	deviceOps, acceptedTokens := 0, 0
	for i, tx := range receipt.Transactions {
		if tx.DowngradeReason != "" || tx.TargetDecodeSteps != 0 || tx.FullTargetReplaySteps != 0 {
			t.Fatalf("transaction %d used fallback/replay: %+v", i, tx)
		}
		if tx.Path != targetVerificationBoundaryRejectPath && (!tx.OneOperation || tx.TargetVerificationOperations != 1) {
			t.Fatalf("transaction %d did not use one resident target operation: %+v", i, tx)
		}
		deviceOps += tx.TargetVerificationOperations
		acceptedTokens += tx.AcceptedTokens
	}
	if deviceOps == 0 || acceptedTokens == 0 {
		t.Fatalf("production run never exercised accepted resident target work: operations=%d accepted=%d receipt=%+v", deviceOps, acceptedTokens, receipt)
	}
	committed := append(append([]int(nil), prompt...), run.Output...)
	if _, err := productionTarget.VerifyTokenLineage(committed); err != nil {
		t.Fatalf("production target lineage: %v", err)
	}
	for pos := range committed {
		if _, err := productionTarget.TargetHiddenAt(pos); err != nil {
			t.Fatalf("production target raw hidden %d: %v", pos, err)
		}
	}
}

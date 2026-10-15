package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41BF16PublicationAuthority(t *testing.T) {
	b := newV41BF16PublishRecorder()
	s := &Session{M: &Model{Cfg: Config{HiddenSize: 4, NumLayers: 1}}, Backend: b}
	canonical := s.v41BindBF16Publication()
	ops := &v41DeviceGatherOps{owner: s, bf16Publish: canonical}
	if canonical == nil || canonical.core == nil || ops.bf16PublishFunc() == nil {
		t.Fatal("canonical binding absent")
	}
	ordinary := ops.bf16PublishFunc()
	got, err := ordinary(0, "indexer.wq_b.weight projection", []float32{1.1, 2.2, 3.3, 4.4})
	if err != nil || !v41BF16Widened(got) || b.calls != 1 {
		t.Fatalf("ordinary adapter changed: %v %v calls=%d", got, err, b.calls)
	}
	marker := errors.New("ordinary replacement")
	ops.setBF16Publication(func(int, string, []float32) ([]float32, error) { return nil, marker })
	if ops.bf16Publish == canonical || ops.bf16Publish.core != nil || ops.bf16Publish.owner != nil || ops.bf16Publish.model != nil || ops.bf16Publish.backend != nil {
		t.Fatal("replacement retained native authority")
	}
	if canonical.core == nil || canonical.owner != s {
		t.Fatal("replacement mutated prior binding")
	}
	if _, err := ops.bf16PublishFunc()(0, "anything", nil); err != marker {
		t.Fatal("replacement behavior lost")
	}
	if _, err := ordinary(0, "indexer.wq_b.weight projection", []float32{1.1}); err != nil || b.calls != 2 {
		t.Fatal("retained ordinary adapter changed")
	}
	ops.setBF16Publication(nil)
	if ops.bf16PublishFunc() != nil || ops.bf16Publish.core != nil {
		t.Fatal("nil replacement retained capability")
	}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	refusal := ops.bf16PublishFunc()
	if refusal == nil {
		t.Fatal("nil ordinary replacement bypassed device-only refusal")
	}
	if _, err := refusal(0, "indexer.wq_b.weight projection", nil); err == nil || b.calls != 2 {
		t.Fatalf("missing device-only refusal or unexpected publication: %v", err)
	}
	var absent *v41DeviceGatherOps
	if absent.bf16PublishFunc() != nil {
		t.Fatal("nil accessor changed")
	}
}

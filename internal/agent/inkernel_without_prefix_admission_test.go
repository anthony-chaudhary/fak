package agent

import (
	"context"
	"testing"
)

type withoutPrefixAdmissionParentKey struct{}

func TestWithoutPrefixAdmissionMarksRequestAndKeepsParentValues(t *testing.T) {
	if inKernelSkipPrefixAdmission(context.Background()) {
		t.Fatal("plain context must admit prefixes")
	}
	parent := context.WithValue(context.Background(), withoutPrefixAdmissionParentKey{}, "tenant")
	ctx := WithoutPrefixAdmission(parent)
	if !inKernelSkipPrefixAdmission(ctx) {
		t.Fatal("WithoutPrefixAdmission must skip prefix admission")
	}
	if got := ctx.Value(withoutPrefixAdmissionParentKey{}); got != "tenant" {
		t.Fatalf("parent value lost: %v", got)
	}
}

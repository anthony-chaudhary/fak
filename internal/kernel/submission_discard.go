package kernel

import "github.com/anthony-chaudhary/fak/internal/abi"

// DiscardSubmission removes an unreaped submission without dispatch. It returns
// false if another consumer has already reaped or discarded the handle.
func (k *Kernel) DiscardSubmission(h abi.SubmissionHandle) bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	_, exists := k.pending[h.Seq]
	delete(k.pending, h.Seq)
	return exists
}

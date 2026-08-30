package modelengine

import (
	"container/heap"
	"time"
)

// waitingDeadlineHeap is the scheduler-owned candidate for replacing one deadline
// timer per waiting lane. The scheduler supplies the clock so expiry order and wake
// behavior can be witnessed without wall-clock sleeps.
type waitingDeadlineHeap struct {
	now     func() time.Time
	entries waitingDeadlineEntries
	byLane  map[*schedLane]*waitingDeadlineEntry
	nextSeq uint64
}

type waitingDeadlineEntry struct {
	lane     *schedLane
	deadline time.Time
	seq      uint64
	index    int
}

type waitingDeadlineEntries []*waitingDeadlineEntry

func newWaitingDeadlineHeap(now func() time.Time) *waitingDeadlineHeap {
	if now == nil {
		now = time.Now
	}
	return &waitingDeadlineHeap{
		now:    now,
		byLane: make(map[*schedLane]*waitingDeadlineEntry),
	}
}

func (h *waitingDeadlineHeap) schedule(lane *schedLane, deadline time.Time) {
	if entry := h.byLane[lane]; entry != nil {
		entry.deadline = deadline
		heap.Fix(&h.entries, entry.index)
		return
	}
	entry := &waitingDeadlineEntry{
		lane:     lane,
		deadline: deadline,
		seq:      h.nextSeq,
		index:    -1,
	}
	h.nextSeq++
	h.byLane[lane] = entry
	heap.Push(&h.entries, entry)
}

func (h *waitingDeadlineHeap) cancel(lane *schedLane) bool {
	entry := h.byLane[lane]
	if entry == nil {
		return false
	}
	heap.Remove(&h.entries, entry.index)
	delete(h.byLane, lane)
	return true
}

func (h *waitingDeadlineHeap) expireReady(dst []*schedLane) []*schedLane {
	now := h.now()
	for len(h.entries) > 0 && !h.entries[0].deadline.After(now) {
		entry := heap.Pop(&h.entries).(*waitingDeadlineEntry)
		delete(h.byLane, entry.lane)
		dst = append(dst, entry.lane)
	}
	return dst
}

func (h *waitingDeadlineHeap) nextDeadline() (time.Time, bool) {
	if len(h.entries) == 0 {
		return time.Time{}, false
	}
	return h.entries[0].deadline, true
}

func (h *waitingDeadlineHeap) len() int { return len(h.entries) }

func (h waitingDeadlineEntries) Len() int { return len(h) }

func (h waitingDeadlineEntries) Less(i, j int) bool {
	if h[i].deadline.Equal(h[j].deadline) {
		return h[i].seq < h[j].seq
	}
	return h[i].deadline.Before(h[j].deadline)
}

func (h waitingDeadlineEntries) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *waitingDeadlineEntries) Push(value any) {
	entry := value.(*waitingDeadlineEntry)
	entry.index = len(*h)
	*h = append(*h, entry)
}

func (h *waitingDeadlineEntries) Pop() any {
	old := *h
	last := len(old) - 1
	entry := old[last]
	old[last] = nil
	entry.index = -1
	*h = old[:last]
	return entry
}

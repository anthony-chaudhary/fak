// Package deadlineadmit is a pure, tier-1 admission policy. It orders pending
// work earliest-deadline-first (EDF) and sheds degradation-eligible items it
// predicts will miss their deadline, so the survivors keep their SLO under
// contention instead of every request missing together.
//
// It is a clean-room policy inspired by Mooncake's admission queue (Apache-2.0);
// no source bytes are vendored. The package imports only the Go standard
// library so it stays tier-1 pure-root eligible.
//
// The clock is an abstract integer chosen by the caller: Deadline, now, and
// PredictedCost must all be expressed in the same unit. An item is dispatched no
// earlier than now, so it is predicted to finish at now+PredictedCost.
//
// Invariant: deadline admission decisions are fail-closed and deterministic.
// Guard: non-degradable items are never shed, ensuring critical tasks are retained even under contention.
// Guard: on-time items are never shed (miss margin must be strictly positive and >= dropThreshold).
package deadlineadmit

import "sort"

// Item is one unit of pending work considered for deadline-ordered admission.
// All fields are plain policy inputs.
type Item struct {
	// ID is a caller-assigned stable identifier, echoed back in a Plan's Order
	// and Shed slices. Callers are expected to keep IDs unique.
	ID int
	// Deadline is the absolute time by which the item must finish.
	Deadline int
	// PredictedCost is the predicted time the item takes to complete once it is
	// dispatched.
	PredictedCost int
	// Degradable reports whether the item may be shed (degradation-eligible).
	// A non-degradable item is never shed, even when it is predicted to miss.
	Degradable bool
}

// Plan is the outcome of Admit.
type Plan struct {
	// Order lists the surviving item IDs in earliest-deadline-first dispatch
	// order. Shed items are excluded from Order.
	Order []int
	// Shed lists the item IDs dropped as predicted deadline misses, also in
	// earliest-deadline-first order. Every ID here is degradable; a
	// non-degradable item is never placed in Shed.
	Shed []int
}

// predictedFinish returns the time the item is predicted to complete if it were
// dispatched at now.
func predictedFinish(it Item, now int) int {
	return now + it.PredictedCost
}

// missMargin returns how far past its deadline the item is predicted to finish.
// A positive result means the item is predicted to miss; zero or negative means
// it is predicted to finish on time.
func missMargin(it Item, now int) int {
	return predictedFinish(it, now) - it.Deadline
}

// shouldShed reports whether an item is dropped under the policy. It must be
// degradable and predicted to overshoot its deadline by at least dropThreshold,
// and by a strictly positive margin so that an on-time item is never shed even
// when dropThreshold is zero.
func shouldShed(it Item, now, dropThreshold int) bool {
	if !it.Degradable {
		return false
	}
	m := missMargin(it, now)
	return m > 0 && m >= dropThreshold
}

// byDeadlineThenID sorts items into a deterministic total order: Deadline
// ascending, ties broken by ascending ID. The result is independent of the
// input order, which keeps dispatch reproducible.
func byDeadlineThenID(items []Item) {
	sort.Slice(items, func(i, j int) bool {
		if items[i].Deadline != items[j].Deadline {
			return items[i].Deadline < items[j].Deadline
		}
		return items[i].ID < items[j].ID
	})
}

// ids projects a slice of items onto their IDs, preserving order.
func ids(items []Item) []int {
	out := make([]int, len(items))
	for i := range items {
		out[i] = items[i].ID
	}
	return out
}

// Order returns the earliest-deadline-first dispatch order of the given items
// as a slice of IDs. Ties on Deadline are broken by ascending ID, so the order
// is deterministic and independent of input order. Order does not apply
// shedding; every input item appears in the result. The input slice is not
// modified.
func Order(items []Item) []int {
	ordered := make([]Item, len(items))
	copy(ordered, items)
	byDeadlineThenID(ordered)
	return ids(ordered)
}

// Shed returns the IDs of items dropped as predicted deadline misses, in
// earliest-deadline-first order. Only degradable items appear: a non-degradable
// item that is predicted to miss is retained (never silently dropped). The
// input slice is not modified.
func Shed(items []Item, now, dropThreshold int) []int {
	dropped := make([]Item, 0, len(items))
	for _, it := range items {
		if shouldShed(it, now, dropThreshold) {
			dropped = append(dropped, it)
		}
	}
	byDeadlineThenID(dropped)
	return ids(dropped)
}

// Admit computes the full admission plan: the surviving items in EDF dispatch
// order and the shed set. An item is shed only when it is degradable and its
// predicted finish overshoots its deadline by at least dropThreshold; every
// other item — including a non-degradable predicted miss — survives and is
// dispatched in earliest-deadline-first order. The input slice is not modified.
func Admit(items []Item, now, dropThreshold int) Plan {
	shed := make(map[int]struct{}, len(items))
	survivors := make([]Item, 0, len(items))
	dropped := make([]Item, 0, len(items))
	for _, it := range items {
		if shouldShed(it, now, dropThreshold) {
			shed[it.ID] = struct{}{}
			dropped = append(dropped, it)
		}
	}
	for _, it := range items {
		if _, gone := shed[it.ID]; !gone {
			survivors = append(survivors, it)
		}
	}
	byDeadlineThenID(survivors)
	byDeadlineThenID(dropped)
	return Plan{
		Order: ids(survivors),
		Shed:  ids(dropped),
	}
}

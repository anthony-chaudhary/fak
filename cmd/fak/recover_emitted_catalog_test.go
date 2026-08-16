package main

import "testing"

func TestEmittedRecoveryPlansBelongToTreeCatalog(t *testing.T) {
	tree := treeRecoveryPlans("main")
	for _, reason := range emittedRecoveryReasons {
		if _, ok := tree[reason]; !ok {
			t.Errorf("emitted recovery reason %s is outside the tree catalog", reason)
		}
	}
}

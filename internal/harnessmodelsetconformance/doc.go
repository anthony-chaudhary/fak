// Package harnessmodelsetconformance owns the captured end-to-end witness for
// generated-harness model-set resolution and startup compatibility.
//
// Tier: mechanism (3) - see internal/architest. The production package is an
// evidence boundary; its external tests compose the shipped model-set leaves.
//
// Invariant: model-set conformance verification operates fail-closed, rejecting partial or unverified model selections deterministically.
//
// Contract: conformance evaluation requires explicit role bindings and validated candidate digests before asserting startup compatibility.
//
// Precondition: caller test fixtures must define canonical role requirements and uncorrupted candidate inventory records.
package harnessmodelsetconformance

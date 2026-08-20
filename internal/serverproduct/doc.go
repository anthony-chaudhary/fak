// Package serverproduct defines the versioned, secret-free boundary between an
// independently owned inference server and its consumers.
//
// Schema v1 compatibility is deliberately narrow:
//   - a consumer accepts only the exact SchemaV1 value;
//   - the receipt's spec digest, server name, artifact, adapter, protocol, auth
//     reference, and requested fixed port must agree with the authored spec;
//   - every required capability must appear in the probe-backed observed set;
//   - authored provenance belongs to the spec, while artifact, adapter,
//     endpoint, readiness, and ownership receipt fields are observed;
//   - generation is positive and consumers should reject an older generation
//     for the same instance after observing a newer one; and
//   - receipts are immutable consumer inputs. Only NewReadyReceipt or
//     DecodeReadyReceipt can produce the value accepted by WriteReadyReceipt.
//
// Schema v1 does not promise compatibility across schema versions, adapter
// version constraints, or protocol revisions. Later adapters and harness
// importers must fail closed unless these rules pass.
package serverproduct

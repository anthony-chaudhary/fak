# Inkling Metal operators

The C++ target, draft, Metal command buffer, and shader operators come from incoai/splash at commit `2afc7337aacc044722fa95e4cb3da75c281a19a8` under Apache-2.0. The adjacent LICENSE and THIRD_PARTY_NOTICES retain the upstream notices. FAK loads model tensors, drives each token and candidate block, selects tokens, and commits accepted request state through its own in-process adapter in internal/inklingmetal.

Regenerate the shader translation-unit index with `go run ./internal/inklingmetal/cmd/genshaders` after modifying a shader. The Metal device compiles embedded MSL source from these translation units.

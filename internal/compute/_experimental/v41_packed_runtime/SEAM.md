# Batch lifetime integration contract

The adjacent header models bookkeeping only. Its `COMPLETE` state is simulated,
not a hardware completion receipt. `retireChecked` accepts assertions for design
review; no runtime wrapper may expose those booleans as caller-supplied proof.
The proposed C API has declarations only, no definitions or callers.

Required integration, none implemented by this package:

1. Mint a native step-owner token and monotonic nonzero serial for every recording
   window, including automatic reopen after Read. Reject inflight/restore/foreign
   work before any unchecked drain. Reserve records before command begin, then
   activate without allocation; never identify a window by a reusable command.
2. Enroll exact source/destination allocation identities and retain Go owners
   before recording. Reserve bookkeeping before commands. Reject overlapping
   unresolved uses, same-allocation copies, foreign owners and stale generations.
3. Inside `v41EndSubmitWaitChecked`, mint an identity-bound receipt after the real
   successful end/reset/submit/wait and before command recycling. Empty windows
   are CANCELLED. Missing receipts, global status, Read success or queue-idle
   guesses never establish completion. Selected or uncertain failures quarantine.
4. Retain receipts and unresolved resources across context replacement. Guard
   every native free, destruction, pool-reuse and device-retirement route, not
   merely Go Free/Close. The sidecar does not own commands, descriptors or staging;
   actual retained backend owners must be associated with the same identity.
5. Keep pending writes private and poisoned after failure. Publish initialized
   rows and numerical eligibility only after exact completion, then acknowledge
   each use once. Acknowledgement never frees a buffer or restores context health.
   Only the owning coordinator may forget receipts after all observers consume
   them. A completed A receipt remains a fence fact if reopened B fails, but does
   not authorize access or release on the failed context.

No reset/reconstruction may discard retained records or evade quarantine.
Generation rebinding requires a newer generation and zero retained records.
The existing synchronous storage guards and absent kernel resolver stay intact.

Required UNRUN witnesses: each end/reset/submit/wait failure; A completion with B
reopen and B failure; reused command with stale serial; foreign/generation mismatch;
empty window; serial exhaustion; allocation/enrollment failure before recording;
duplicate acknowledgement; source/destination Free/Close while pinned; no metadata
publication on failure; exact ownership retention on uncertainty. This contract
is not runtime admission or a claim that those tests passed.

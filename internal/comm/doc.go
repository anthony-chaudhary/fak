// Package comm is the first-class agent communicator: a deterministic, adjudicated
// group descriptor (rank/size/split + spawn membership) over the dos-arbitrate lane
// partition. It is the single agent-level identity layer the collective / topology /
// spawn / cohort leaves of the MPI-shaped epic (#639) all bind against.
//
// What it provides. A [Group] is an ordered set of member agents. Rank is a member's
// position in the SORTED member set, so the same members always receive the same
// ranks regardless of the order they were handed in — rank is a deterministic
// function of the member identities, never of arrival order or a member's output.
// [Group.Split] partitions a group by color; each color binds to a dos.toml lane, so
// a split IS a dos-arbitrate lease — two splits that name overlapping lanes serialize
// by refusal at the arbiter, not by a lock in this package. [Membership] is the
// rank-stamped value minted when a wave spawns its members.
//
// The adjudication floor. A group op that admits a tool call routes it through
// abi.Kernel.Submit (the adjudication chokepoint) — there is no collective that is
// exempt from refusal. [Group.Gather] folds member outputs through modelroute.Combine
// in RANK order, satisfying Combine's "votes in member order" contract on STRUCTURE:
// the layout is deterministic even though a member agent's text is not.
//
// Honesty caveat (borrow the term, disclaim the scope). This is NOT MPI. A Group
// moves ZERO bytes and runs NO collective; it inherits no interconnect, message-rate,
// or collective-latency property. It is explicitly NOT model.DistComm
// (internal/model/dist_collective.go): that is a REAL cross-process tensor collective
// whose ranks are tensor-parallel host-float32 shards of ONE model (already disclaimed
// in-file as not-NCCL / not-multi-GPU). A comm.Group's ranks index DETACHED OS
// processes that communicate only through git and leases, never through a comm.
// Determinism here is pinned to the group LAYOUT/partition only — rank assignment,
// split-color→lane binding, and the rank order handed to Combine — never to a member
// agent's non-bit-exact output. (This copies the modelroute.ReduceAllReduce
// borrow-the-distributed-systems-term / disclaim-the-scope template.)
//
// Tier: mechanism (2) — see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package comm

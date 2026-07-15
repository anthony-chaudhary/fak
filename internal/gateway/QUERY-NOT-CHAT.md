# Query-not-chat implementation invariant

The gateway's managed-session contract has two state classes:

- `ManagedQuerySession.PinnedOriginatingTask` is the stable, content-addressed identity of the one query the session serves.
- `ManagedQuerySession.WorkingState` is one replaceable slot for the current result, error, or next affordance.

A swap may replace `WorkingState` only while presenting the same originating-task pin. Appending another volatile state is detected as transcript accumulation. Detection is observe-only while `FAK_QUERY_NOT_CHAT_ENFORCE` is unset; setting it to `true` turns the same seam into a hard refusal. Context restore remains the read-only mechanism for paging durable detail beneath this stable pin.

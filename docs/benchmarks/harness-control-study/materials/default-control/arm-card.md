# Default-control arm

Use the pinned `fak` binary and the files in this directory. Do not inspect the other arm or repository examples.

Your starting product is `product.lock.json`. Its effective response style is `concise`; all other required behavior is already present. Complete the outcomes in the byte-identical `task-card.md` by using `fak harness --help` and its inspect, derive or override, preview, and runtime-verification commands.

You may read command `--help`. Ask the facilitator for a hint only when you would otherwise stop; every hint is recorded as one help request. Keep every command and output in your arm directory. Stop the clock only at the boundary stated in the task card.

## Evidence boundary

Before the task clock starts, run this from the extracted assigned-arm directory using the IDs and order supplied privately by the facilitator:

```sh
./fak harness study control packet verify --dir .
./fak harness study control packet receipt start --dir . --participant-id PERSON_ID --pair-id PAIR_ID --pair-order ORDER
```

The `STARTED` line is the clock boundary. During the task, append each command exactly as entered to `commands.txt`, one command per line. Append each error to `errors.txt`, one error per line; leave that file absent when there were no errors. Keep the final task artifact inside this arm directory.

At the task-card stop boundary, stop working and tell the facilitator. The facilitator independently verifies the result, records help requests and the human-reported outcome fields, then runs `receipt finalize` with the artifact and transcript paths. On arm 2 only, finalization also records the preference asked after both artifacts exist. Do not hand-edit `receipt.json`; inspect it after finalization and return that file with the arm artifacts.

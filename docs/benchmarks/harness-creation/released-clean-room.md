# Released harness clean-room witness

Issue: #6957; release acceptance: #6935.

After a release is published, download one target archive and its `.sha256` sidecar from
the GitHub release page. Run the witness on the matching host outside the fak checkout:

```text
fak harness release witness \
  --archive fak_<version>_linux_amd64.tar.gz \
  --checksum fak_<version>_linux_amd64.tar.gz.sha256 \
  --target linux_amd64 \
  --dir /tmp/fak-released-harness-linux \
  --module example.test/released-harness-linux \
  --receipt linux-amd64.json \
  --rollback-command "install the previously verified v0.43.0 archive"
```

Use the corresponding `.zip`, `windows_amd64`, and Windows paths for the Windows run.
The verb fails before extraction if the sidecar does not match, rejects archive traversal,
and refuses an existing product directory. It then uses the released binary—not the
checkout—to initialize an external product, replaces only the stable user-owned task card,
builds and selfchecks it, reads provenance and the exact upgrade command from
`harness.lock.json`, regenerates with the pinned version, and proves the user file hash is
unchanged.

The JSON receipt records archive and binary hashes, target, durations, exact commands and
exits, lock provenance, upgrade and operator-supplied rollback commands, preservation, and
outcome. Archive the Windows and Linux receipts here only after checking they contain no
machine-specific sensitive paths. These release-maintainer runs satisfy #6935's artifact
boundary but remain calibration; they do not enter #6911's unfamiliar-builder denominator.

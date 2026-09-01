# GLM-5.3-Flash identity fixtures

Pinned metadata only; no weights are committed or downloaded by tests.

- Source: `zai-org/GLM-5.3-Flash` revision `04c4e9e95c5da8862dced7e5056455116f83a7e0`
- Upstream license: MIT (`LICENSE` at the pinned revision)
- `config.json`: exact upstream config used to identify the GLM5Next envelope.
- `tensor_inventory.json`: representative names, dtypes, shapes, and shard assignments read from the pinned `model.safetensors.index.json` and the selected shards' safetensors headers using HTTP range reads; no tensor payload was fetched.

Invalidating assumption: if the pinned upstream revision changes its architecture-defining config, tensor names, shapes, shard count, or license, these fixtures must be refreshed before fak treats that new revision as the same recognized envelope.

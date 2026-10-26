# CPU Torch activation fixtures

Generated with PyTorch 2.12.0+cpu, protocol 2, on CPU. Original synthetic tensors;
no model weights or third-party source code. No GPU is needed to read these files.

The generation expression is `torch.save({'value': value, 'meta': {'name': name,
'step': step, 'rank': rank}}, path)` for two ranks and two steps. Step 0 has ids
`[1, 2]`, step 1 has `[3]`; each has two Llama decoder outputs (arange reshaped
to rows of 16 plus the layer index, wrapped in a one-element output list) and
bf16 attention outputs (ones).

`reader.pt` uses `torch.save({'value': [v[:, 1::2], torch.tensor([1., -2., .5],
dtype=torch.bfloat16), torch.tensor([1., -2., .5], dtype=torch.float16),
torch.tensor([1, 16777217, -2], dtype=torch.int64)], 'meta': {'source': 'CPU fixture'}},
path)` with `v = torch.arange(12, dtype=torch.float32).reshape(3, 4)`.

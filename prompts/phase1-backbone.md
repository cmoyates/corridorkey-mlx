Work only on the PyTorch reference harness.

Goal:
Create a deterministic reference pipeline that loads the original CorridorKey checkpoint and dumps intermediate tensors needed for staged MLX parity.

Deliverables:

- scripts/dump_pytorch_reference.py
- reference/fixtures/ sample inputs and outputs
- tests that validate fixture generation shape contracts
- README updates describing the fixture format

Requirements:

- Load the model via state_dict, not entire-model pickle semantics.
- Save:
  - 4 backbone feature maps
  - alpha coarse logits
  - fg coarse logits
  - alpha coarse probs
  - fg coarse probs
  - delta logits
  - final alpha
  - final fg
- Make fixture generation deterministic where practical.
- Keep one tiny golden example checked in.
- Print a concise shape report.

Do not:

- start MLX implementation
- refactor unrelated files
- add training code

Before editing:

- inspect the original CorridorKey model code carefully
- summarize exact tensors that will be dumped
- define file naming and serialization format first

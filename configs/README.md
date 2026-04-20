# configs/

Two config schemas, consumed by different CLI commands.

## `configs/*.yaml` — Evaluation configs

Used by `vla-eval run`. Define server URL, Docker image, and benchmark parameters.

Schema: `{server, docker, output_dir, benchmarks[]}`

## `configs/model_servers/` — Model server configs

Used by `vla-eval serve`. Define which script to run and with what arguments.

Schema: `{script, args, [extends]}`

### Directory convention

Each model gets its own subdirectory: `configs/model_servers/<name>/`. A single-config model has one YAML; multi-benchmark models use `_base.yaml` + per-benchmark overrides.

### `extends` — config inheritance

Configs can inherit from a base file using `extends`:

```yaml
# _base.yaml — shared settings
script: "src/vla_eval/model_servers/mymodel.py"
args:
  port: 8000

# libero.yaml — benchmark-specific override
extends: _base.yaml
args:
  checkpoint: org/model-libero
  chunk_size: 16
```

The `args` dict is deep-merged: the child's `args` override the base's on a per-key basis. The `script` key is inherited if not specified in the child.

### Observation params

Model servers declare their observation requirements (e.g. wrist camera,
proprioceptive state) via the HELLO handshake, so the benchmark is
auto-configured without manual `--param` flags. See each model server
config's header comments for details.

# Add A New Benchmark

SoftSkill follows the SkillOpt adapter pattern for benchmark integrations.

## Steps

1. Copy the scaffold:

```bash
cp -r skillopt/envs/_template skillopt/envs/your_benchmark
```

2. Rename and implement the adapter files:

```bash
cd skillopt/envs/your_benchmark
mv env_template.py adapter.py
mv loader_template.py loader.py
```

3. Update class names and imports in the new files.

4. Add a config at `configs/your_benchmark/default.yaml`, starting from
   `skillopt/envs/_template/config_template.yaml`.

5. Register the adapter in `_register_builtins()` in `scripts/train.py` and
   `scripts/eval_only.py`.

6. Add focused tests for the loader, adapter setup, and any benchmark-specific
   scoring or rollout behavior.

Keep raw benchmark corpora out of git. Release only lightweight manifests and
setup notes unless the upstream dataset license explicitly allows redistribution.

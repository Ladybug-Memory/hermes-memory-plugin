# Ladybug Memory Installed

Install the Python dependency:

```bash
pip install ladybug-memory
```

Link the plugin into the Hermes memory provider directory:

```bash
ln -s ~/.hermes/plugins/ladybug \
      ~/.hermes/hermes-agent/plugins/memory/ladybug
```

Then run setup to activate it:

```bash
hermes memory setup
```

Choose `ladybug` in the setup wizard.

For GLiNER2 entity extraction (enables the `ladybug_entity` tool):

```bash
pip install ladybug-memory[extract]
```

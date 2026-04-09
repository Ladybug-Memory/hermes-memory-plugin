# Ladybug Memory Plugin for Hermes

[Ladybug Memory](https://github.com/Ladybug-Memory/ladybug-memory/) is an agent memory interface with
multiple implementations. The simpler, devx focused one uses a single storage engine: LadybugDB.

To use, copy the directory structure below to `hermes-agent` and apply the following change to pyproject.toml

```
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -58,6 +58,7 @@ pty = [
   "pywinpty>=2.0.0,<3; sys_platform == 'win32'",
 ]
 honcho = ["honcho-ai>=2.0.1,<3"]
+ladybug = ["ladybug-memory>=0.1.4,<1"]
 mcp = ["mcp>=1.2.0,<2"]
 homeassistant = ["aiohttp>=3.9.0,<4"]
 sms = ["aiohttp>=3.9.0,<4"]
@@ -87,6 +88,7 @@ all = [
   "hermes-agent[slack]",
   "hermes-agent[pty]",
   "hermes-agent[honcho]",
+  "hermes-agent[ladybug]",
   "hermes-agent[mcp]",
   "hermes-agent[homeassistant]",
   "hermes-agent[sms]",
```

# Upstream Dependencies

This directory is reserved for external upstream repositories.

`upstream/pypto/` is intentionally absent from the source tree until it is added
as a git submodule:

```bash
git submodule add https://github.com/hw-native-sys/pypto.git upstream/pypto
```

Do not commit a copied PyPTO source tree here. Use a submodule so Sonata can pin
and update PyPTO explicitly.

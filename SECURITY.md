# Security

Please don't open a public issue for a vulnerability. Report it privately
through [GitHub's advisory form](https://github.com/CanReader/FastNN/security/advisories/new)
and I'll get back to you within a week.

The parts most worth looking at are the ones that read untrusted input:
checkpoint and safetensors loading, and the dataset downloads.

Only the latest release gets fixes.

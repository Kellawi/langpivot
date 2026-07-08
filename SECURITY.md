# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |
| < 1.0   | ❌        |

Security fixes are released against the latest `1.0.x`.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public GitHub
issue for a vulnerability.

- **Email:** BasharKilawe@gmail.com (put `SECURITY: langpivot` in the subject)
- Alternatively, use GitHub's private
  ["Report a vulnerability"](https://github.com/Kellawi/langpivot/security/advisories/new)
  advisory flow.

Please include a description, affected version, and reproduction steps. I aim
to acknowledge reports within **7 days** and to provide a fix or mitigation
timeline after triage. Please give a reasonable disclosure window before any
public discussion.

## Security posture

- The core package has **zero required dependencies** and executes no code at
  build or install time (declarative `pyproject.toml`, `hatchling` backend).
- The bundled model is plain **JSON** — no `pickle`, no `eval`/`exec`, no
  `subprocess`.
- `PivotRouter.decide()` and all analysis run **locally with no network**.
- `PivotClient.chat()` is the only component that contacts an external API. It
  reads keys from environment variables, never logs them, and refuses to send
  a key over a non-`https://` endpoint.
- Releases to PyPI are published only from CI via **Trusted Publishing
  (OIDC)** with PEP 740 attestations — no long-lived API tokens exist.

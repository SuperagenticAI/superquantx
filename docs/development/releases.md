# Release process

SuperQuantX uses the version in `src/superquantx/version.py` and the matching
section of `CHANGELOG.md`. The current release is **0.3.0** (Python 3.11+).
PyPI versions are immutable: always choose a new version for changed artifacts.

## Prepare and validate

Merge the intended pull requests and resolve their interactions before tagging.
Update the version tuple, release date, release name, changelog, README and
related guides together. Keep planned features in the Unreleased section.

From the repository root, with Python 3.12 and uv installed:

```bash
uv sync --locked --all-extras
uv run --no-sync pytest tests/unit tests/integration
uv run --no-sync mkdocs build --strict
uv run --no-sync python examples/showcase.py
uv run --no-sync superquantx list-backends
uv build
```

Run SDK checks on local simulators. Hardware checks require provider credentials
and are separate from local simulation validation. Inspect actual test results:
existing CI steps using `|| true` do not enforce a passing suite. Do not publish
with unexplained test failures.

If a cloud machine has a read-only home directory, set `XDG_CONFIG_HOME`,
`XDG_CACHE_HOME`, and `MPLCONFIGDIR` to writable directories before importing SDKs.
Use `MPLBACKEND=Agg` for headless validation.

## Publish

The [Publish workflow](https://github.com/SuperagenticAI/superquantx/actions/workflows/publish.yml)
builds distributions, uploads them using the repository's `PYPI_API_TOKEN`
secret, and creates a GitHub release with notes from `CHANGELOG.md`.

After the validated commit is on `main`, create and push its version tag:

```bash
git tag -a v0.3.0 -m "SuperQuantX 0.3.0"
git push origin v0.3.0
```

The workflow also supports manual dispatch with an existing tag. If publication
fails, inspect the failed step and PyPI before retrying; an upload may have
succeeded before a later GitHub release step failed. Do not move an already
published release tag or overwrite a published package version.

The documentation workflow deploys `docs/` on pushes to `main`. Check that
workflow separately from package publication.

## Verify publication

Confirm the Publish workflow succeeds and PyPI lists the wheel and source
distribution for the new version. In a fresh virtual environment, install
`superquantx==0.3.0`, confirm `superquantx --version`, and run a Bell-state
simulation. Install `superquantx[mcp]==0.3.0` to verify the MCP entry point and
protocol. Finally, check GitHub release notes, assets and deployed docs.

Never put PyPI tokens in repository files, release notes or shell logs. Use the
configured GitHub Actions secret or a securely provided publishing credential.

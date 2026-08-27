# Contributing

Contributions are welcome in the library, examples, and documentation.

## Development setup

```console
git clone https://github.com/cxzzzz/wavekit.git
cd wavekit
poetry install
```

Install the documentation dependencies when working on the docs site:

```console
poetry install --with docs
```

## Quality checks

Run the full checks from the repository root:

```console
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest
```

The example tests compile their HDL fixtures with Icarus Verilog and execute the
analysis scripts. Install `iverilog` before running them.

## Build the documentation

The docs toolchain currently requires Python 3.10 or newer because Zensical does.
Build the site in strict mode so missing pages and broken internal links fail the
check:

```console
poetry run zensical build --clean --strict
```

## Documentation conventions

- Give every page an explicit H1.
- Use relative links between documentation pages.
- Keep complete runnable examples in `example/`; link to them rather than copying
  their source into Markdown.
- Use NumPy-style docstrings for public APIs.
- Update the hand-maintained [examples index](examples.md) when adding a complete
  example and its validation test.

## Pull requests

Keep each pull request focused and describe any user-facing, API, or example
changes. Run the relevant checks before submitting.

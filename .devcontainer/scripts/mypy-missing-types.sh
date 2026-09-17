#!/usr/bin/bash
mypy . --cache-dir ./.mypy_cache/ --cache-fine-grained
batcat .mypy_cache/missing_stubs
uv add -r .mypy_cache/missing_stubs
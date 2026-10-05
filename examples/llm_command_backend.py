"""Template for `--llm "cmd:python examples/llm_command_backend.py"`.

trainmedic writes one JSON request to stdin:
    {"system": str, "prompt": str, "schema": {JSON schema}, "max_tokens": int}
and expects one JSON object matching the schema on stdout. Adapt `call_model`
to whatever local or self-hosted model your team runs. This template shells out
to the `ollama` CLI as an example.
"""

import json
import subprocess
import sys

MODEL = "qwen3-coder"  # any local model


def call_model(system: str, prompt: str, schema: dict) -> str:
    full = (
        f"{system}\n\n{prompt}\n\nRespond with a single JSON object that validates against this JSON schema:\n"
        f"{json.dumps(schema)}"
    )
    out = subprocess.run(["ollama", "run", MODEL, "--format", "json"], input=full, capture_output=True, text=True, check=True)
    return out.stdout


def main() -> None:
    req = json.load(sys.stdin)
    text = call_model(req["system"], req["prompt"], req["schema"])
    json.dump(json.loads(text), sys.stdout)


if __name__ == "__main__":
    main()

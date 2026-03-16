from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any

from llm_lab.config_utils import load_json_object, resolve_value
from llm_lab.registry import get_family, list_families

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
except ImportError:
    uvicorn = None
    FastAPI = None
    HTTPException = None
    StreamingResponse = None


def _require_server_deps() -> None:
    if FastAPI is None or uvicorn is None or StreamingResponse is None:
        raise ImportError("Serve mode requires fastapi + uvicorn.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="llm_lab serve (OpenAI-like endpoints)")
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--family-config", type=str, default=None, help="family JSON config path")
    parser.add_argument("--list-families", action="store_true")

    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--assistant-model-dir", type=str, default=None)

    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default=None)

    parser.add_argument("--model-id", type=str, default="llm_lab-local")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def _engine_args(args: argparse.Namespace) -> dict[str, Any]:
    family_cfg = load_json_object(args.family_config) if args.family_config else {}
    inference_cfg = family_cfg.get("inference", {})
    if not isinstance(inference_cfg, dict):
        raise ValueError("`inference` in family config must be an object.")

    return {
        "weights_dir": resolve_value(args.weights_dir, inference_cfg.get("weights_dir"), None),
        "backend": resolve_value(args.backend, inference_cfg.get("backend"), "native"),
        "device": resolve_value(args.device, inference_cfg.get("device"), "cpu"),
        "assistant_model_dir": resolve_value(args.assistant_model_dir, inference_cfg.get("assistant_model_dir"), None),
        "source": resolve_value(args.source, inference_cfg.get("source"), "sft"),
        "model_tag": resolve_value(args.model_tag, inference_cfg.get("model_tag"), None),
        "step": resolve_value(args.step, inference_cfg.get("step"), None),
        "device_type": resolve_value(args.device_type, inference_cfg.get("device_type"), ""),
    }


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(args: argparse.Namespace) -> FastAPI:
    _require_server_deps()
    family_cfg = load_json_object(args.family_config) if args.family_config else {}
    family_name = resolve_value(args.family, family_cfg.get("family"), "qwen35")
    family = get_family(family_name)
    engine = family.create_inference_engine(_engine_args(args))
    app = FastAPI(title=f"llm_lab ({family.name})")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "family": family.name,
            "tokenizer": family.tokenizer_name,
            "chat_template": family.chat_template_name,
        }

    @app.get("/v1/models")
    def list_models():
        return {"data": [{"id": args.model_id, "object": "model", "owned_by": "local"}], "object": "list"}

    @app.post("/v1/completions")
    def completions(body: dict[str, Any]):
        prompt = body.get("prompt")
        if prompt is None:
            raise HTTPException(status_code=400, detail="`prompt` is required.")
        generation_kwargs = {
            "max_new_tokens": body.get("max_new_tokens", body.get("max_tokens", 128)),
            "temperature": body.get("temperature", 0.7),
            "top_k": body.get("top_k", 50),
            "top_p": body.get("top_p"),
            "seed": body.get("seed", 42),
        }
        stream = bool(body.get("stream", False))
        if stream:
            req_id = f"cmpl-{uuid.uuid4().hex}"

            def event_stream():
                try:
                    for delta in engine.stream_text(prompt=str(prompt), generation_kwargs=generation_kwargs):
                        payload = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": args.model_id,
                            "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                        }
                        yield _sse_data(payload)
                    end_payload = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": args.model_id,
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                    }
                    yield _sse_data(end_payload)
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    yield _sse_data({"id": req_id, "object": "error", "error": str(exc)})
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        output = engine.generate_text(prompt=str(prompt), generation_kwargs=generation_kwargs, return_full_text=False)
        text = output["texts"][0]
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": args.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": output.get("prompt_tokens", 0),
                "completion_tokens": output.get("completion_tokens", 0),
                "total_tokens": output.get("prompt_tokens", 0) + output.get("completion_tokens", 0),
            },
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict[str, Any]):
        messages = body.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            raise HTTPException(status_code=400, detail="`messages` must be a non-empty list.")
        prompt = family.render_chat_prompt(messages)
        generation_kwargs = {
            "max_new_tokens": body.get("max_new_tokens", body.get("max_tokens", 128)),
            "temperature": body.get("temperature", 0.7),
            "top_k": body.get("top_k", 50),
            "top_p": body.get("top_p"),
            "seed": body.get("seed", 42),
        }
        output = engine.generate_text(prompt=prompt, generation_kwargs=generation_kwargs, return_full_text=False)
        text = output["texts"][0]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": output.get("prompt_tokens", 0),
                "completion_tokens": output.get("completion_tokens", 0),
                "total_tokens": output.get("prompt_tokens", 0) + output.get("completion_tokens", 0),
            },
        }

    return app


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.list_families:
        for family in list_families():
            print(
                f"{family.name}: {family.description} | "
                f"builder={family.model_builder_name} | "
                f"training_supported={family.supports_training}"
            )
        return 0

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from os import path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from semantic_text_splitter import TextSplitter
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from http import HTTPStatus
import requests

from protocol import (
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    UsageInfo,
    StreamOptions,
    ModelCard,
    ModelList,
)

# Create an argument parser
parser = argparse.ArgumentParser(description="Triton Inference Server OpenAI API Proxy")

# Add arguments
parser.add_argument(
    "--triton_http_proto",
    type=str,
    required=False,
    help="Triton Inference Server HTTP host protocol. (default: http)",
    default="http",
)
parser.add_argument(
    "--triton_http_host",
    type=str,
    required=False,
    help="Triton Inference Server HTTP host IP. (default: 127.0.0.1)",
    default="127.0.0.1",
)
parser.add_argument(
    "--triton_http_port",
    type=int,
    required=False,
    help="Triton Inference Server HTTP host port. (default: 8000)",
    default=8000,
)
parser.add_argument(
    "--tokenizer_dir",
    type=str,
    required=True,
    help="Path to a HuggingFace folder containing `tokenizer.json`",
)
parser.add_argument(
    "--default_max_tokens",
    type=int,
    required=False,
    help="Set the maximum tokens in output to use if not explicitly specified in incoming request.(default: 512)",
    default=512,
)
parser.add_argument(
    "--host",
    type=str,
    required=False,
    help="Set the ip address to listen.(default: 0.0.0.0)",
    default="0.0.0.0",
)
parser.add_argument(
    "--port",
    type=int,
    required=False,
    help="Set the port to listen.(default: 11434)",
    default=11434,
)
parser.add_argument(
    "--verbose",
    type=bool,
    required=False,
    help="Enable verbose logging. (default: False)",
    default=False,
)

# Parse the arguments
args = parser.parse_args()

# Use the provided arguments
triton_http_proto = args.triton_http_proto
triton_http_host = args.triton_http_host
triton_http_port = args.triton_http_port
tokenizer_dir = args.tokenizer_dir
default_max_tokens = args.default_max_tokens
host = args.host
port = args.port
verbose = args.verbose

# Ignore any models except `ensemble`
ignored_models_regex = re.compile(
    "^(preprocessing|postprocessing|tensorrt_llm|tensorrt_llm_bls)$"
)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def ts() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def triton_server() -> str:
    return f"{triton_http_proto}://{triton_http_host}:{triton_http_port}"


# Fix model creation date to time this server was started
server_start_ts = ts()


class ChatCompletionResponseStreamJsonEncoder(json.JSONEncoder):
    """Support JSON encoding for SSE "text/event-stream" events on StreamingResponse"""

    def default(self, o):
        if isinstance(o, UsageInfo):
            return (
                None
                if (o.prompt_tokens <= 0) and (o.total_tokens <= 0)
                else {
                    "prompt_tokens": o.prompt_tokens,
                    "total_tokens": o.total_tokens,
                    "completion_tokens": o.completion_tokens,
                }
            )
        if isinstance(o, DeltaMessage):
            return {"role": o.role, "content": o.content}
        if isinstance(o, ChatCompletionResponseStreamChoice):
            return {
                "index": o.index,
                "finish_reason": o.finish_reason,
                "delta": (
                    {}
                    if (o.finish_reason == "stop") or (o.finish_reason == "length")
                    else o.delta
                ),
            }
        if isinstance(o, ChatCompletionStreamResponse):
            return (
                {
                    "id": o.id,
                    "object": o.object,
                    "created": o.created,
                    "model": o.model,
                    "choices": o.choices,
                    "usage": o.usage,
                }
                if o.usage
                else {
                    "id": o.id,
                    "object": o.object,
                    "created": o.created,
                    "model": o.model,
                    "choices": o.choices,
                }
            )
        return json.JSONEncoder.default(self, o)


app = FastAPI()


@app.get("/")
def index() -> Response:
    role = "assistant"
    content = ChatMessage(role=role, content=parser.description)
    return content  # JSONResponse(content=content, status_code=HTTPStatus.OK)


@app.get("/health")
@app.get("/v1/health")
def checkHealth() -> Response:
    response = requests.get(f"{triton_server()}/v2/health/ready")
    return Response(status_code=response.status_code)


@app.get("/models")
@app.post("/models")
@app.get("/v1/models")
@app.post("/v1/models")
def listModels() -> Response:
    response = requests.post(f"{triton_server()}/v2/repository/index")
    if response.encoding is None:
        response.encoding = "utf-8"

    response_json = json.loads(response.text)
    models = filter(
        lambda a_model: not bool(ignored_models_regex.search(a_model["name"])),
        response_json,
    )

    # https://platform.openai.com/docs/api-reference/models/list
    models = [
        model
        for model in map(
            lambda a_model: ModelCard(
                id=a_model["name"],
                object="model",
                created=server_start_ts,
                owned_by="triton-openai-api",
            ),
            models,
        )
    ]
    content = ModelList(data=models)
    return content  # JSONResponse(content=content, status_code=HTTPStatus.OK)


@app.get("/models/{model}")
@app.get("/v1/models/{model}")
def listModel(model: str) -> Response:
    response = requests.post(f"{triton_server()}/v2/repository/index")
    if response.encoding is None:
        response.encoding = "utf-8"

    response_json = json.loads(response.text)
    models = filter(lambda a_model: model == a_model["name"], response_json)

    # https://platform.openai.com/docs/api-reference/models/retrieve
    models = [
        model
        for model in map(
            lambda aModel: ModelCard(
                id=aModel["name"],
                object="model",
                created=server_start_ts,
                owned_by="triton-openai-api",
            ),
            models,
        )
    ]
    content = ModelList(data=models)

    if len(content.data) != 1:
        return Response(status_code=HTTPStatus.NOT_FOUND)

    return (
        content.data.pop()
    )  # JSONResponse(content=content, status_code=HTTPStatus.OK)


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
def chatCompletion(request: ChatCompletionRequest, raw_request: Request):
    # Setup chat response details
    id = f"chatcmpl-{random_uuid()}"
    model = request.model
    role = "assistant"

    max_tokens = int(request.max_tokens or default_max_tokens)
    stopwords = request.stop if request.stop else []
    stream = bool(request.stream)
    stream_options = request.stream_options or StreamOptions(include_usage=False)
    stream_include_usage = bool(stream_options.include_usage)

    # Setup the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{tokenizer_dir}")
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "left"

    messages = [message for message in request.messages]
    messages.insert(
        0,
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },  # Always preclude a `system`` message for better model response
    )

    # https://platform.openai.com/docs/guides/prompt-engineering/six-strategies-for-getting-better-results
    text_input = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md#generate-vs-generate_stream
    # NOTE: `generate_stream` endpoint simply sends _same_ response as non-stream endpoint, one character at a time over SSE.
    # Instead of looking out for `text_output` characters in incoming stream, and sending them over SSE one character at a time,
    # it is better to always just use non-streaming endpoint, then stream tokens (aka words) instead of characters over SSE.
    # action = "generate_stream" if stream else "generate"
    action = "generate"

    data = json.dumps(
        {
            "text_input": text_input,
            "parameters": {
                "max_tokens": max_tokens,
                "bad_words": stopwords,
                "stop_words": stopwords,
                # Model defaults variables
                "return_log_probs": False,
                "return_context_logits": False,
                "return_generation_logits": False,
                # Remaining values copied as-is
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                # Unknown to model parameters
                # "n":request["n"],
                # "user": request["user"],
            },
        }
    )

    response = requests.post(
        url=f"{triton_server()}/v2/models/{model}/{action}",
        data=data,
        stream=False,  # see notes for `action`
    )

    if response.encoding is None:
        response.encoding = "utf-8"

    response_json = json.loads(response.text)

    prompt = messages[-1]["content"]  # User request
    text_output = response_json["text_output"]

    # Require `Tokenizer`` for TextSplitter, as it won't directly support `AutoTokenizer`
    # (needs `to_str` - https://huggingface.co/docs/tokenizers/api/tokenizer#tokenizers.Tokenizer.to_str`)
    # https://github.com/benbrandt/text-splitter?tab=readme-ov-file#with-hugging-face-tokenizer
    # https://pypi.org/project/semantic-text-splitter/
    output_tokenizer = Tokenizer.from_file(path.join(tokenizer_dir, "tokenizer.json"))

    if stream:

        def chunks():
            splitter = TextSplitter.from_huggingface_tokenizer(
                output_tokenizer, 3, trim=False
            )
            chunks = splitter.chunks(text_output)

            """Usage statistics for the completion request.
            completion_tokens - Number of tokens in the generated completion.
            prompt_tokens - Number of tokens in the prompt.
            total_tokens - Total number of tokens used in the request (prompt + completion)."""
            prompt_tokens = len(output_tokenizer.encode(prompt).tokens)
            completion_tokens = len(output_tokenizer.encode(text_output).tokens)
            total_tokens = prompt_tokens + completion_tokens

            usage = (
                UsageInfo(
                    prompt_tokens=-1,
                    completion_tokens=-1,
                    total_tokens=-1,
                )
                if stream_include_usage
                else None
            )  # see ChatCompletionResponseStreamJsonEncoder

            # for data in response.iter_content(decode_unicode=True):
            for index, content in enumerate(chunks):
                # finish_reason=None
                delta = DeltaMessage(role=role, content=content)
                choice = ChatCompletionResponseStreamChoice(
                    index=0, delta=delta, finish_reason=None
                )
                chunk = ChatCompletionStreamResponse(
                    id=id, model=model, choices=[choice], usage=usage, created=ts()
                )
                yield f"data: {json.dumps(chunk, cls=ChatCompletionResponseStreamJsonEncoder)}\n\n"

            # finish_reason="stop"
            delta = DeltaMessage(role=role, content="")
            choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=delta,
                finish_reason="length" if (completion_tokens == max_tokens) else "stop",
            )

            chunk = ChatCompletionStreamResponse(
                id=id, model=model, choices=[choice], usage=usage, created=ts()
            )
            yield f"data: {json.dumps(chunk, cls=ChatCompletionResponseStreamJsonEncoder)}\n\n"

            """If set, an additional chunk will be streamed before the data: [DONE] message. 
            The usage field on this chunk shows the token usage statistics for the entire request, 
            and the choices field will always be an empty array. 
            All other chunks will also include a usage field, but with a null value."""
            if stream_include_usage:
                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                chunk = ChatCompletionStreamResponse(
                    id=id, model=model, choices=[], usage=usage, created=ts()
                )
                yield f"data: {json.dumps(chunk, cls=ChatCompletionResponseStreamJsonEncoder)}\n\n"

            # Send the final done message
            yield "data: [DONE]"

        return StreamingResponse(
            chunks(), media_type="text/event-stream", status_code=HTTPStatus.OK
        )

    # stream = False
    """Usage statistics for the completion request.
    completion_tokens - Number of tokens in the generated completion.
    prompt_tokens - Number of tokens in the prompt.
    total_tokens - Total number of tokens used in the request (prompt + completion)."""
    prompt_tokens = len(output_tokenizer.encode(prompt).tokens)
    completion_tokens = len(output_tokenizer.encode(text_output).tokens)
    total_tokens = prompt_tokens + completion_tokens

    message = ChatMessage(role=role, content=text_output)
    choice = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason="length" if (completion_tokens == max_tokens) else "stop",
    )
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    content = ChatCompletionResponse(
        id=id, model=model, choices=[choice], usage=usage, created=ts()
    )

    return content  # JSONResponse(content=content, status_code=HTTPStatus.OK)


if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=1,
            log_level="debug" if verbose else "info",
        )
    except KeyboardInterrupt:
        pass  # Ctrl+C shutdown, hence ignore

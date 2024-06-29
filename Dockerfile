# The Dockerfile is used to construct an image that can be directly used
# to run the OpenAI compatible Triton Inference Server  server.

# prepare basic build environment
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 AS build
WORKDIR /opt/tritonserver/openai

# To build TensorRT-LLM engines, see https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama for one such example
# Use `python3 -c "import tensorrt_llm"` to find TensorRT-LLM version used by Triton Inference Server
ARG TENSORRT_LLM_VERSION="v0.9.0"
ARG VENV="pyinstaller"

# install build and runtime dependencies
# pyinstaller bundles `app.py` into a single fat executable included with all necessary dependencies
COPY *.py .
COPY requirements.txt requirements.txt

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
  && apt-get install -y python3.10-venv \
  && python3 -m venv ${VENV} \
  && ${VENV}/bin/python3 -m pip install --upgrade --requirement requirements.txt \
  && ${VENV}/bin/python3 -m pip install --upgrade pyinstaller \
  && ${VENV}/bin/pyinstaller --onefile --paths=. --clean app.py \
  && git clone --depth 1 --branch ${TENSORRT_LLM_VERSION} https://github.com/NVIDIA/TensorRT-LLM.git /opt/tritonserver/third-party-src/TensorRT-LLM \
  && git clone --depth 1 --branch ${TENSORRT_LLM_VERSION} https://github.com/triton-inference-server/tensorrtllm_backend.git /opt/tritonserver/third-party-src/tensorrtllm_backend

FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
COPY --from=build --chown=triton-server:triton-server /opt/tritonserver/openai/dist/app /opt/tritonserver/bin/tritonopenaiserver
COPY --from=build --chown=triton-server:triton-server /opt/tritonserver/third-party-src/ /opt/tritonserver/third-party-src/

EXPOSE 11434/tcp
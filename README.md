# Triton Inference Server OpenAI compatible API proxy
[This project](https://github.com/visitsb/triton-inference-server-openai-api) provides an OpenAI API compatible proxy for NVIDIA [Triton Inference Server](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/). More specifically, LLMs on NVIDIA GPUs can benefit from high performance inference with [TensorRT-LLM](https://developer.nvidia.com/tensorrt#inference) backend running on [Triton Inference Server compared to using llama.cpp](https://jan.ai/post/benchmarking-nvidia-tensorrt-llm#key-findings).

Triton Inference Server supports [HTTP/REST and GRPC inference protocols](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md) based on the community developed [KServe protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2), but that is not useable with existing OpenAI API clients.

This proxy bridges that gap and it currently API supports **text** generation [OpenAI API](https://platform.openai.com/docs/api-reference/introduction) endpoints only which are suitable for use with [Open WebUI](https://docs.openwebui.com/) or similar OpenAI clients-
```text
GET|POST /v1/models           (or /models)
GET      /v1/models/{model}   (or /models/{model})
POST     /v1/chat/completions (or /v1/completions) streaming supported
```

## Docker image
**Recommended** Use a pre-published [Docker image](https://hub.docker.com/r/visitsb/tritonserver)
```bash
docker image pull visitsb/tritonserver:24.07-trtllm-python-py3
```

Alternatively, use the `Dockerfile` to build a local image. The proxy is built on top of existing [Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) docker image which includes the TensorRT-LLM backend.

```bash
# Pull upstream NVIDIA docker image
docker image pull nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
# Clone this repository
git clone <this repository>
cd triton-inference-server-openai-api
# Build your custom docker image with proxy bundled
docker buildx build --no-cache --tag myimages/tritonserver:24.07-trtllm-python-py3 .
```

## Usage
Once your image is pulled (or built locally) you can run it directly using Docker-
```bash
# Run Triton Inference Server alongwith proxy as shoen in `sh -c` command
docker run --rm --tty --interactive \
       --gpus all --shm-size 4g --memory 32g \
       --cpuset-cpus 0-3 --publish 11434:11434/tcp \
       --volume <your Triton models folder>:/models:rw \
       --name triton \
       visitsb/tritonserver:24.07-trtllm-python-py3 \
       sh -c '/opt/tritonserver/bin/tritonserver \
              --model-store /models/mymodel/model \
            & /opt/tritonserver/bin/tritonopenaiserver \
              --tokenizer_dir /models/mymodel/tokenizer \
              --engine_dir /models/mymodel/engine'
```

Alternatively using `docker-compose.yml`-
```yaml
triton:
    image: visitsb/tritonserver:24.07-trtllm-python-py3
    command: >
      sh -c '/opt/tritonserver/bin/tritonserver --model-store /models/mymodel/model & /opt/tritonserver/bin/tritonopenaiserver --tokenizer_dir /models/mymodel/tokenizer --engine_dir /models/mymodel/engine'
    ports:
      - "11434:11434/tcp" # OpenAI API Proxy
      - "8000:8000/tcp"   # HTTP
      - "8001:8001/tcp"   # GRPC
      - "8080:8080/tcp"   # Sagemaker, Vertex
      - "8002:8002/tcp"   # Prometheus metrics
    volumes:
      - <your Triton models folder>:/models:rw
    shm_size: "4G"
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 8G
          devices: 
            - driver: nvidia
              count: all
              capabilities: [compute,video,utility]
    ulimits:
      stack: 67108864
      memlock:
        soft: -1
        hard: -1
```

## Performance
Using [GenAI-Perf](https://github.com/triton-inference-server/client/tree/main/src/c%2B%2B/perf_analyzer/genai-perf) to measure performance for [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on a [NVIDIA RTX 4090 GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) the following was observed-

Test: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) evaluated using NVIDIA [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/docs/tutorial.html#openai-chat-completions-api). For llama.cpp evaluation [QuantFactory/Meta-Llama-3-8B-GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF) - `Meta-Llama-3-8B.Q8_0.gguf` was used.

```text
Backend          Loaded model size     GPU Util Tokens/sec
-------          -----------------     -------- ----------
TensorRT (gRPC)  15879MiB /  24564MiB  91%      97.04
TensorRT (HTTP)  15879MiB /  24564MiB  91%      56.73 
llama.cpp         9491MiB /  24564MiB  74%      70.23
```

In summary, TensorRT (gRPC) inference is better than llama.cpp, but using TensorRT (HTTP) gave similar performance to llama.cpp.

The raw performance numbers are as below-
#### TensorRT (gRPC)
```text
[INFO] genai_perf.wrapper:135 - Running Perf Analyzer : 'perf_analyzer -m llama3 --async --service-kind triton -u triton:8001 --measurement-interval 4000 --stability-percentage 999 -i grpc --streaming --shape max_tokens:1 --shape text_input:1 --concurrency-range 1'
                                  LLM Metrics                                   
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃            Statistic ┃    avg ┃    min ┃     max ┃    p99 ┃     p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ Request latency (ns) │ 1,081… │ 1,048… │ 1,311,… │ 1,284… │ 1,083,… │ 1,064… │
│     Num output token │    105 │    100 │     110 │    110 │     109 │    107 │
│      Num input token │    200 │    200 │     200 │    200 │     200 │    200 │
└──────────────────────┴────────┴────────┴─────────┴────────┴─────────┴────────┘
Output token throughput (per sec): 97.04
Request throughput (per sec): 0.92
```

#### TensorRT (HTTP) via this OpenAI API Proxy
```text
[INFO] genai_perf.wrapper:135 - Running Perf Analyzer : 'perf_analyzer -m llama3 --async --endpoint v1/chat/completions --service-kind openai -u triton:11434 --measurement-interval 4000 --stability-percentage 999 -i http --concurrency-range 1'
                                  LLM Metrics                                   
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃            Statistic ┃    avg ┃    min ┃     max ┃    p99 ┃     p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ Request latency (ns) │ 2,033… │ 1,732… │ 3,856,… │ 3,723… │ 2,525,… │ 1,802… │
│     Num output token │    115 │    110 │     121 │    121 │     120 │    119 │
│      Num input token │    200 │    200 │     200 │    200 │     200 │    200 │
└──────────────────────┴────────┴────────┴─────────┴────────┴─────────┴────────┘
Output token throughput (per sec): 56.73
Request throughput (per sec): 0.49
```

#### llama.cpp
```text
[INFO] genai_perf.wrapper:135 - Running Perf Analyzer : 'perf_analyzer -m llama3 --async --endpoint v1/chat/completions --service-kind openai -u llama:11434 --measurement-interval 4000 --stability-percentage 999 -i http --concurrency-range 1'
                                  LLM Metrics                                   
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃            Statistic ┃    avg ┃    min ┃     max ┃    p99 ┃     p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ Request latency (ns) │ 1,656… │ 1,596… │ 1,822,… │ 1,810… │ 1,701,… │ 1,649… │
│     Num output token │    116 │    104 │     149 │    147 │     132 │    118 │
│      Num input token │    200 │    200 │     200 │    200 │     200 │    200 │
└──────────────────────┴────────┴────────┴─────────┴────────┴─────────┴────────┘
Output token throughput (per sec): 70.23
Request throughput (per sec): 0.60
```

**Note** This proxy uses TensorRT (HTTP) currently, so above performance numbers should be considered relative. Performance will vary for TensorRT-LLM models based on [build and deployment options](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#using-the-tensorrt-llm-backend) used.

Additional optimizations like speculative sampling and FP8 quantization can further improve throughput. For more on the throughput levels that are possible with TensorRT-LLM for different combinations of model, hardware, and workload, see the [official benchmarks](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md).

## Build and deploy your own models
The image includes [TensorRT-LLM toolbox](https://github.com/NVIDIA/TensorRT-LLM.git) and [backend](https://github.com/triton-inference-server/tensorrtllm_backend.git) for building your own TensorRT-LLM models. Both can be found under `/opt/tritonserver/third-party-src/` inside your Docker image.

The basic steps to build a TensorRT model are outlined [here](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#using-the-tensorrt-llm-backend) which essentially involves
1. Downloading a [Hugging Face model](https://huggingface.co/models) of your choice, 
2. Converting it to a TensorRT format, and 
3. Lastly building a compiled model that can be deployed on Triton Inference Server. 

Additionally, you can also use the steps mentioned [here](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#retrieve-the-model-weights) to build your TensorRT model. Once your model is built, you can [deploy](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#deploy-with-triton-inference-server) and use it through the OpenAI API proxy.

## Further references-
 - [Benchmarking NVIDIA TensorRT-LLM](https://jan.ai/post/benchmarking-nvidia-tensorrt-llm) - TensorRT-LLM was 30-70% faster than [llama.cpp](https://github.com/ggerganov/llama.cpp) on the same hardware, consumes less memory on consecutive runs with marginally more GPU VRAM utilization than llama.cpp and models are 20%+ smaller compiled model sizes than llama.cpp.
 - [Use Llama 3 with NVIDIA TensorRT-LLM and Triton Inference Server](https://docs.lxp.lu/howto/llama3-triton/) - 30-minute tutorial to show how to use TensorRT-LLM to build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs using Llama3 model as an example. 
 - Similar guide can be on [Serverless TensorRT-LLM (LLaMA 3 8B)](https://modal.com/docs/examples/trtllm_llama) - how to use the TensorRT-LLM framework to serve Meta’s LLaMA 3 8B model at a total throughput of roughly 4,500 output tokens per second on a single NVIDIA A100 40GB GPU.
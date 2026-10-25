# Open-source component inventory

This inventory distinguishes code integrated into BuildVisual from model
weights that are only candidates for deployment. Model availability in a
configuration file is not evidence that inference ran.

| Component | Exact version / revision | License evidence | Runtime status |
|---|---|---|---|
| BuildVisual deterministic model renderer | Repository implementation; NumPy and Pillow versions pinned in `backend/requirements.txt` | In-repository implementation; dependency licenses are published by their upstream projects | Integrated and used in local pipeline tests; this is a model render, not generated AI footage |
| FFmpeg / ffprobe | Host: 8.1.2; Debian 13 Docker image: 7.1.5 | FFmpeg licensing depends on build flags; inspect the target package build before redistribution | Present on this host and verified in the built image |
| faster-whisper | PyPI `1.2.1` | MIT: [upstream license](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE); package metadata: [PyPI](https://pypi.org/project/faster-whisper/1.2.1/) | Optional adapter integrated; package and model weights are not installed or run in this environment |
| Systran/faster-whisper-small | Hugging Face revision `536b0662742c02347bc0e980a01041f333bce120`; `model.bin` SHA-256 `3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671` | MIT in the pinned [model card](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120) | Selected optional local checkpoint; 483,546,902-byte weights are absent and are never downloaded automatically |
| Qwen/Qwen3-VL-2B-Instruct | Hugging Face revision `89644892e4d85e24eaac8bacfd4f463576704203`; `model.safetensors` SHA-256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | Apache-2.0 in the pinned [model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203) | Selected local vision checkpoint; 4,255,140,312-byte weights are absent. A self-hosted OpenAI-compatible API adapter is implemented and contract-tested; model serving/inference is not |
| Local video model | None selected | Not verified | No local video checkpoint is installed or integrated; no hosted video-generation fallback is used as a claimed self-hosted path |

Before redistributing the Docker image, verify the actual FFmpeg build's enabled
codecs and corresponding LGPL/GPL obligations. The optional model checkpoint
must be obtained by the operator from the pinned revision and verified against
the listed SHA-256 digest before enabling local speech inference.

The local vision adapter is configured with `LLM_PROVIDER=local_openai`,
`LLM_MODEL=Qwen/Qwen3-VL-2B-Instruct`, and `LLM_BASE_URL` pointing to the
self-hosted server. No inference server, GPU runtime, or Qwen weights are
included in the application image; this integration has only been tested with
a mocked OpenAI-compatible response.

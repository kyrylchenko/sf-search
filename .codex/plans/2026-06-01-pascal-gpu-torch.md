# Pascal GPU PyTorch Runtime

## Goal

Make the Docker embedding runtime execute on this production server's NVIDIA GeForce GTX 1080 GPU.

## Non-goals

- Do not change model architecture or embedding behavior.
- Do not replace Qdrant, Postgres, NATS, or SigNoz.
- Do not commit secrets or generated runtime artifacts.

## Context

The server GPU is Pascal `sm_61`. The current lock resolves `torch==2.12.0+cu130`, which exposes CUDA but fails real GPU kernels with `no kernel image is available for execution on the device`. A one-off container test with `torch==2.9.1+cu126` succeeds on the same GPU.

## Approach

1. Pin the embedding dependency to `torch==2.9.1+cu126`.
2. Add an explicit uv source for the official PyTorch CUDA 12.6 wheel index.
3. Regenerate `services/main/uv.lock` using uv in the same Python 3.14 Docker image used by the service build.
4. Rebuild the Docker app image.
5. Verify a CUDA tensor operation and then start the GPU pipeline with the observability override.

## Verification

- `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi`
- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm embedding uv run python -c 'import torch; x=torch.ones((1,), device="cuda"); print(x+1)'`
- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.observability.yml --profile pipeline up -d`
- `docker compose logs -f embedding monitoring`

## Risks

The local host still has a pending kernel upgrade from `6.8.0-100` to `6.8.0-124`; NVIDIA modules were built for both kernels, so this should survive a later reboot, but GPU health should be checked after reboot.

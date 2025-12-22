# p3g

## Requirements
- Docker installed

## Build
docker build -t p3g:local .

## Run (mount your models)
docker run --rm -it \
  -p 7860:7860 \
  -e MODEL_PATH=/models/your-model.gguf \
  -e MMPROJ_PATH=/models/your-mmproj.gguf \
  -v /path/on/host/models:/models:ro \
  p3g:local

Download Qwen2-VL models and place them in the models/ folder.

## Structure

```shell
├── benchmark.py # Run benchmark
├── config # Config files
├── src # Source code
├── train.py # Run training
└── validate.py # Run validation
```

## Config

### .env

| Name             | Description                                                       | Default       |
|------------------|-------------------------------------------------------------------|---------------|
| HF_TOKEN         | Hugging Face Token to downlaod dataset                            | None          |
| WANDB_API_KEY    | WandDB Token to upload models metric                              | None          |
| MODEL_DIRECTORY  | Models directory                                                  | ./models      |
| RUNS_DIRECTORY   | Train/Validation/Benchmark output directory                       | ./runs        |
| AI_DEVICE        | Device where to run Yolo                                          | auto          |
| LOG_LEVEL        | Debugging log level                                               | INFO          |
| LOG_WANDB_ENABLE | Enable/disable metric upload                                      | False         |
| DB_FILE          | Benchmark output DB. Recommended different file name per project. | benchmarks.db |

### Train/Validate/Benchmark config

```shell
model:
  yolo_version: str # Yolo version
  model_size: str # Model size

labels:
  load_other: false # Load label "other" If true ["drone", "other"] else ["drone"]

dataset:
  hf_name: # HF dataset name
  hf_revision: # HF revision
  image_transform: "RGB"

#https://docs.ultralytics.com/modes/train/
train:
  # Yolo train parameters

#https://docs.ultralytics.com/modes/val/
validation:
  # Yolo validation parameters

#https://docs.ultralytics.com/modes/benchmark/
benchmark:
  # Yolo benchmark parameters

reproducibility:
  seed: int # Random seed for reproducibility
```

## How to run

### Install uv

UV is a fast Python package and project manager written in Rust.
It serves as a modern alternative to traditional Python package
managers like pip and poetry, offering significantly faster installation
times and better dependency resolution. UV can manage Python versions, create virtual environments, and handle project
dependencies with a single unified tool.

**curl** (recommended)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**pip**

```shell
pip install uv
```

### Train

```shell
uv run train.py --config=./config/RGB_yolo26_nano.yaml
```

Add option `--quiet` or `-q` to hide output.

### Validate

```shell
uv run validate.py --config=./config/RGB_yolo26_nano.yaml --model-name=model_name
```

Model name is the name of the model file in `models` folder.

### Benchmark

Validate store and compare model performance.

```shell
uv run benchmark.py --config=./config/RGB_yolo26_nano.yaml --model-name=model_name
```

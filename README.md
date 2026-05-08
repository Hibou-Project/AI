<img src="https://avatars.githubusercontent.com/u/232561603?s=200&v=4" width="100px" align="left">

### `Hibou AI`
Hibou AI tools, train, validate, benchmark models
</br>

This repository contains various draft and production AI and algorithm implementations.

## Audio

The audio direction contains all the ML models, dataset handling (training and processing) and a lot of drafts (mostly).

You can also find some tests/implementations for direction of arrival implementations and transfer function calculations. (!AI Slop!).


## Vision

### 🗄️ Structure

Main project structure is:

```shell
├── benchmark.py # Run benchmark
├── config # Config files
├── src # Source code
├── train.py # Run training
└── validate.py # Run validation
```

### ⚙️ Config

#### .env

Global project settings are defined in `.env` file. If no file is created it will be created automatically from
.env.example.

| Name                 | Description                                                       | Default       |
|----------------------|-------------------------------------------------------------------|---------------|
| TOKEN_HF             | Hugging Face Token to downlaod dataset                            | None          |
| TOKEN_WANDB          | WandDB Token to upload models metric                              | None          |
| TOKEN_AWS_KEY_ID     | AWS Key ID token for s3                                           | None          |
| TOKEN_AWS_SECRET_KEY | AWS Secret key for s3                                             | None          |
| DATASET_PATH         | Path where datasets are downloaded merged                         | ./datasets    |
| MODEL_DIRECTORY      | Models directory                                                  | ./models      |
| RUNS_DIRECTORY       | Train/Validation/Benchmark output directory                       | ./runs        |
| AI_DEVICE            | Device where to run Yolo                                          | auto          |
| LOG_LEVEL            | Debugging log level                                               | INFO          |
| LOG_WANDB_ENABLE     | Enable/disable metric upload                                      | False         |
| DB_FILE              | Benchmark output DB. Recommended different file name per project. | benchmarks.db |

#### Train/Validate/Benchmark config

```shell
model:
  yolo_version: str # Yolo version
  model_size: str # Model size

labels:
  load_other: false # Load label "other" If true ["drone", "other"] else ["drone"]

dataset:
  image_transform: "RGB"
  split_ratio: [0.8, 0.1, 0.1] # Train, val, test
  providers: # Only AWS, HuggingFace, and local directories are supported
    hf:
    ...
    aws_s3:
    ...
    local:
    ...

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

### 🏃‍➡️ How to run

#### Install uv

UV is a fast Python package and project manager written in Rust.

**curl** (recommended)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**pip**

```shell
pip install uv
```

#### Train

```shell
uv run train.py --config=./config/RGB_yolo26_nano.yaml
```

Add option `--quiet` or `-q` to hide output.

#### Validate

```shell
uv run validate.py --config=./config/RGB_yolo26_nano.yaml --model-name=model_name
```

Model name is the name of the model file in `models` folder.

#### Benchmark

Validate store and compare model performance.

```shell
uv run benchmark.py --config=./config/RGB_yolo26_nano.yaml --model-name=model_name
```


# Tools

We mostly used jupyter notebooks but for some python scripts we used `uv`.


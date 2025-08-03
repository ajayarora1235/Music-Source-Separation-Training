# Music Source Separation CoreML Model Creation

Repository for creating CoreML models for music source separation for on-device inference. This repository enables conversion of trained music source separation models to CoreML format for efficient execution on Apple devices.

**Thanks to [ZFTurbo](https://github.com/ZFTurbo) for the original model and inference implementations.**

## Compatible Models

Currently compatible models for CoreML conversion:

* **MDX23C** based on [KUIELab TFC TDF v3 architecture](https://github.com/kuielab/sdx23/). Key: `mdx23c`.
* **Demucs4HT** [[Paper](https://arxiv.org/abs/2211.08553)]. Key: `htdemucs`.
* **Band Split RoFormer** [[Paper](https://arxiv.org/abs/2309.02612), [Repository](https://github.com/lucidrains/BS-RoFormer)] . Key: `bs_roformer`.
* **Mel-Band RoFormer** [[Paper](https://arxiv.org/abs/2310.01809), [Repository](https://github.com/lucidrains/BS-RoFormer)]. Key: `mel_band_roformer`.
* **SCNet** [[Paper](https://arxiv.org/abs/2401.13276), [Official Repository](https://github.com/starrytong/SCNet), [Unofficial Repository](https://github.com/amanteur/SCNet-PyTorch)] Key: `scnet`.

**Note**: Thanks to [@lucidrains](https://github.com/lucidrains) for recreating the RoFormer models based on papers.

## How to: CoreML Conversion

To convert a trained model to CoreML format:

1) Run the model conversion script:
```bash
python model_coreml_conversion.py \
    --model_type <model_type> \
    --config_path <config_path> \
    --checkpoint <checkpoint_path>
```

2) Since iSTFT is not supported by CoreML, you must export it separately:
```bash
python istft_coreml_conversion.py \
    --model_type <model_type> \
    --config_path <config_path> \
    --checkpoint <checkpoint_path>
```

### Conversion Example

```bash
# Convert the main model
python model_coreml_conversion.py \
    --model_type mel_band_roformer \
    --config_path configs/config_mel_band_roformer_vocals.yaml \
    --checkpoint results/model.ckpt

# Convert the iSTFT component separately
python istft_coreml_conversion.py \
    --model_type mel_band_roformer \
    --config_path configs/config_mel_band_roformer_vocals.yaml \
    --checkpoint results/model.ckpt
```

## How to: Testing

To test your converted CoreML models:

```bash
python test_coreml_conversion.py <path_to_model.mlpackage> <path_to_istft.mlpackage>
```

### Testing Example

```bash
python test_coreml_conversion.py model.mlpackage istft.mlpackage
```

## How to: Inference

For regular inference without CoreML to test modules, you can run:

```bash
python inference_coreml.py \
    --model_type mdx23c \
    --config_path configs/config_mdx23c_musdb18.yaml \
    --start_check_point results/last_mdx23c.ckpt \
    --input_folder input/wavs/ \
    --store_dir separation_results/
```

This uses the same arguments as the original `inference.py` script.

## Useful notes

* CoreML models are optimized for on-device inference on Apple hardware (iOS, macOS).
* The iSTFT component must be exported separately due to CoreML limitations.
* For fastest runtime performance, consider implementing iSTFT directly in Swift or C++. The iSTFT conversion provided here is for convenience.
* Make sure you have the necessary dependencies installed for CoreML conversion.

## Code description

* `configs/config_*.yaml` - configuration files for models
* `models/*` - set of available models for training and inference
* `dataset.py` - dataset which creates new samples for training
* `gui-wx.py` - GUI interface for code
* `inference.py` - process folder with music files and separate them
* `train.py` - main training code
* `train_accelerate.py` - experimental training code to use with `accelerate` module. Speed up for MultiGPU.
* `utils.py` - common functions used by train/valid
* `valid.py` - validation of model with metrics
* `ensemble.py` - useful script to ensemble results of different models to make results better (see [docs](docs/ensemble.md)).   

## Pre-trained models

Look here: [List of Pre-trained models](docs/pretrained_models.md)

If you trained some good models, please, share them. You can post config and model weights [in this issue](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/1).

## Dataset types

Look here: [Dataset types](docs/dataset_types.md)

## Augmentations

Look here: [Augmentations](docs/augmentations.md)

## Graphical user interface

Look here: [GUI documentation](docs/gui.md) or see tutorial on [Youtube](https://youtu.be/M8JKFeN7HfU)

## Citation

* [arxiv paper](https://arxiv.org/abs/2305.07489)

```text
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

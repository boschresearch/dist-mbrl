# Value-Distributional Model-Based Reinforcement Learning
Official PyTorch implementation of the paper ["Value-Distributional Model-Based
Reinforcement Learning"](https://arxiv.org/abs/2308.06590).

## Installation

Prerequisites:
- `conda` (optional, install for Option #1 below)
- `docker` (optional, install for  Option #2 below)

### Option #1: `conda` environment

1. Clone the repository and `cd` into it
```bash
git clone https://github.com/boschresearch/dist-mbrl.git && cd dist-mbrl
```
2. Create a conda environment
```bash
conda env create --file=environment.yml
```
3. Activate the environment and install the package + dependencies
 ```bash
conda activate dist_mbrl
pip install -e .
```

### Option #2: Docker container.
Make sure `docker` is installed and configured.

1. Build docker image
```bash
cd docker/
./build_docker.sh
```
2. After the image is created, you can access it via
```bash
docker run --rm -ti dist-mbrl
```

## Usage

### Running experiments
The entry point for training is [train.py](dist_mbrl/train/train.py). At the bottom of
the file, you can modify the configuration passed on to the training script. The
agent configurations are generated in [default.py](dist_mbrl/config/default.py) and the model learning configurations are stored in this [YAML file](dist_mbrl/config/mbrl_lib.yaml).
```bash
cd {path_to_repo}/dist_mbrl
python train/train.py
```
### Reproducing paper plots
All the plots shown in the paper can be reproduced by running the provided [Jupyter
notebooks](notebooks).

## Citation
```
@article{luis2023value,
  title={Value-Distributional Model-Based Reinforcement Learning},
  author={Luis, Carlos E and Bottero, Alessandro G and Vinogradska, Julia and Berkenkamp, Felix and Peters, Jan},
  journal={arXiv preprint arXiv:2308.06590},
  year={2023}
}
```


## License
The code is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

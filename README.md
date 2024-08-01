# Deep Learning Weather Prediction Model and Backbone Comparison on Navier-Stokes and WeatherBench

A benchmark to compare different deep learning models and their backbones on synthetic Navier-Stokes and real-world data from [WeatherBench](https://arxiv.org/abs/2002.00469), published in the [ICLR 2024 AI4DiffEq Workshop](https://ai4diffeqtnsinsci.github.io/) and in the [NeurIPS 2024 Datasets and Benchmarks](https://neurips.cc/Conferences/2024/CallForDatasetsBenchmarks) track:

- [Comparing and Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes Dynamics](https://openreview.net/forum?id=jxfjvks0d7) (workshop paper)
- [Comparing and Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes and Atmospheric Dynamics](https://arxiv.org/abs/2407.14129)

If you find this work useful, please cite our paper

```
@article{karlbauer2024comparing,
  title={Comparing and Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes and Atmospheric Dynamics},
  author={Karlbauer, Matthias and Maddix, Danielle C and Ansari, Abdul Fatir and Han, Boran and Gupta, Gaurav and Wang, Yuyang and Stuart, Andrew and Mahoney, Michael W},
  journal={arXiv preprint arXiv:2407.14129},
  year={2024}
}
```


## Getting Started

To install the package, first create an environment, cd into it, and install the DLWPBench package via

```
conda create -n dlwpbench python=3.11 -y && conda activate dlwpbench
pip install -e .
```

In the pip Neuraloperator package, the tucker decomposition for TFNO is not installed, so manually the package from the [source repository](https://github.com/NeuralOperator/neuraloperator) with

```
mkdir packages
cd packages
git clone https://github.com/NeuralOperator/neuraloperator
git checkout 05c01c3  # (optional) use the repository state that is compatible with checkpoints from our work
cd neuraloperator
pip install -e .
pip install -r requirements.txt
cd ../..
```

Moreover, install the `torch-harmonics` package for Spherical Fourier Neural Operators from the [source repository](https://github.com/NVIDIA/torch-harmonics) with the following commands

```
cd packages
git clone https://github.com/NVIDIA/torch-harmonics.git
cd torch-harmonics
pip install -e .
cd ../..
```

To install the CUDA versions of Deep Graph Library, follow [these instructions](https://www.dgl.ai/pages/start.html) and issue

```
pip uninstall dgl -y
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

> [!IMPORTANT]  
> This DGL version requires CUDA 12.1 to be installed, e.g., following [these instructions](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

Finally, change into the benchmark directory, which will be considered the root directory in the following, that is, `cd src/dlwpbench`


## Navier-Stokes

To generate data and run experiments in the synthetic Navier-Stokes environment, please go to [the respective subdirectory](src/nsbench/) and follow the steps detailed there.


## WeatherBench

To download and preprocess data and run experiments in the real-world WeatherBench environment, please go to [the respective subdirectory](src/dlwpbench/) and follow the steps detailed there.


## Resources

Deep learning model repositories that are used in this study:

- HEALPix remapping: https://github.com/CognitiveModeling/dlwp-hpx
- Convolutional LSTM: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
- Fourier Neural Operator: https://github.com/neuraloperator/neuraloperator
- FourCastNet: https://github.com/NVlabs/FourCastNet
- Spherical Fourier Neural Operator: https://github.com/NVIDIA/torch-harmonics
- SwinTransformer: https://github.com/microsoft/Swin-Transformer/tree/main
- Pangu-Weather: https://github.com/lizhuoq/WeatherLearn/blob/master/weatherlearn/models/pangu/pangu.py
- MeshGraphNet: https://github.com/NVIDIA/modulus/tree/main/modulus/models/meshgraphnet
- GraphCast: https://github.com/NVIDIA/modulus/tree/main/modulus/models/graphcast

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

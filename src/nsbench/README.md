# Navier-Stokes Environment

This subdirectory contains details about how to generate Navier-Stokes data and run according experiments.


## Data

The data generation scripts depends on an older version of PyTorch, i.e., v1.6, and we recommend creating a separate environment to generate the data without messing up the PyTorch version for training the models.

Create separate environment, cd into it and install the requirements with pip via

```
conda create -n nsgen python=3.7 -y && conda activate nsgen
pip install torch==1.6.0 "xarray[parallel]" scipy einops tqdm matplotlib
```

Subsequently, run the data generation script with
```
python data/ns_generation/generate_ns_2d.py
```
You can provide additional command-line arguments if desired, as listed when calling `python data/ns_generation/generate_ns_2d.py -h`

Note that the data geneneration can take long, depending on the number of samples that are generated and the simulation configuration. A small data set for experimentation with two samples and a simulation time of 2 can be created, for example, with
```
python data/ns_generation/generate_ns_2d.py -n 5 -t 50 -b 1
```

To generate the train, validation, and test splits as used in Experiment 1 of the paper, run the following commands (following Table 3 in the paper's appendix):

```
python data/ns_generation/generate_ns_2d.py -n 1000 -t 50
python data/ns_generation/generate_ns_2d.py -n 50 -t 50
python data/ns_generation/generate_ns_2d.py -n 200 -t 50
```

## Model Training

Training and evaluation require the `dlwpbench` environment activated, that is `conda activate dlwpbench`.

Model training can be initiated via the training script, e.g., calling
```
python scripts/train.py model=unet data=exploration model.name=unet_example training.epochs=10 device=cuda:0
```
to train an exemplary U-Net. Note that this will require an example dataset as generated in the section above with the arguments `-n 5 -t 50 -b 1`

An exhaustive list containing all training commands used in the paper is provided in [train_commands.txt](scripts/train_commands.txt).


## Evaluation

To evaluate a successfully trained model, run
```
python scripts/evaluate.py -c outputs/unet_example
```
The model.name must be given to the -c argument. The evaluation script will compute the RMSE metric and print it to console, create an RMSE-over-leadtime plot called `rmse_plot.pdf` in the `.` directory, and write a video to the `outputs/unet_example` directory.

# WeatherBench Environment

This subdirectory contains details about how to download and preprocess the WeatherBench data and run according experiments.


## Data

### Download

Download the 5.625 degree data from the [TUM server](https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2F)---which can also be done via the command line, as detailed [here](https://mediatum.ub.tum.de/1524895)--- and store the folders containing the separate variables in `(src/dlwpbench/)data/netcdf/weatherbench/`, i.e.,

```
.
|-- data
|   |-- netcdf
|       |-- weatherbench
|           |-- 10m_u_component_of_wind
|           |-- 10m_v_component_of_wind
|           |-- 2m_temperature
|           |-- constants
|           |-- geopotential
|           |-- potential_vorticity
|           |-- relative_humidity
|           |-- specific_humidity
|           |-- temperature
|           |-- toa_incident_solar_radiation
|           |-- total_cloud_cover
|           |-- total_precipitation
|           |-- u_component_of_wind
|           |-- v_component_of_wind
|           `-- vorticity
```

> [!IMPORTANT]  
> The `5.625deg` suffix must not be part of the directory names and might required to be removed, depending on the download method. Also, 'geopotential_500' and 'temperature_850' must not be part of the directory tree.

> [!TIP]
> Exemplarily, download the `2m_temperature` field via command line, uncompress it, and move it to the correct directory with
> ```
> mkdir -p data/netcdf/weatherbench
> rsync -P rsync://m1524895@dataserv.ub.tum.de/m1524895/5.625deg/2m_temperature/2m_temperature_5.625deg.zip data/netcdf/weatherbench/2m_tmperature_5.625deg.zip
> ```
> enter password `m1524895`
> ```
> unzip data/netcdf/weatherbench/2m_temperature_5.625deg.zip -d data/netcdf/weatherbench/2m_temperature_5.625deg
> rm data/netcdf/weatherbench/2m_temperature_5.625.zip
> ```

The entire dataset with `250GB` can be downloaded from [https://dataserv.ub.tum.de/s/m1524895/download?path=/5.625deg&files=all_5.625deg.zip](https://dataserv.ub.tum.de/s/m1524895/download?path=/5.625deg&files=all_5.625deg.zip)

### Convert to zarr

For faster data loading from disk when training and evaluating models, the data can be converted from `netcdf` (with `.nc` extension) to `zarr` (with `.zarr` extension) using the [nc_to_zarr.py](data/processing/nc_to_zarr.py) script. That is
```
python data/processing/nc_to_zarr.py
```
In the following, the files are expected to be in `zarr` format, but the package is compatible with `netcdf` files as well. Working with `.nc` files, though, requires modifications in the used [data config file](configs/data/weatherbench.yaml), by setting `data_path: data/netcdf/weatherbench/` and `engine: netcdf4`.

Statistics for data normalization are computed and contained in the [datasets.py](data/datasets/datasets.py) class and can be recomputed using the `self._compute_statistics()` function in that class.

### Project to HEALPix

The conversion from the rectangular LatLon to the HEALPix mesh can be realized by calling the [healpix_mapping.py](data/processing/healpix_mapping.py) script with according arguments. For example, to project `2m_temperature_5.625deg` files in `zarr` file format to HEALPix, run
```
python data/processing/healpix_mapping.py -v 2m_temperature_5.625deg
```
or provide additional arguments if desired.

This will convert the data of shape `[time, (level), lat, lon]` to shape `[time, (level), face, height, width]`, where `level` is optional, depending on the variable, `face = 12` and `height = width = nside`, with `nside` specified in the conversion process.


## Training

Training and evaluation require the `dlwpbench` environment activated, thus
```
conda activate dlwpbench
```

Model training can be invoked via the training script, e.g., calling
```
python scripts/train.py model=unet model.type=UNetHPX model.name=unet_hpx_example data=example training.epochs=10 device=cuda:0 data=example model.constant_channels=0 model.prescribed_channels=0 model.prognostic_channels=1
```
to train an exemplary U-Net on `2m_temperature` on the HEALPix projection.

> [!Note]
> Running the command above will require the `2m_temperature` variable converted to `zarr` and projected to HEALPix, as descrobed in the Data section. Also, make sure to follow the naming convention of the data directory tree above, i.e., remove `5.625deg` from the `2m_temperature_5.625deg` directory name.

> [!Note]
> GraphCast requires a pre-generated `icosahedral.json` file, which specifies the mesh on which the model operates. These files can be generated using the [icospheres.py](models/graphcast/utils/icospheres.py) script for arbitrary mesh resolutions (number of hierarchical levels) and must be linked in GraphCast's [config.yaml](configs/model/graphcast.yaml) file. Originally GraphCast uses six levels, but here it must be reduced to three due to the smaller resolution of the 5.625° WeatherBench data.


## Pretrained Models and Exemplary Outputs

A selection of pretrained models (see list below) can be downloaded [here](https://unitc-my.sharepoint.com/:u:/g/personal/siaka01_cloud_uni-tuebingen_de/EamiTF1_xnlLpTGPOWhcjq4BJijOiiSxhkgnx-DZdw0MHA?e=7BpKHp). To use a respective model, place its `*.ckpt` into the according `checkpoints` directory in the model's `outputs` folder, e.g., `outputs/clstm16m_cyl_4x228_v2/checkpoints/clstm1m_cyl_4x228_v2_best.ckpt`.

The drive contains weights and exemplary outputs of the following models:

```
clstm16m_cyl_4x228_v2
clstm16m_hpx8_4x228_v2
unet128m_cyl_128-256-512-1024-2014_v2
unet16m_hpx8_92-184-368-736_v0
swint2m_cyl_d88_l2x4_h2x4_v0
swint16m_hpx8_d120_l3x4_h3x4_v2
pangu32m_d216_h6-12-12-6_v1
fno2d64m_cyl_d307_v0
tfno2d128m_cyl_d477_v0
fcnet4m_emb272_nopos_l6_v1
fcnet8m_emb384_nopos_p2x4_l6_v2
fcnet64m_emb940_nopos_p4x4_l8_v0
sfno2d128m_cyl_d686_equi_nonorm_nopos_v0
mgn500k_l4_d116_v0
mgn32m_l8_d470_v0
gcast500k_p4_b1_d99_v0
gcast16m_p4_b1_d565_v2
```


## Evaluation

To evaluate a successfully trained model, run
```
python scripts/evaluate.py -c outputs/unet_hpx_example
```
The model.name must be provided as -c argument. The evaluation script will compute the ACC and RMSE metrics and write them to `outputs/unet_hpx_example/evaluation/`, create an RMSE-over-leadtime plot called `rmse_plot.pdf` in the `.` directory, and write a video to the `outputs/unet_hpx_example/evaluation/videos/` directory.

More plots in line with the paper can be generated with the [plot_results.py](scripts/plot_results.py) script, which requires the installation of the `cartopy` package.

> [!NOTE]  
> To generate videos, the `ffmpeg` package will be required.

### Baselines 

`Persistence` and `Climatology` baselines can be generated using the [build_baselines.py](scripts/build_baselines.py) script. This requires, however, another deep learning model trained and evaluated before, as the script follows the selected variables and resolution in the respective forecast.

#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import time
import numpy as np
import torch as th
import pandas as pd
import xarray as xr


class WeatherBenchDataset(th.utils.data.Dataset):

    # Statistics are computed over the training period from 1979-01-01 to 2014-12-31 using the
    # self._compute_statistics() of this class
    STATISTICS = {
        "u10": {
            "file_name": "10m_u_component_of_wind",
            "mean": -0.09109575,
            "std": 5.547917
        },
        "v10": {
            "file_name": "10m_v_component_of_wind",
            "mean": 0.2246149,
            "std": 4.7760262
        },
        "t2m": {
            "file_name": "2m_temperature",
            "mean": 278.44608,
            "std": 21.24761
        },
        "z": {
            "file_name": "geopotential",
            "mean_overall": 77781.484,
            "std_overall": 59468.266,
            "level": {
                50: {"mean": 199354.859375, "std": 5869.6171875},
                100: {"mean": 157619.953125, "std": 5501.45703125},
                150: {"mean": 133120.28125, "std": 5816.61767578125},
                200: {"mean": 115311.0078125, "std": 5814.1572265625},
                250: {"mean": 101206.546875, "std": 5531.34130859375},
                300: {"mean": 89399.90625, "std": 5087.197265625},
                400: {"mean": 69970.0703125, "std": 4146.70654296875},
                500: {"mean": 54107.8671875, "std": 3349.03125},
                600: {"mean": 40642.8515625, "std": 2691.708740234375},
                700: {"mean": 28924.94921875, "std": 2132.567626953125},
                850: {"mean": 13748.33203125, "std": 1467.218994140625},
                925: {"mean": 7014.69921875, "std": 1225.9998779296875},
                1000: {"mean": 738.5037841796875, "std": 1069.619140625}
            }
        },
        "z500": {
            "file_name": "geopotential_500",
            "mean": 54107.863,
            "std": 3349.0322
        },
        "pv": {
            "file_name": "potential_vorticity",
            "mean_overall": -1.3307364e-07,
            "std_overall": 1.0213214e-05,
            "level": {
                50: {"mean": -4.3703346364054596e-07, "std": 3.1687788577983156e-05},
                100: {"mean": 7.641371091438032e-09, "std": 1.1946386621275451e-05},
                150: {"mean": -5.214129217279151e-08, "std": 7.157286745496094e-06},
                200: {"mean": 1.3350896210795327e-07, "std": 5.499758572113933e-06},
                250: {"mean": 1.2745846333928057e-07, "std": 4.152241672272794e-06},
                300: {"mean": 3.9340093849205005e-08, "std": 2.5396282126166625e-06},
                400: {"mean": 7.962564829711027e-09, "std": 9.462796697334852e-07},
                500: {"mean": 2.153144329497536e-09, "std": 6.550909574798425e-07},
                600: {"mean": -1.1735989602357222e-07, "std": 2.3843126655265223e-06},
                700: {"mean": -4.165769382780127e-07, "std": 3.95903225580696e-06},
                850: {"mean": -4.6878389525772945e-07, "std": 4.7122630348894745e-06},
                925: {"mean": -3.2587902865088836e-07, "std": 5.099419922771631e-06},
                1000: {"mean": -2.3024716711006477e-07, "std": 5.707132459065178e-06}
            }
        },
        "r": {
            "file_name": "relative_humidity",
            "mean_overall": 48.673717,
            "std_overall": 36.1341,
            "level": {
                50: {"mean": 6.4983978271484375, "std": 15.516167640686035},
                100: {"mean": 26.236711502075195, "std": 33.42652130126953},
                150: {"mean": 26.779787063598633, "std": 32.22017288208008},
                200: {"mean": 35.66021728515625, "std": 34.10388946533203},
                250: {"mean": 47.30855941772461, "std": 34.410457611083984},
                300: {"mean": 53.88154220581055, "std": 33.86103820800781},
                400: {"mean": 52.60612106323242, "std": 34.085533142089844},
                500: {"mean": 50.388450622558594, "std": 33.47996139526367},
                600: {"mean": 51.58530044555664, "std": 32.6628303527832},
                700: {"mean": 54.95896911621094, "std": 31.358800888061523},
                850: {"mean": 69.12675476074219, "std": 26.306550979614258},
                925: {"mean": 79.09678649902344, "std": 21.494625091552734},
                1000: {"mean": 78.63064575195312, "std": 18.144744873046875}
            }
        },
        "q": {
            "file_name": "specific_humidity",
            "mean_overall": 0.0017692719,
            "std_overall": 0.00354159,
            "level": {
                50: {"mean": 2.6648656330507947e-06, "std": 3.768455485442246e-07},
                100: {"mean": 2.6282473299943376e-06, "std": 5.841049528498843e-07},
                150: {"mean": 5.2293335102149285e-06, "std": 3.7594475088553736e-06},
                200: {"mean": 1.9305318346596323e-05, "std": 2.2453708879766054e-05},
                250: {"mean": 5.7458048104308546e-05, "std": 7.384442142210901e-05},
                300: {"mean": 0.00012682017404586077, "std": 0.00016722768486943096},
                400: {"mean": 0.00038354037678800523, "std": 0.000503877701703459},
                500: {"mean": 0.0008488624007441103, "std": 0.0010718436678871512},
                600: {"mean": 0.0015365154249593616, "std": 0.001762388157658279},
                700: {"mean": 0.0024213239084929228, "std": 0.002534921746701002},
                850: {"mean": 0.004560411442071199, "std": 0.004099337849766016},
                925: {"mean": 0.006017220206558704, "std": 0.005065328907221556},
                1000: {"mean": 0.007018645294010639, "std": 0.00591137632727623}
            }
        },
        "t": {
            "file_name": "temperature",
            "mean_overall": 243.07938,
            "std_overall": 29.063475,
            "level": {
                50: {"mean": 212.51316833496094, "std": 10.26008129119873},
                100: {"mean": 208.4076385498047, "std": 12.5313081741333},
                150: {"mean": 213.30043029785156, "std": 8.939778327941895},
                200: {"mean": 218.03001403808594, "std": 7.188199043273926},
                250: {"mean": 222.72799682617188, "std": 8.516169548034668},
                300: {"mean": 228.82302856445312, "std": 10.704428672790527},
                400: {"mean": 242.09046936035156, "std": 12.685832977294922},
                500: {"mean": 252.9086151123047, "std": 13.064330101013184},
                600: {"mean": 261.095703125, "std": 13.417461395263672},
                700: {"mean": 267.34954833984375, "std": 14.770682334899902},
                850: {"mean": 274.518798828125, "std": 15.591468811035156},
                925: {"mean": 277.30401611328125, "std": 16.09958267211914},
                1000: {"mean": 280.95977783203125, "std": 17.14033317565918}
            }
        },
        "tcc": {
            "file_name": "total_cloud_cover",
            "mean": 0,  # Do not normalize as it is already in [0, 1]
            "std": 1
        },
        "u": {
            "file_name": "u_component_of_wind",
            "mean_overall": 7.2384787,
            "std_overall": 14.196661,
            "level": {
                50: {"mean": 5.628922462463379, "std": 15.274863243103027},
                100: {"mean": 10.256104469299316, "std": 13.494525909423828},
                150: {"mean": 13.524600982666016, "std": 16.02048110961914},
                200: {"mean": 14.20456600189209, "std": 17.651451110839844},
                250: {"mean": 13.345967292785645, "std": 17.9445858001709},
                300: {"mean": 11.803243637084961, "std": 17.101341247558594},
                400: {"mean": 8.812711715698242, "std": 14.329045295715332},
                500: {"mean": 6.551909446716309, "std": 11.97260570526123},
                600: {"mean": 4.793718338012695, "std": 10.324584007263184},
                700: {"mean": 3.293602466583252, "std": 9.192586898803711},
                850: {"mean": 1.3884272575378418, "std": 8.179688453674316},
                925: {"mean": 0.5692682266235352, "std": 7.943700790405273},
                1000: {"mean": -0.07277797907590866, "std": 6.1379289627075195}
            }
        },
        "v": {
            "file_name": "v_component_of_wind",
            "mean_overall": 0.03683709,
            "std_overall": 9.292757,
            "level": {
                50: {"mean": 0.004283303394913673, "std": 7.03949499130249},
                100: {"mean": 0.014493774622678757, "std": 7.4672160148620605},
                150: {"mean": -0.035589370876550674, "std": 9.562729835510254},
                200: {"mean": -0.04457025229930878, "std": 11.869003295898438},
                250: {"mean": -0.030027257278561592, "std": 13.37221908569336},
                300: {"mean": -0.022819651290774345, "std": 13.336838722229004},
                400: {"mean": -0.017354506999254227, "std": 11.227628707885742},
                500: {"mean": -0.023081326857209206, "std": 9.177905082702637},
                600: {"mean": -0.030738739296793938, "std": 7.795202732086182},
                700: {"mean": 0.040801987051963806, "std": 6.884781360626221},
                850: {"mean": 0.16805106401443481, "std": 6.281022548675537},
                925: {"mean": 0.23628801107406616, "std": 6.482055187225342},
                1000: {"mean": 0.21914489567279816, "std": 5.314945220947266}
            }
        },
        "vo": {
            "file_name": "vorticity",
            "mean_overall": -1.7855017e-07,
            "std_overall": 4.3787608e-05,
            "level": {
                50: {"mean": -1.031515466820565e-06, "std": 1.8286706108483486e-05},
                100: {"mean": -7.213484991552832e-07, "std": 2.102252074109856e-05},
                150: {"mean": -5.153465849616623e-07, "std": 2.9635022656293586e-05},
                200: {"mean": -3.664559926619404e-07, "std": 4.069161514053121e-05},
                250: {"mean": -2.50777844712502e-07, "std": 5.349168350221589e-05},
                300: {"mean": -1.8695526193823753e-07, "std": 6.158988253446296e-05},
                400: {"mean": -1.2000741378415114e-07, "std": 5.8507790527073666e-05},
                500: {"mean": -5.564273664049324e-08, "std": 4.774249464389868e-05},
                600: {"mean": -7.289649062158787e-08, "std": 4.137582436669618e-05},
                700: {"mean": 5.301599799167889e-07, "std": 4.1167502786265686e-05},
                850: {"mean": 5.3918963516252916e-08, "std": 4.5911368943052366e-05},
                925: {"mean": 3.541983346622146e-07, "std": 4.7632172936573625e-05},
                1000: {"mean": 6.151589104774757e-08, "std": 3.8373880670405924e-05}
            }
        },
        "tisr": {
            "file_name": "toa_incident_solar_radiation",
            "mean": 1074504.8,
            "std": 1439846.4
        },
        "orography": {
            "file_name": "constants",
            "mean": 379.4976,
            "std": 859.87225
        },
        "slt": {
            "file_name": "constants",
            "mean": 0,  # Do not normalize sind it is a discrete variable
            "std": 1
        },
        "lsm": {
            "file_name": "constants",
            "mean": 0,  # Do not normalize since it is in [0, 1] already
            "std": 1
        },
        "lat2d": {
            "file_name": "constants",
            "mean": 0,
            "std": 51.936146191742026
        },
        "lon2d": {
            "file_name": "constants",
            "mean": 177.1875,
            "std": 103.9103617607503
        }
    }

    def __init__(
            self,
            data_path: str,
            prognostic_variable_names_and_levels: dict,
            prescribed_variable_names: list = None,
            constant_names: list = None,
            start_date: np.datetime64 = np.datetime64("1979-01-01"),
            stop_date: np.datetime64 = np.datetime64("2014-12-31"),
            timedelta: int = 6,
            init_dates: np.array = None,
            sequence_length: int = 15,
            noise: float = 0.0,
            normalize: bool = False,
            downscale_factor: int = 1,
            context_size: int = 1,
            engine: str = "netcdf4",
            height: int = 32,
            width: int = 64,
            **kwargs
        ):
        """
        Constructor of a pytorch dataset module.

        :param data_name: The name of the data
        :param data_start_date: The first time step of the data on the disk
        :param data_stop_date: The last time step of the data on the disk
        :param used_start_date: The first time step to be considered by the dataloader
        :param used_stop_date: The last time step to be considered by the dataloader
        :param data_src_path: The source path of the data
        :param sequence_length: The number of time steps used for training
        """

        self.stats = WeatherBenchDataset.STATISTICS
        self.prognostic_variable_names_and_levels = prognostic_variable_names_and_levels
        self.prescribed_variable_names = prescribed_variable_names
        self.constant_names = constant_names

        self.timedelta = timedelta
        self.sequence_length = sequence_length
        self.noise = noise
        self.normalize = normalize
        self.downscale_factor = downscale_factor
        self.context_size = context_size
        self.init_dates = init_dates

        # Get paths to all (yearly) netcdf/zarr files
        fpaths = []
        for p in prognostic_variable_names_and_levels:
            fpaths += glob.glob(os.path.join(data_path, self.stats[p]["file_name"], "*"))
        for p in prescribed_variable_names:
            fpaths += glob.glob(os.path.join(data_path, self.stats[p]["file_name"], "*"))
        if constant_names: 
            fpaths += glob.glob(os.path.join(data_path, "constants", "*"))

        print(f"\tLoading dataset from {start_date} to {stop_date} into RAM...", sep=" ", end=" ", flush=True)
        a = time.time()
        # Load the data as xarray dataset
        self.ds = xr.open_mfdataset(fpaths, engine=engine).sel(time=slice(start_date, stop_date, timedelta))

        # Chunk and load dataset to memory (distinguish between HEALPix, i.e., when "face" in coords, and LatLon mesh)
        if "face" in self.ds.coords:
            chunkdict = dict(time=self.sequence_length+1, face=12, height=height, width=width)
        else:
            chunkdict = dict(time=self.sequence_length+1, lat=height, lon=width)
        self.ds = self.ds.chunk(chunkdict).load()
        print(f"took {time.time() - a} seconds")

        # Downscale dataset if desired
        if downscale_factor > 1:
            assert "face" not in self.ds.coords, "Downscaling only supported with LatLon and not with HEALPix data."
            self.ds = self.ds.coarsen(lat=downscale_factor, lon=downscale_factor).mean()

        # Prepare the constants to shape [#constants, lat, lon]
        if constant_names:
            constants = []
            for c in constant_names:
                lazy_data = self.ds[c]
                if self.normalize: lazy_data = (lazy_data-self.stats[c]["mean"])/self.stats[c]["std"]
                constants.append(lazy_data.compute())
            self.constants = np.expand_dims(np.float32(np.stack(constants)), axis=0)
        else:
            self.constants = th.nan  # Dummy tensor is returned if no constants are used
        
    def __len__(self):
        if self.init_dates is None:
            # Randomly sample initialization dates from the dataset
            return (self.ds.sizes["time"]-self.sequence_length)//self.sequence_length
        else:
            return len(self.init_dates)

    def __getitem__(self, item):
        """
        return: Four arrays of shape [batch, time, dim, (face), height, width], where face is optionally added when
            operating on the HEALPix mesh.
        """

        item = item*self.sequence_length if self.init_dates is None else item

        # Load the (normalized) prescribed variables of shape [time, #prescribed_vars, lat, lon] into memory
        if self.prescribed_variable_names:
            prescribed = []
            for p in self.prescribed_variable_names:
                manual_tisr = False
                if self.init_dates is None:
                    lazy_data = self.ds[p].isel(time=slice(item, item+self.sequence_length))
                else:
                    lazy_data = self.ds[p].sel(
                        time=slice(self.init_dates[item],
                                   self.init_dates[item]+pd.Timedelta(f"{self.sequence_length*self.timedelta}h"))
                    )
                if self.init_dates is not None and self.sequence_length > len(lazy_data.time):
                    # Augment TISR with values from 2017 when exceeding the date of the stored data
                    manual_tisr = True
                    diff = self.sequence_length - len(lazy_data.time)
                    start_date = self.init_dates[item]
                    stop_date = self.init_dates[item] + pd.Timedelta(f"{self.sequence_length*self.timedelta}h")
                    dates = pd.date_range(start=start_date, end=stop_date, freq=f"{self.timedelta}h")
                    lazy_data = lazy_data.values
                    tmp = list()
                    # Overide year with 2017 under consideration of leap years
                    for date in dates[-diff:]:
                        date = date.replace(year=2017, day=28) if date.month == 2 and date.day > 28 else date.replace(year=2017)
                        tmp.append(self.ds.tisr.sel(time=date).values)
                    lazy_data = np.concatenate((lazy_data, np.array(tmp)))
                if self.normalize: lazy_data = (lazy_data-self.stats[p]["mean"])/self.stats[p]["std"]
                prescribed.append(lazy_data.compute() if not manual_tisr else lazy_data)  # Loads data into memory
            prescribed = np.float32(np.stack(prescribed, axis=1))
        else:
            prescribed = th.nan  # Dummy tensor returned if no prescribed variables are used

        # Load the (normalized) prognostic variables of shape [time, #prognostic_vars, lat, lon] into memory
        prognostic = []
        for p in self.prognostic_variable_names_and_levels:
            if self.init_dates is None:
                lazy_data = self.ds[p].isel(time=slice(item, item+self.sequence_length+1))
            else:
                lazy_data = self.ds[p].sel(
                    time=slice(self.init_dates[item],
                               self.init_dates[item]+pd.Timedelta(f"{(self.sequence_length+1)*self.timedelta}h"))
                )
            # Load the data to memory
            if "level" in lazy_data.coords:
                for l in self.prognostic_variable_names_and_levels[p]:
                    lazy_data_l = lazy_data.sel(level=l)
                    if self.normalize:
                        lazy_data_l = (lazy_data_l-self.stats[p]["level"][l]["mean"])/self.stats[p]["level"][l]["std"]
                    prognostic.append(lazy_data_l.compute())
            else:
                if self.normalize: lazy_data = (lazy_data-self.stats[p]["mean"])/self.stats[p]["std"]
                prognostic.append(lazy_data.compute())
        prognostic = np.float32(np.stack(prognostic, axis=1))
        # Append zeros to the prog. vars when exceeding the date of the stored data (required for long rollouts)
        if len(prognostic) < self.sequence_length:
            diff = self.sequence_length - len(prognostic)
            fill = np.zeros((diff, *prognostic.shape[1:]), dtype=np.float32)
            prognostic = np.concatenate((prognostic, fill), axis=0)

        # Separate prognostic variables into inputs and targets
        target = prognostic[1:]
        prognostic = prognostic[:-1] + np.float32(np.random.randn(*prognostic[:-1].shape)*self.noise)

        return self.constants, prescribed, prognostic, target[self.context_size:]

    
    def compute_statistics(self):
        """
        Computes the statistics of the given prognostic variables and prints mean and standard deviation to console
        """
        # Constants
        for c in self.constant_names:
            print(c)
            lazy_data = self.ds[c]
            mean, std = lazy_data.mean().values, lazy_data.std().values
        # Prescribed variables
        for p in self.prescribed_variable_names:
            print(p)
            lazy_data = self.ds[p]
            mean, std = lazy_data.mean().values, lazy_data.std().values
            print(f'"mean": {mean},\n"std": {std}')
        # Prognostic variables (optionally with levels)
        for p in self.prognostic_variable_names_and_levels:
            print(p)
            lazy_data = self.ds[p]
            if "level" in lazy_data.coords:
                for l in lazy_data.level.values:
                    mean, std = lazy_data.sel(level=l).mean().values, lazy_data.sel(level=l).std().values
                    print(f'{l}: {{"mean": {mean}, "std": {std}}},')
            else:
                mean, std = lazy_data.mean().values, lazy_data.std().values
                print(f'"mean": {mean},\n"std": {std}')
            print()


if __name__ == "__main__":

    # Example of creating a WeatherBench dataset for the PyTorch DataLoader
    dataset = WeatherBenchDataset(
        data_path="data/netcdf/weatherbench/",
        prognostic_variable_names_and_levels={
            #"u10": [],   # 10m_u_component_of_wind
            #"v10": [],   # 10m_v_component_of_wind
            #"t2m": [],   # 2m_temperature
            "z": [500],     # geopotential
            #"z500",  # geopotential_500
            #"pv": [500],    # potential_vorticity
            #"r": [500, 700],     # relative_humidity
            #"q": [500],     # specific_humidity
            #"t": [50, 500, 850],     # temperature
            #"tcc": [],   # total_cloud_cover
            #"u": [500],     # u_component_of_wind
            #"v": [500],     # v_component_of_wind
            #"vo": [500]     # vorticity
        },
        prescribed_variable_names=[
            #"tisr"   # top of atmosphere incoming solar radiation
        ],
        constant_names=[
            #"orography",
            #"lsm",   # land-sea mask
            #"slt",   # soil type
            #"lat2d",
            #"lon2d"
        ],
        sequence_length=15,
        noise=0.0,
        normalize=True,
        downscale_factor=None
    )

    train_dataloader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    for constants, prescribed, prognostic, target in train_dataloader:
        print(constants.shape, prescribed.shape, prognostic.shape, target.shape)
        break

    print(dataset)

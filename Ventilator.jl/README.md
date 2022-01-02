# Ventilator.jl

## Overview

This is a Julia package used for experiments on Google Brain competition on Kaggle: Ventilator Pressure Prediction.

The package is complete and can be used to run experiments out of the box (if you have the data available). For more details about how to run some experiments yourself, go to section **How to use this code base**.

## Structure

This folder contains a whole package.

### Source

Folder `src` contains all source files with the main file being `Ventilator.jl` which creates the package itself and exports functions created. Most functions are documented, for more details, go to the individual scripts and read through the code.

Files:

- `data.jl` contains functions to load and preprocess data
- `features.jl` contains preprocessing functions for feature engineering and transformations
- `utils.jl` contains helper functions for model constructors, minibatching, etc.

### Scripts

Folder `scripts` contains scripts used for experimenting. The subfolder of interest is `run_scripts`  which contains complete scripts for experimentation. For more details about models, scripts and other features, there is another `README.md` file in the `scripts` folder.

## Notebooks

If you just want to see how all this works, open the notebook in the `notebooks` folder and you can see. Also, you can try it yourself in Google Colab, just make sure you follow the instructions ;)

## How to use this code base

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Ventilator.jl

It is authored by **masenka31**.

To (locally) reproduce this project, do the following:

0. Download Julia (ideally 1.6.x version) and download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.

1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

   This will install all necessary packages for you to be able to run the scripts and
   everything should work out of the box, including correctly finding local paths.

To run any scripts, continue with these steps:

2. Download data from Google drive at [Google Brain Data](https://drive.google.com/drive/folders/1oqn790bd8s0bDRE52vfhw1Z7ozeqt4-9?usp=sharing) and copy the .csv files to the folder `data`.

3. Navigate to `Ventilator.jl/scripts/run_scripts` in the terminal.

4. Run any of the following
   
   - `julia simple_run.jl`
   - `julia rnn_onehot.jl`
   - `julia rnn_onehot_engineered.jl`
   - `julia rnn_lags.jl`
   - `julia --threads 9 9models.jl`

   Beware that the maximum training time is probably quite large. To reduce it, just pass an extra parameter in the command line with the maximum number of training seconds like
   ```bash
   julia simple_run.jl 600
   ```
   for 10 minutes of training.

   The script will start in the terminal and display outputs, usually indicators of data loaded, training started and the validation loss during training. After the script finishes, data is saved to the `data` folder to subfolders corresponding to individual scripts.
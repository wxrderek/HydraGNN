# Task: Downstream Density

Status: TODO
Estimated scope: large

---

# Objective

Implement plotting functionality, called during the inference pipeline, in the `examples/qmugs/` directory.

Ensure you leave inline comments as specified in the Validation section of this markdown.

Specifics below.

## figs.py

- the `main()` function of this script will be optionally called by `qmugs_densmat_inference.py` to create plots of the inference run. however, it should be defined such that it can be run alone in a shell script if giving the log directory of a previous inference run. if also provided the artifacts directory and electron density files are detected, then electron density visuals may be done. if not provided the artifacts directory, skip electron density visualizations with a warning and proceed with the other visualizations.

- `main()` assumes the outputs of `qmugs_densmat_inference.py`, namely the JSON's of per matrix results. as stated above, if the artifacts directory is provided, it can then make electron density plots in addition, but does not error if not provided. 

- the plots made by `main()` will include:
    - optionally load the specified test set, if the call is not during inference. if it is, the dataset loaded can just be passed in.
    - if the call is not during inference, read in inference results from the specified directory.
    - two possibilities for density-based visuals:
        - randomly sample n molecules (default to 5) from the test set and do plotting for all conformers of these molecules
        - cherrypick the n conformers (default to 10) from the test set with the lowest RMSE
    - ensure that this sampling takes place before plotting, as some plotting features may rely on rerunning the inference on a couple samples. note is the call of `main()` is not during inference time, we need to redeclare the `QMugsInference` class using the config saved by `qmugs_densmat_inference.py` in the log directory. if the information saved is insufficient to declare the class and run quick inference on the sampled samples, add the remaining necessary information in the config written by `qmugs_densmat_inference.py`. 
    - density-based visuals:
        - density matrix visuals, after preprocessing density matrices and masks (if masks are present):
            1. heatmaps of true matrices
            2. heatmaps of predicted matrices
            3. heatmaps of difference between true and predited
            4. heatmaps of absolute difference between true and predited
            5. 1-4, but with the mask overlaid (if masks are present); you may refer to the mask plotting function in `utils/utils.py`, but do NOT use this directly
            6. a reconstruction scatter for each individual matrix, i.e. predicted vs. true values of every entry (normalized). note the inference must be recomputed here, but only for the sampled matrices
            7. a reconstruction scatter including entries of all sampled matrices
        - electron density (as scalar field) visuals, after preprocessing either cubeprops or recomputed manual densities using the same pipeline as in `QMugsInference` in `qmugs.py` using functions in `utils/inference_utils.py` (by in 3D, i mean 2D projection of object rendered in 3D):
            1. a 3D plot of the molecule, surrounded by the true density as a point cloud, using conventional packages such as RDKit
            2. an analogous 3D plot for the predicted density
            3. an analogous 3D plot for the absolute difference between true and predicted
    - other downstream visuals:
        - if dipole moments are calculated (either during the inference run or, if this is run with an existing inference run, as specified in the config file saved during inference), plot:
            1. the molecule, with two arrows pointing through the center, one corresponding to the predicted dipole moment, one to the true dipole moment
            2. same as above, but with arrows normalized
        - no need to have support for others calculations yet.
    - evaluation metrics (basically everything written to `per_matrix_metrics.json`):
        1. all evaluation metrics across molecule sizes (number of atoms), with error bars according to standard deviations
        2. all evaluation metrics across density matrix dimensions, with  error bars according to standard deviations

- all plots should be saved to the inference run log directory, either in `qmugs_densmat_inference.py` if this is run by `qmugs_densmat_inference.py`, or as specified it this is run after an inference run.
    
- all helpers needed to preprocess things for plotting should be here.

- there is no need to pass in every single possible config as a CLI argument for a `main()` call outside of an inference run. do include the following:
    - the inference log directory for the run to plot figures for
    - if present/needed for densities, the artifacts directory of the inference run
    - flags for whether to randomly sample molecules to plot, or to cherrypick the best conformers to plot
    - flags for whether to run the density and dipole visuals (basically, visuals that require picking samples from the test set); the plots of evaluation metrics should always run

- specific plotting functionalities to call here will be in the scripts described below.

## density_matrix_visualizers.py

- in `figs/density_matrix_visualizers.py`, add a `density_matrix_heatmap()` function that plots and saves a heatmap of a density matrix in a specified directory (defaults to the log directory of the inference run). make this function maxmally customizable under the matplotlib framework. set sensible defaults, possibly overridden by the calls in `figs.py`.

- in the same file, add a `density_matrix_heatmap_with_mask()` function that wraps the `density_matrix_heatmap()` function, but takes in an additional argument of the mask, and overlays it.

- for both functions, assume the matrix/mask have been preprocessed before passing in. the preprocessing will be done in`figs.py`'s `main()` function by helper functions in `figs.py`.

## density_visualizers.py

- in `figs/density_visualizers.py`, add a `molecule_visual()` function that uses a conventional chemistry package to just visualize a molecule in 3D, with sensible defaults. this should also save the file to the desired directory.

- in the same file, add a `density_visual()` function that overlays the electron density as a point cloud over the given molecule, possibly using `molecule_visual()` if needed. again, try to use a conventional chemistry package to do this. 

- for both functions, assume the density has been preprocessed before passing in. the preprocessing will be done in`figs.py`'s `main()` function by helper functions in `figs.py`.

## dipole_moment_visualizers.py

- in `figs/dipole_moment_visualizers.py`, add a `dipole_moment_visual()` function that uses a conventional chemistry package to visualize a molecule in 3D, plus a passed-in dipole moment as an arrow passing through the geometric center of the molecule.

- in the same file, as a `multiple_dipole_moment_visual()` that takes in a list of vectors corresponding to dipole moments instead of one, and plots them all as arrows through the center of the molecule.

- for both functions, assume the dipole moment has been preprocessed before passing in. the preprocessing will be done in`figs.py`'s `main()` function by helper functions in `figs.py`.

## metrics_visualizers.py

- in `figs/metrics_visualizers.py`, add a `matplotlib` based function that plots a given metric's values over number of atoms, matrix size, or any other per-molecule feature, over all conformers in the given inference run.

- in the same file, a simple function to plot the reconstruction scatter for a given list of true and predicted matrices.

## Other updates

- update `qmugs_densmat_inference.py` to optionally run plotting with `main()` in `figs.py` during the inference procedure.

- update `qmugs_inference_perlmutter.sh` to take in a flag on whether to run plotting, and if plotting is run, whether to run the plots that need to sample molecules/conformers and rerun inference on them.

- add a new shell script in `scripts/` called `qmugs_figs_perlmutter.sh` that runs figures given the log directory of a previous inference run, in the same style as the rest of the scripts in `scripts/`.

---

# Motivation

Create visualizations to evaluate model performance.

---

# Files to add / edit

- `figs.py`
- `figs/density_matrix_visualizers.py`
- `figs/density_visualizers.py`
- `figs/dipole_moment_visualizers.py`
- `figs/metrics_visualizers.py`
- `qmugs_figs_perlmutter.sh`
- `qmugs_densmat_inference.py`
- `qmugs_inference_perlmutter.sh`
- `qmugs.py` if needed

---

# Files to refer to

Reference the files I mentioned in the Objective section of this Markdown.

Reference other Python scripts in `qmugs` for commenting style.

---

# Constraints

Edit only the files specified above. 

Do NOT:

- change random seeds
- copy any changes to the two train running directories
- refer to ANY FILES in the train-running directories, namely `qmugs_run` and `qmugs_run_mask`

Prefer:

- minimal edits
- readable code for scientists

---

# Validation

Do NOT run code.
I will validate myself after the code edits. 

Do the following as preliminary validation:

- add inline comments to indicate tensor sizes whenever application, on your new code only; follow the style of other scripts in the `qmugs` directory
- add short, one sentence docstrings to all new methods; follow the style of other scripts in the `qmugs` directory

---

# Deliverables

Provide: 

1. summary of changes, with brief but precise explanations
2. files modified
3. risks
4. a detailed list of every visual that can now be produced upon running `main()` in `figs.py`

---

# Out of scope

- modifications to the backbone logic of the inference pipeline. in particular, no plotting should be done during the streaming inference itself, only afterwards, even if it is flagged to run in the inference script. in particular, streaming inference is not needed since predicted matrices on only a small subset of the test set is needed for plots

- modifications to the training pipeline.

---

# Other references

Make sure you have read these files:

- `AGENTS.md` at root
- `examples/qmugs/AGENTS.md`

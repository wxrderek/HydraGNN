# Perlmutter to Duke Compute Cluster Environment Transfer

## Objective

Due to upcoming Perlmutter downtime, we are moving HydraGNN work to the Duke Compute Cluster (DCC).
The most difficult aspect is rebuilding the conda environment in a compatible manner with the DCC, which is
significantly less sophisticated a system than Perlmutter.

Follow the following procedure, documenting useful things in `references/dcc_info/perlmutter_to_dcc_transfer.md`:

1. comprehensively learn the setup of the Perlmutter environment
    - read `examples/qmugs/README.md`, which contains instructions on how the Perlmutter environment was built. however, these are NOT all the commands that ended up running to build the current environment on Perlmutter; the others are not documented.
    - read `installation_DOE_supercomputers/module_loads_perlmutter.sh`, which contains the command to load Perlmutter modules. find and carefully read real info on Perlmutter module setup in `references/perlmutter_info/modules/`; note the `module-list-postload.txt` file is output AFTER the module loading command has been run.
    - read `installation_DOE_supercomputers/module_loads_perlmutter.sh/hydragnn_installation_bash_script_perlmutter.sh`, which is the script run to set up the environment on Perlmutter. find and carefully read real info on the current Perlmutter environment in `references/perlmutter_info/hydragnn_venv/`.

2. learn the modules available on DCC
    - read `references/dcc_info/modules/module-avail.txt`, which contains all available modules.
    - run other commands as necessary to find out more; save the output from these in seperate files in `references/dcc_info/modules/` with descriptive names.

3. learn the dependencies by reading the following:
    - `requirements-base.txt`
    - `requirements-deepspeed.txt`
    - `requirements-dev.txt`
    - `requirements-optional.txt`
    - `requirements-pyg.txt`
    - `requirements-torch.txt`
    - `examples/qmugs/requirements.txt`
    note for our use case, we only worry about compatibility with tasks in `examples/qmugs/`, so DeepSpeed, DeepHyper, ADIOS2, and potentially other dependencies are not strictly needed.

4. determine which dependencies are completely incompatible and which would be inconvenient to install; output your determination in `references/dcc_info/perlmutter_to_dcc_transfer.md`.

5. write the module loading script in `installation_other/module_loads_dcc.sh`, following the style and usage of `installation_DOE_supercomputers/module_loads_perlmutter.sh`, and usage patterns in `examples/qmugs/README.md`.

6. write the conda environment installation script in `installation_other/hydragnn_installation_bash_script_dcc.sh`, following the style and usage of `installation_DOE_supercomputers/module_loads_perlmutter.sh/hydragnn_installation_bash_script_perlmutter.sh`, and usage patterns in `examples/qmugs/README.md`.

7. document your choices in `references/dcc_info/perlmutter_to_dcc_transfer.md`, including risks.

Note you should NOT run any of the task-specific shell scripts or Python scripts in `examples/qmugs/`. 
You may only run the installation scripts you've written to test things.

If I choose to let you run actual task-specific scripts to test things, I will specify this in the prompt.


## Files to edit

- `installation_other/module_loads_dcc.sh`
- `installation_other/hydragnn_installation_bash_script_dcc.sh`
- `references/dcc_info/perlmutter_to_dcc_transfer.md`

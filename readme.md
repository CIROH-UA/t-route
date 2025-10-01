# T-Route - Tree-Based Channel Routing 

**Fast, flexible, modular channel routing for the National Water Model and beyond**:  

T-Route, a dynamic channel routing model, offers a comprehensive solution for river network routing problems. It is designed to handle 1-D channel routing challenges in vector-based river network data, such as the USGS's NHDPlus High Resolution dataset, and OGC WaterML 2.0 Surface Hydrology Features (HY_Features) data model used in NextGen framework.

Provided a series lateral inflows for each node in a channel network, T-Route computes the resulting streamflows. T-Route requires that all routing computations srictly obey an upstream-to-downstream ordering. Such ordering facilitates the heterogenous application of routing models in a single river network. For example, hydrologic models - such as Muskingum Cunge - may be more appropriate for low-order headwater streams where backwater conditions have minimal impact on flooding. In contrast, T-Route empowers users to apply computationally intensive hydraulic models, such as the diffusive wave approximation of St. Venant equations, for high-order streams and rivers, where backwater flooding is a significant concern. This flexibility enables users to allocate computational resources precisely where they are needed most, optimizing efficiency.

Expanding its capabilities, T-Route now supports the OGC WaterML 2.0 Surface Hydrology Features (HY_Features) data model, facilitating the management of complex acyclic network connectivity. HY_Features data model provides users with a choice of routing solutions, including Muskingum-Cunge, Diffusive Wave, or Dynamic Wave.

**Key Features of T-Route:**
- Adaptable to multiple network formulations.
- Supports a range of routing solutions.
- Redesigned core operations accessible via standard BMI functions.
- Modular design for independent reservoir and data assimilation modules.
- Capable of disaggregating routing across large spatial domains.
- Facilitates heterogenous application of routing models in a single network.

## Routing Model Comparisons
| Feature | Muskingum-Cunge (MC) | Diffusive Wave |
|---------|----------------------|----------------|
| Domain | Individual stream segment | Independent sub-network |
| Computing Method | Parallel on CONUS using junction order and short-time step | Serial on a sub-network using junction order |
| Routing Direction | Upstream to downstream | Both directions |
| Numerical Scheme | [One-dimensional explicit scheme](https://ral.ucar.edu/sites/default/files/public/WRF-HydroV5TechnicalDescription.pdf) | [Implicit Crank-Nicolson scheme](https://onlinelibrary.wiley.com/doi/10.1111/1752-1688.13080) |

T-Route's flexible design is ideal for NOAA's National Water Model (NWM) 3.0, but its utility extends to various research and practical applications. While the code is currently bespoke for NWM, efforts are ongoing to generalize the core utilities, making T-Route compatible with any standardized network and forcing data.

## General Scheme:
The figure below illustrates the workflow for executing T-Route via two interfaces: the Command Line Interface (CLI) and the Basic Model Interface (BMI). When using the CLI, the user can select from two river network representations: NHDNetwork or HYFeatures. In contrast, the BMI exclusively supports HYFeatures. For the routing method, users have the option to apply either the Muskingum-Cunge method or the Diffusive Wave method.

<img src=https://raw.githubusercontent.com/NOAA-OWP/T-Route/master/doc/images/scheme.png height=400>

## Project Overview:
- **Technology Stack**: Combines Python with Fortran for core routing model engines. The river network pre-processor, river network traversal framework, and time series data model are all written in Python. The routing model engines (e.g. Muskingum-Cunge and diffusive wave) are primarily written in Fortran, though we can imagine future additions to the engine suite being writted in any number of languages that can be packaged as Python extensions.
- **Current Status**: Focused on integration with NWM 3.0.
- **Demonstrations**: The `/test` directory includes a T-Route demonstration on the Lower Colorado River, TX, showcasing capabilities like streamflow data assimilation and diffusive wave routing.

## Mission Alignment
T-Route development is rigorously aligned with and guided by the NOAA Office of Water Prediction mission: *Collaboratively research, develop and deliver timely and consistent, state-of-the-science national hydrologic analyses, forecast information, data, guidance and equitable decision-support services to inform essential emergency management and water resources decisions across all time scales.*

## Structure of Folders:
- `src/kernel/`: Fortran modules.
- `src/troute-config/`: Configuration parser for T-Route specific file.
- `src/troute-network/`: Manages network, data assimilation, and routing types.
- `src/troute-nwm/`: Coordinates T-Route’s modules.
- `src/troute-routing/`: Handles flow segment and reservoir routing modules.

## Summary:
T-Route represents streamflow channel routing and reservoir routing, assimilating data on vector-based channel networks. It fits into a broader framework where it interacts with land surface models, Forcing Engines, and coastal models, each playing a pivotal role in hydrologic forecasting and analysis.

## Installation

### T-Route Setup Instructions for Mac Users
Please follow our [macOS installation guide](mac_installation.md).

### T-Route Setup Instructions and Troubleshooting Guide for Linux and Windows Subsystem for Linux (WSL) Users

**Note**: The following instructions are for setting up T-Route on a Linux environment (standalone, no MPI). If you are using Windows, please install WSL (Windows Subsystem for Linux) before proceeding.

0. **Install WSL if on Windows:**
See the [WSL instructions](https://learn.microsoft.com/en-us/windows/wsl/install).

1. **Set up system requirements:**
```shell
sudo apt update
sudo apt install python3-pip
```
     
2. **Install Python 3.10:**
Python 3.10 is required for T-Route. If you attempt to build T-Route with a different Python version, it will not work.
- Install required build dependencies
```shell
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev curl
```
- Install Python 3.10 release
```shell
wget https://www.python.org/ftp/python/3.10.18/Python-3.10.18.tar.xz
```
- Build Python 3.10. Here, `sudo make altinstall` installs Python 3.10 as an alternate Python version, instead of replacing your default Python version.
```shell
tar -xf Python-3.10.18.tar.xz
cd Python-3.10.18
./configure
sudo make altinstall
```
 
3. **Clone T-Route and create its environment:**
- Go to a folder of your choice and create a T-Route directory
   ```shell
   mkdir ~/troute1
   cd ~/troute1
   ```
- Clone a T-Route repository (the current main branch is used as an example):
   ```shell
   git clone https://github.com/CIROH-UA/t-route.git
   cd troute1
   ```
- Activate your virtual environment
   ```shell
   python3.10 -m venv name_of_venv
   source name_of_venv/bin/activate
   ```
- Install python packages per requirements file
   ```shell
   pip install -r requirements.txt
   ```

4. **Download & build NetCDF Fortran libraries from UCAR:**
- Go to a folder of your choice and download the source code:
   ```shell
   mkdir ~/netcdf-fortran
   cd ~/netcdf-fortran
   wget https://downloads.unidata.ucar.edu/netcdf-fortran/4.6.1/netcdf-fortran-4.6.1.tar.gz
   tar xvf netcdf-fortran-4.6.1.tar.gz
   cd netcdf-fortran-4.6.1/
   ```
- Install some prerequisites (Fortran compiler, build essentials, standard C-netCDF library):
   ```shell
   sudo apt install gfortran libnetcdf-dev
   ```
- Configure the fortran-netcdf libraries:
   ```shell
   ./configure
   ```
- There should be no error message, and the output log should end up with something like:
   ![image](https://github.com/user-attachments/assets/48268212-0b74-4f75-9d52-97f68e6c80d0)
   (Warnings about zstd support are okay.)
- Finally, install the libraries:
   ```shell
   sudo make install
   ```
- Output should be something like:
   ![image](https://github.com/user-attachments/assets/57e48501-18f4-4004-9b10-5a9245186e38)

5. **Build and test T-Route:**
- Go back to your T-Route folder:
   ```shell
   cd ~/troute1
   ```
- Here, you will have to do a little investigating. You will need to find the location of netcdf.mod and include it in your compiler script.
   ```shell
   find /usr/ -name netcdf.mod
   ```
- Define the path of the directory that includes netcdf.mod in the compiler.sh file in T-Route (before the `if [-z “NETCDF …” ]` statement): `export NETCDF="/path/of/dir/`. In many cases, this directory is `/usr/local/lib/`.
- Compile T-Route (may take a few minutes, depending on the machine):
   ```shell
   ./compiler.sh
   ```
- Set path to runtime netcdf-Fortran library. We also recommend including this in the .bashrc file or your equivalent so that you don't have to run this every time:
   ```shell
   export LD_LIBRARY_PATH=/usr/local/lib/
   ```
- Run one of the demo examples provided (this is a hybrid MC + diffusive routing example that should run within a few minutes at most):
   ```shell
   cd test/LowerColorado_TX
   python3 -m nwm_routing -f -V4 test_AnA_V4_NHD.yaml
   ```


### T-Route Setup Troubleshooting Guide

**Handle Permission Errors:**
- Use `sudo chmod 777 <path>` for "permission denied" errors (replace `<path>` with the relevant directory).
- If you are not allowed to execute a script, use `chmod +x /path/to/script.sh` to make the script executable.

By following these instructions, you should successfully install and set up T-Route on your Linux system. For any issues or questions, feel free to seek assistance or open an issue.


## Known issues

We are constantly looking to improve. Please see the Git Issues for additional information.

## Getting help

T-Route team and the technical maintainers of the repository for more questions are: 
dongha.kim@noaa.gov 
sean.horvath@noaa.gov
amin.torabi@noaa.gov
jurgen.zach@noaa.gov

## Getting involved

Among other things, we are working on preparing more robust I/O capability for testing and on improving the speed of the parallel tree traversal. We welcome your thoughts, recommendations, comments, and of course, PRs. 

Please feel free to fork the repository and let us know if you will be issuing a pull request. 
More instructions will eventually be documented in [CONTRIBUTING](contributing.md).


----

## Open source licensing info
1. [TERMS](TERMS.md)
2. [LICENSE](LICENSE)


----

## Credits and references

A great deal of credit is owed to Drs. Ehab Meslehe and Fred Ogden, and as well to the entire NCAR WRF-Hydro development team. Continued leadership and support from Dr. Trey Flowers.

----

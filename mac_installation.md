# T-Route Setup Instructions and Troubleshooting Guide for macOS Users

Written and tested by Quinn Lee and Sonam Lama

Contact qylee@ua.edu if you have any questions

1. **Install Homebrew:**
```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Set up venv-based virtual environment:**
- Update Homebrew
```zsh
brew update
```
- Go to a folder of your choice and create a T-Route directory
    ```zsh
    mkdir troute
    cd troute
    ```
- Install Python 3.10
Use an installer from https://www.python.org/downloads/macos/
- Create a virtual environment for T-Route (named here 'troute_env1'):
    ```zsh
    python3.10 -m venv troute_env1
    ```
- Activate your shiny new virtual environment:
    ```zsh
    source troute_env1/bin/activate
    ```
- Now, the command prompts in the Shell window should start with (troute_env1)

3. **Clone T-Route:**

   - Clone a T-Route repository (the current main branch is used as an example):
      ```zsh
      git clone --progress --single-branch --branch ngiab http://github.com/CIROH-UA/t-route.git
      cd t-route
      ```
   - Install python packages per requirements file
      ```zsh
      pip install -r requirements.txt
      ```

4. **Download & build netcdf fortran libraries from UCAR:**
   - Go to a folder of your choice and download the source code:
      ```zsh
      cd ..
      brew install wget
      wget https://downloads.unidata.ucar.edu/netcdf-fortran/4.6.1/netcdf-fortran-4.6.1.tar.gz
      ```
   - Unzip it:
      ```zsh
      tar xvf netcdf-fortran-4.6.1.tar.gz
      ```
   - Enter the directory:
      ```zsh
      cd netcdf-fortran-4.6.1/
      ```
   - Install some prerequisites (Fortran compiler, standard C-netcdf libraries):
      ```zsh
      brew install gcc
      brew install netcdf
      brew install netcdf-cxx
      ```
      GCC takes a notoriously long time to compile, so go for a walk or something while it compiles.
   - Configure the fortran-netcdf libraries:
      ```zsh
      export CPPFLAGS="-I$(brew --cellar gcc)/$(brew list --versions gcc | tr ' ' '\n' | tail -1)/gcc/include -I$(dirname $(sudo find /opt/homebrew -name 'netcdf.h' | head -n 1))"
      export LDFLAGS="-L$(dirname $(sudo find /opt/homebrew -name 'libgfortran*.dylib' | head -n 1)) -L$(dirname $(sudo find /opt/homebrew -name 'libnetcdf*.dylib' | head -n 1))"
      export HDF5_PLUGIN_PATH=$(nc-config --plugindir)
      ./configure
      ```
   - The output log should end up with something like:
     ![image](https://github.com/user-attachments/assets/48268212-0b74-4f75-9d52-97f68e6c80d0)
   - Finally, install the libraries:
      ```zsh
      sudo make install
      ```
   - Output should be something like:
      ![image](https://github.com/user-attachments/assets/57e48501-18f4-4004-9b10-5a9245186e38)

5. **Build and test T-Route:**
   - Go to your T-Route folder:
      ```zsh
      cd ../t-route
      ```
   - Compile T-Route (may take a few minutes, depending on the machine):
      ```zsh
      ./compiler_mac.sh
      ```
   - Run one of the demo examples provided:
      ```zsh
      cd test/LowerColorado_TX
      python3 -m nwm_routing -f -V4 test_AnA_V4_NHD.yaml
      ```
   - The latter is a hybrid (MC + diffusive) routing example that should run within a few minutes at most
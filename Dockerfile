FROM rockylinux:9.2 AS rocky-base
RUN yum install -y epel-release
RUN yum install -y netcdf netcdf-fortran netcdf-fortran-devel netcdf-mpich

RUN yum install -y git cmake python python-devel pip
ENV FC=gfortran NETCDF=/usr/lib64/gfortran/modules/

WORKDIR "/t-route/"

RUN pip3 install uv && uv venv
ENV PATH="/t-route/.venv/bin:$PATH"

COPY . .

RUN uv pip install -r requirements.txt

# disable everything except the kernel builds
RUN sed -i 's/build_[a-z]*=/#&/' compiler.sh

RUN ./compiler.sh no-e

# install / build using UV because it's so much faster
# no build isolation needed because of cython namespace issues
RUN uv pip install --config-setting='--build-option=--use-cython' src/troute-network/
RUN uv pip install --no-build-isolation --config-setting='--build-option=--use-cython' src/troute-routing/
RUN uv pip install --no-build-isolation src/troute-config/
RUN uv pip install --no-build-isolation src/troute-nwm/

# increase max open files soft limit
RUN ulimit -n 10000
ENTRYPOINT ["/t-route/.venv/bin/python", "-m", "nwm_routing"]

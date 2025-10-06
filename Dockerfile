FROM rockylinux:9.2 AS rocky-base
RUN yum install -y epel-release
RUN yum install -y netcdf netcdf-fortran netcdf-fortran-devel netcdf-mpich

RUN yum install -y git cmake python python-devel pip
ENV FC=gfortran NETCDF=/usr/lib64/gfortran/modules/

ENV PATH="/root/.cargo/bin:${PATH}"
ENV UV_INSTALL_DIR=/root/.cargo/bin
ENV UV_COMPILE_BYTECODE=1
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv self update

WORKDIR "/t-route/"

ENV PATH="/t-route/.venv/bin:$PATH"

COPY ./requirements.txt .

RUN uv venv -p 3.9
RUN uv pip install -r requirements.txt
RUN uv pip install build wheel

COPY . .

# disable everything except the kernel builds
RUN sed -i 's/build_[a-z]*=/#&/' compiler.sh

RUN ./compiler.sh no-e

# install / build using UV because it's so much faster
# no build isolation needed because of cython namespace issues
RUN uv pip install --no-build-isolation --config-setting='--build-option=--use-cython' --editable src/troute-network/ --config-setting='editable_mode=compat'
RUN uv pip install --no-build-isolation --config-setting='--build-option=--use-cython' --editable src/troute-routing/ --config-setting='editable_mode=compat'
RUN uv pip install --no-build-isolation --editable src/troute-config/
RUN uv pip install --no-build-isolation --editable src/troute-nwm/
# increase max open files soft limit
RUN ulimit -n 10000
ENTRYPOINT ["/t-route/.venv/bin/python", "-m", "nwm_routing"]

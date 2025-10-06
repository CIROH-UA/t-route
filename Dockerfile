FROM troute

WORKDIR "/t-route/"

COPY . .

# install / build using UV because it's so much faster
# no build isolation needed because of cython namespace issues
RUN uv pip install --config-setting='--build-option=--use-cython' src/troute-network/
# RUN uv pip install --no-build-isolation --config-setting='--build-option=--use-cython' src/troute-routing/
# RUN uv pip install --no-build-isolation src/troute-config/
# RUN uv pip install --no-build-isolation src/troute-nwm/

# increase max open files soft limit
RUN ulimit -n 10000
ENTRYPOINT ["/t-route/.venv/bin/python", "-m", "nwm_routing"]

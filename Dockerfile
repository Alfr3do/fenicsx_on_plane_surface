FROM dolfinx/dolfinx:v0.11.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    nano \
    && rm -rf /var/lib/apt/lists/*
# Install Python packages for PyTorch & plotting
RUN pip install --no-cache-dir \
    torch \
    matplotlib \
    scipy \
    h5py \
    pyyaml \
    hydra-core \
    tqdm \
    mpi4py \
    pyvista \
    pyproj \
    meshio

COPY . /app
RUN chmod +x /app/exec.sh
#ENTRYPOINT ["mpirun", "-n", "2", "python3", "ns_code2.py"]
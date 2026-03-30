{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # Needed for CUDA/NVIDIA
          config.cudaSupport = true;
        };
        python = pkgs.python3.override {
          packageOverrides = py-self: py-super: {
            jax = py-super.jax.overridePythonAttrs (old: {
              doCheck = false;
              pytestCheckPhase = "true";
              checkPhase = "true";
              installCheckPhase = "true";
            });
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python.withPackages (ps: with ps; [
              jax
              jaxlib-bin
              equinox
              optax
              numpy
              matplotlib
              pytest
              mypy
              jaxtyping
              typeguard
              stickytape
              pygame
            ]))
            pyright
            # NVIDIA / CUDA dependencies
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
            cudaPackages.cuda_nvcc
            linuxPackages.nvidia_x11
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.linuxPackages.nvidia_x11
            ]}:$LD_LIBRARY_PATH
            export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
          '';
        };
      });
}

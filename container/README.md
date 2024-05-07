# Installation

This repository contains the necessary files to build a container locally using either Docker or Apptainer. You can choose depending on which software you have available on your system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building with Docker](#building-with-docker)
- [Building with Apptainer](#building-with-apptainer)

## Prerequisites

- Docker installed for Docker build, or
- Apptainer installed for Apptainer build
- Conda installed for Conda environment setup
- Clone the RydbergGPT repository to your local machine

## Installation

If you prefer to run the code outside of a container, you can set up a Conda environment using the provided `environment.yml` file.

1. **Navigate to the RydbergGPT repository directory**

   ```bash
   cd /path/to/RydbergGPT
   ```

2. **Create the Conda Environment**

   Run the following command to create a Conda environment based on the `environment.yml` file:

   ```bash
   conda env create -f container/environment.yml
   ```

## Building with Docker

1. **Navigate to the `container` directory in the RydbergGPT repository**

   ```bash
   cd /path/to/RydbergGPT/container
   ```

2. **Build the Docker Image**

   ```bash
   docker build -t pytorch_bundle:latest .
   ```

## Building with Apptainer

1. **Navigate to the `container` directory in the RydbergGPT repository**

   ```bash
   cd /path/to/RydbergGPT/container
   ```

2. **Build the Apptainer Image**

   ```bash
   apptainer build my_container.sif pytorch_recipe.def
   ```

## Install RydbergGPT

Once you have built the environment or container, you can install the RydbergGPT package in developer mode by running the following command:

```bash
pip install -e .
```

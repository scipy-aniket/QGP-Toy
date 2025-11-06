# Glauber Model Monte Carlo Simulation

A Monte Carlo implementation of the Glauber model for simulating heavy-ion collisions (Pb-Pb at 208 nucleons).

## Overview

This code simulates nucleus-nucleus collisions using the Glauber model to calculate:
- Number of participants (N_part)
- Number of binary collisions (N_coll)
- Spatial eccentricity coefficients (ε₂, ε₃)
- Impact parameter distributions

## Requirements

- Julia (tested on 1.x)
- Required packages:
  ```julia
  using Plots
  using Random
  using Distributions
  using SpecialFunctions
  using LaTeXStrings
  using Statistics
  ```

## Installation

Install required packages:
```julia
using Pkg
Pkg.add(["Plots", "Distributions", "SpecialFunctions", "LaTeXStrings"])
```

## Usage

Simply run the script:
```julia
julia glauber_simulation.jl
```

The simulation will:
1. Generate Pb-208 nuclei using Woods-Saxon density distribution
2. Run 1,000,000 collision events
3. Calculate collision observables and eccentricities
4. Generate 15 output figures

## Key Parameters

- **Mass number (A)**: 208 (Pb)
- **Nuclear radius**: R = 1.07 × A^(1/3) fm
- **Skin depth**: a = 0.54 fm
- **Inelastic cross section**: σ_NN = 6.5 fm²
- **Max impact parameter**: b_max = 20 fm

## Output

The code generates 15 figures:
- Woods-Saxon distribution
- 3D nucleus visualization
- Collision geometry (3D and transverse views)
- N_part and N_coll distributions
- Eccentricity vs impact parameter/N_part
- Participant plane angle distributions

All figures are saved as PNG files (fig1_*.png through fig15_*.png).

## Physics

Based on the optical Glauber model with:
- Woods-Saxon nuclear density profile
- Geometric collision criterion (d < √(σ_NN/π))
- r²-weighted eccentricity calculation following Alver & Roland (2010)

using Plots
using Random
using Distributions
using SpecialFunctions
using LaTeXStrings
using Statistics  # Add this at the top of the file if not already present

# Set random seed for reproducibility
Random.seed!(42)

# ============================================
# 1. Woods-Saxon Density Distribution
# ============================================

# Parameters for Pb-208
const A = 208  # Mass number
const R = 1.07 * A^(1/3)  # Nuclear radius in fm
const a = 0.54  # Skin depth in fm
const ρ_0 = 0.16  # fm^-3 (approximate)

"""
Woods-Saxon density distribution
"""
function woods_saxon_density(r)
    return ρ_0 / (1 + exp((r - R) / a))
end

"""
Probability density for sampling: ρ(r) * r²
"""
function woods_saxon_prob(r)
    return woods_saxon_density(r) * r^2
end

"""
Sample radius from Woods-Saxon distribution using rejection sampling
"""
function sample_radius(rmax=15.0)
    # Find maximum of ρ(r) * r² for rejection sampling
    r_test = range(0, rmax, length=1000)
    prob_max = maximum(woods_saxon_prob.(r_test))
    
    while true
        r = rand() * rmax
        prob = woods_saxon_prob(r)
        if rand() * prob_max < prob
            return r
        end
    end
end

"""
Sample uniformly on sphere surface and combine with radius
"""
function sample_nucleon_position()
    r = sample_radius()
    
    # Sample angles uniformly
    cos_theta = 2 * rand() - 1  # Uniform in [-1, 1]
    theta = acos(cos_theta)
    phi = 2π * rand()
    
    # Convert to Cartesian coordinates
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    
    return [x, y, z]
end

# Plot Woods-Saxon distribution
r_range = range(0, 15, length=500)
ρ_vals = woods_saxon_density.(r_range)
prob_vals = woods_saxon_prob.(r_range)

# Normalize the Woods-Saxon density
ρ_normalized = ρ_vals ./ maximum(ρ_vals)
prob_normalized = prob_vals ./ maximum(prob_vals)

p1 = plot(r_range, ρ_normalized, label=L"\rho(r) \, \mathrm{(normalized)}", 
          xlabel="r (fm)", ylabel="Normalized Density", 
          title="Woods-Saxon Distribution for Pb-208",
          linewidth=2, legend=:topright; dpi = 300)
plot!(p1, r_range, prob_normalized, 
      label=L"r^2 \rho(r) \, \mathrm{(normalized)}", linewidth=2)

display(p1)
savefig(p1, "fig1_woods_saxon.png")

# ============================================
# 2. Generate Nucleus (208 nucleons)
# ============================================

"""
Generate a nucleus with A nucleons
"""
function generate_nucleus(A=208)
    nucleons = [sample_nucleon_position() for _ in 1:A]
    return hcat(nucleons...)'  # Return as matrix (A × 3)
end

println("\nGenerating Pb nucleus...")
nucleus_A = generate_nucleus(A)

# 3D visualization of one nucleus
println("Creating 3D visualization of nucleus...")
p2 = scatter(nucleus_A[:, 1], nucleus_A[:, 2], nucleus_A[:, 3],
             markersize=3, alpha=0.6, color=:blue,
             xlabel="x (fm)", ylabel="y (fm)", zlabel="z (fm)",
             title="Pb Nucleus (208 nucleons) - 3D View",
             label="Nucleons", camera=(30, 30),
             aspect_ratio=:equal; dpi = 300)

display(p2)
savefig(p2, "fig2_nucleus_3d.png")

# Transverse view of nucleus (x-y plane)
println("Creating transverse view of nucleus...")
p2_transverse = scatter(nucleus_A[:, 1], nucleus_A[:, 2],
                        markersize=4, alpha=0.6, color=:blue,
                        xlabel="x (fm)", ylabel="y (fm)",
                        title="Pb Nucleus (208 nucleons) - Transverse Plane",
                        label="Nucleons", aspect_ratio=:equal,
                        markerstrokewidth=0; dpi = 300)
# Add circle showing nuclear radius
theta_circle = range(0, 2π, length=100)
plot!(p2_transverse, R.*cos.(theta_circle), R.*sin.(theta_circle),
      linewidth=2, color=:black, linestyle=:dash,
      label="Nuclear radius R")

display(p2_transverse)
savefig(p2_transverse, "fig2_nucleus_transverse.png")

# ============================================
# 3. Impact Parameter Sampling
# ============================================

"""
Sample impact parameter from distribution P(b) ∝ b
"""
function sample_impact_parameter(bmax=20.0)
    # P(b) ∝ b means CDF ∝ b²
    # So b = bmax * sqrt(rand())
    return bmax * sqrt(rand())
end

# ============================================
# 4. Collision Detection
# ============================================

const σ_NN_inel = 6.5  # fm (total inelastic nucleon-nucleon cross section)
const d_NN = sqrt(σ_NN_inel / π)  # Effective nucleon diameter

"""
Check if two nucleons collide based on their transverse separation
"""
function nucleons_collide(pos_A, pos_B)
    # Calculate transverse distance (x-y plane)
    dx = pos_A[1] - pos_B[1]
    dy = pos_A[2] - pos_B[2]
    d_transverse = sqrt(dx^2 + dy^2)
    
    return d_transverse < d_NN
end

"""
Simulate a single nucleus-nucleus collision
Returns: (N_part, N_coll, participating_A, participating_B)
"""
function simulate_collision(nucleus_A, nucleus_B, b)
    # Shift nucleus B by impact parameter b in x-direction
    nucleus_B_shifted = copy(nucleus_B)
    nucleus_B_shifted[:, 1] .+= b
    
    # Track participants and collisions
    n_nucleons_A = size(nucleus_A, 1)
    n_nucleons_B = size(nucleus_B, 1)
    participants_A = falses(n_nucleons_A)
    participants_B = falses(n_nucleons_B)
    N_coll = 0
    
    # Check all nucleon pairs
    for i in 1:n_nucleons_A
        for j in 1:n_nucleons_B
            if nucleons_collide(view(nucleus_A, i, :), view(nucleus_B_shifted, j, :))
                participants_A[i] = true
                participants_B[j] = true
                N_coll += 1
            end
        end
    end
    
    N_part = sum(participants_A) + sum(participants_B)
    
    return N_part, N_coll, participants_A, participants_B, nucleus_B_shifted
end

# Visualize a collision
nucleus_B = generate_nucleus(A)
b_sample = 5.0  # fm, peripheral collision for visualization

N_part, N_coll, part_A, part_B, nucleus_B_shifted = simulate_collision(nucleus_A, nucleus_B, b_sample)

println("Sample collision: b = $b_sample fm")
println("  N_part = $N_part")
println("  N_coll = $N_coll")

# 3D visualization of collision
p3 = scatter(nucleus_A[.!part_A, 1], nucleus_A[.!part_A, 2], nucleus_A[.!part_A, 3],
             markersize=3, alpha=0.3, color=:lightblue,
             xlabel="x (fm)", ylabel="y (fm)", zlabel="z (fm)",
             title="Pb+Pb Collision (b = $b_sample fm) - 3D View",
             label="Spectators A", camera=(30, 30); dpi = 300)
scatter!(p3, nucleus_A[part_A, 1], nucleus_A[part_A, 2], nucleus_A[part_A, 3],
         markersize=4, alpha=0.8, color=:darkblue,
         label="Participants A")
scatter!(p3, nucleus_B_shifted[.!part_B, 1], nucleus_B_shifted[.!part_B, 2], nucleus_B_shifted[.!part_B, 3],
         markersize=3, alpha=0.3, color=:lightcoral,
         label="Spectators B")
scatter!(p3, nucleus_B_shifted[part_B, 1], nucleus_B_shifted[part_B, 2], nucleus_B_shifted[part_B, 3],
         markersize=4, alpha=0.8, color=:darkred,
         label="Participants B")

display(p3)
savefig(p3, "fig3_collision_3d.png")

# Transverse view of collision (x-y plane)
println("Creating transverse view of collision...")
p3_transverse = scatter(nucleus_A[.!part_A, 1], nucleus_A[.!part_A, 2],
                        markersize=4, alpha=0.3, color=:lightblue,
                        xlabel="x (fm)", ylabel="y (fm)",
                        title="Pb+Pb Collision (b = $b_sample fm) - Transverse Plane",
                        label="Spectators A", aspect_ratio=:equal,
                        markerstrokewidth=0; dpi = 300)
scatter!(p3_transverse, nucleus_A[part_A, 1], nucleus_A[part_A, 2],
         markersize=5, alpha=0.8, color=:darkblue,
         label="Participants A", markerstrokewidth=0)
scatter!(p3_transverse, nucleus_B_shifted[.!part_B, 1], nucleus_B_shifted[.!part_B, 2],
         markersize=4, alpha=0.3, color=:lightcoral,
         label="Spectators B", markerstrokewidth=0)
scatter!(p3_transverse, nucleus_B_shifted[part_B, 1], nucleus_B_shifted[part_B, 2],
         markersize=5, alpha=0.8, color=:darkred,
         label="Participants B", markerstrokewidth=0)

# Add circles showing nuclear radii
theta_circle = range(0, 2π, length=100)
plot!(p3_transverse, R.*cos.(theta_circle), R.*sin.(theta_circle),
      linewidth=2, color=:blue, linestyle=:dash, label="Nucleus A boundary")
plot!(p3_transverse, b_sample .+ R.*cos.(theta_circle), R.*sin.(theta_circle),
      linewidth=2, color=:red, linestyle=:dash, label="Nucleus B boundary")

# Draw impact parameter line
plot!(p3_transverse, [0, b_sample], [0, 0], linewidth=3, color=:black,
      arrow=true, label="Impact parameter b")

display(p3_transverse)
savefig(p3_transverse, "fig3_collision_transverse.png")

# ============================================
# 5. Monte Carlo Simulation (10^5 events)
# ============================================

const N_events = 100000
const bmax = 20.0  # Maximum impact parameter (fm)

# Storage for results
N_coll_hist = Int[]
N_part_hist = Int[]
b_hist = Float64[]

# Progress indicator
progress_interval = N_events ÷ 20

println("\nStarting Monte Carlo simulation with $N_events events...")

for i in 1:N_events
    if i % progress_interval == 0
        println("  Progress: $(100*i÷N_events)%")
    end
    
    # Generate two nuclei
    nuc_A = generate_nucleus(A)
    nuc_B = generate_nucleus(A)
    
    # Sample impact parameter
    b = sample_impact_parameter(bmax)
    
    # Simulate collision
    local N_part, N_coll  # Declare as local to avoid ambiguity
    N_part, N_coll, _, _, _ = simulate_collision(nuc_A, nuc_B, b)
    
    # Only store events with at least one collision
    if N_coll > 0
        push!(N_coll_hist, N_coll)
        push!(N_part_hist, N_part)
        push!(b_hist, b)
    end
end

println("Total events with N_coll > 0: $(length(N_coll_hist))")

# ============================================
# 6. Plot Results (Figures 4 and 5)
# ============================================

# Figure 4: N_coll histogram
p4 = histogram(N_coll_hist, bins=100, 
               xlabel=L"N_{coll}", ylabel="Counts",
               title="Number of Binary Collisions",
               label="", yscale=:log10, linewidth=0,
               color=:blue, alpha=0.7; dpi = 300)

display(p4)
savefig(p4, "fig4_ncoll.png")

# Figure 5: N_part histogram
p5 = histogram(N_part_hist, bins=100,
               xlabel=L"N_{part}", ylabel="Counts",
               title="Number of Participants",
               label="", yscale=:log10, linewidth=0,
               color=:red, alpha=0.7; dpi = 300)

display(p5)
savefig(p5, "fig5_npart.png")

# Summary statistics
println("\n" * "="^50)
println("SUMMARY STATISTICS")
println("="^50)
println("N_coll:")
println("  Mean: $(round(mean(N_coll_hist), digits=2))")
println("  Max:  $(maximum(N_coll_hist))")
println("\nN_part:")
println("  Mean: $(round(mean(N_part_hist), digits=2))")
println("  Max:  $(maximum(N_part_hist))")
println("\nImpact parameter:")
println("  Mean: $(round(mean(b_hist), digits=2)) fm")
println("="^50)

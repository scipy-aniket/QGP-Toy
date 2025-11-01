using Plots
using Random
using Distributions
using SpecialFunctions
using LaTeXStrings
using Statistics

# Set random seed for reproducibility
Random.seed!(45)

# ============================================
# 1. Woods-Saxon Density Distribution
# ============================================

# Parameters for Pb-208
const A = 208  # Mass number
const R = 1.07 * A^(1/3)  # Nuclear radius in fm
const a = 0.54  # Skin depth in fm
const rho_0 = 0.16  # fm^-3 (approximate)

"""
Woods-Saxon density distribution
"""
function woods_saxon_density(r)
    return rho_0 / (1 + exp((r - R) / a))
end

"""
Probability density for sampling: rho(r) * r^2
"""
function woods_saxon_prob(r)
    return woods_saxon_density(r) * r^2
end

"""
Sample radius from Woods-Saxon distribution using rejection sampling
"""
function sample_radius(rmax=15.0)
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
rho_vals = woods_saxon_density.(r_range)
prob_vals = woods_saxon_prob.(r_range)

# Normalize the Woods-Saxon density
rho_normalized = rho_vals ./ maximum(rho_vals)
prob_normalized = prob_vals ./ maximum(prob_vals)

p1 = plot(r_range, rho_normalized, label=L"\rho(r) \, \mathrm{(normalized)}", 
          xlabel="r (fm)", ylabel="Normalized Density", 
          title="Woods-Saxon Distribution for Pb-208",
          linewidth=2, legend=:topright, dpi=300)
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

nucleus_A = generate_nucleus(A)

# 3D visualization of one nucleus
p2 = scatter(nucleus_A[:, 1], nucleus_A[:, 2], nucleus_A[:, 3],
             markersize=3, alpha=0.6, color=:blue,
             xlabel="x (fm)", ylabel="y (fm)", zlabel="z (fm)",
             title="Pb Nucleus (208 nucleons) - 3D View",
             label="Nucleons", camera=(30, 30),
             aspect_ratio=:equal, dpi=300)

display(p2)
savefig(p2, "fig2_nucleus_3d.png")

# Transverse view of nucleus (x-y plane)
p2_transverse = scatter(nucleus_A[:, 1], nucleus_A[:, 2],
                        markersize=4, alpha=0.6, color=:blue,
                        xlabel="x (fm)", ylabel="y (fm)",
                        title="Pb Nucleus (208 nucleons) - Transverse Plane",
                        label="Nucleons", aspect_ratio=:equal,
                        markerstrokewidth=0, dpi=300)
# Add circle showing nuclear radius
theta_circle = range(0, 2π, length=100)
plot!(p2_transverse, R.*cos.(theta_circle), R.*sin.(theta_circle),
      linewidth=2, color=:black, linestyle=:dash,
      label="Nuclear radius R")

display(p2_transverse)
savefig(p2_transverse, "fig2_nucleus_transverse.png")

# ============================================
# 3. Impact Parameter and Reaction Plane Sampling
# ============================================

"""
Sample impact parameter from distribution P(b) proportional to b
"""
function sample_impact_parameter(bmax=20.0)
    return bmax * sqrt(rand())
end

"""
Sample reaction plane angle uniformly
"""
function sample_reaction_plane_angle()
    return 2π * rand()
end

# ============================================
# 4. Collision Detection
# ============================================

const sigma_NN_inel = 6.5  # fm^2 (total inelastic nucleon-nucleon cross section)
const d_NN = sqrt(sigma_NN_inel / π)  # Effective nucleon diameter
const d_NN_sq = sigma_NN_inel / π  # Effective nucleon diameter squared

"""
Check if two nucleons collide based on their transverse separation
"""
function nucleons_collide(pos_A, pos_B)
    # Calculate transverse distance (x-y plane)
    dx = pos_A[1] - pos_B[1]
    dy = pos_A[2] - pos_B[2]
    # d_transverse = sqrt(dx^2 + dy^2) #sqrt is expensive
    d_transverse_sq = dx^2 + dy^2

    return d_transverse_sq < d_NN_sq
end

"""
Simulate a single nucleus-nucleus collision
Returns: (N_part, N_coll, participating_A, participating_B, nucleus_B_shifted)
"""
function simulate_collision(nucleus_A, nucleus_B, b, psi)
    # Shift nucleus B by impact parameter b at angle psi
    nucleus_B_shifted = copy(nucleus_B)
    dx = b * cos(psi)
    dy = b * sin(psi)
    nucleus_B_shifted[:, 1] .+= dx
    nucleus_B_shifted[:, 2] .+= dy
    
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
psi_sample = π/4  # 45 degrees

N_part, N_coll, part_A, part_B, nucleus_B_shifted = simulate_collision(nucleus_A, nucleus_B, b_sample, psi_sample)

println("\nSample collision: b = $b_sample fm, psi = $(round(psi_sample, digits=3)) rad")
println("  N_part = $N_part")
println("  N_coll = $N_coll")

# 3D visualization of collision
p3 = scatter(nucleus_A[.!part_A, 1], nucleus_A[.!part_A, 2], nucleus_A[.!part_A, 3],
             markersize=3, alpha=0.3, color=:lightblue,
             xlabel="x (fm)", ylabel="y (fm)", zlabel="z (fm)",
             title="Pb+Pb Collision (b = $b_sample fm) - 3D View",
             label="Spectators A", camera=(30, 30), dpi=300)
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
# Calculate shift for plotting
dx_plot = b_sample * cos(psi_sample)
dy_plot = b_sample * sin(psi_sample)

p3_transverse = scatter(nucleus_A[.!part_A, 1], nucleus_A[.!part_A, 2],
                        markersize=4, alpha=0.3, color=:lightblue,
                        xlabel="x (fm)", ylabel="y (fm)",
                        title="Pb+Pb Collision (b = $b_sample fm) - Transverse Plane",
                        label="Spectators A", aspect_ratio=:equal,
                        markerstrokewidth=0, dpi=300)
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
plot!(p3_transverse, R.*cos.(theta_circle), R.*sin.(theta_circle),
      linewidth=2, color=:blue, linestyle=:dash, label="Nucleus A boundary")
plot!(p3_transverse, dx_plot .+ R.*cos.(theta_circle), dy_plot .+ R.*sin.(theta_circle),
      linewidth=2, color=:red, linestyle=:dash, label="Nucleus B boundary")

# Draw impact parameter line
plot!(p3_transverse, [0, dx_plot], [0, dy_plot], linewidth=3, color=:black,
      arrow=true, label="Impact parameter b")

display(p3_transverse)
savefig(p3_transverse, "fig3_collision_transverse.png")

# ============================================
# 5. Eccentricity Calculations
# ============================================

"""
Calculate spatial eccentricity coefficients epsilon_n and participant plane angles Phi_n
Following Alver & Roland (2010) PhysRevC.81.054905

Key points:
1. Center around simple arithmetic mean of participant positions
2. Use r^2 weighting in eccentricity calculation
3. NO negative sign in epsilon_complex calculation
"""
function calculate_eccentricity(participants, n)
    if size(participants, 1) == 0
        return 0.0, 0.0
    end
    
    x = participants[:, 1]
    y = participants[:, 2]
    
    # Center around simple average (arithmetic mean)
    x_cm = mean(x)
    y_cm = mean(y)
    
    x_centered = x .- x_cm
    y_centered = y .- y_cm
    
    # Recalculate r and phi in centered frame
    r = sqrt.(x_centered.^2 .+ y_centered.^2)
    phi = atan.(y_centered, x_centered)
    
    # Calculate complex eccentricity (NO negative sign!)
    # See Alver & Roland Eq. (4) and (7)
    r_squared = r.^2
    numerator = sum(r_squared .* exp.(im .* n .* phi))
    denominator = sum(r_squared)
    
    eps_complex = numerator / denominator  # No negative sign
    
    # Extract magnitude and phase
    eps_n = abs(eps_complex)
    Phi_n = angle(eps_complex) / n
    
    return eps_n, Phi_n
end

"""
Calculate all eccentricity harmonics for an event
"""
function calculate_all_eccentricities(nucleus_A, nucleus_B_shifted, part_A, part_B, max_harmonic=3) # Calculate upto 3rd harmonic
    # Combine all participants (transverse plane only)
    participants_A = nucleus_A[part_A, 1:2]
    participants_B = nucleus_B_shifted[part_B, 1:2]
    all_participants = vcat(participants_A, participants_B)
    
    results = Dict{String, Float64}()
    
    for n in 2:max_harmonic
        eps_n, Phi_n = calculate_eccentricity(all_participants, n)
        results["eps$n"] = eps_n
        results["Phi$n"] = Phi_n
    end
    
    return results
end

# ============================================
# 6. Monte Carlo Simulation
# ============================================

const N_events = 1000000
const bmax = 20.0  # Maximum impact parameter (fm)

# Storage for results
N_coll_hist = Int[]
N_part_hist = Int[]
b_hist = Float64[]
psi_hist = Float64[]

# Eccentricity storage
eps2_hist = Float64[]
eps3_hist = Float64[]
# eps4_hist = Float64[]
# eps5_hist = Float64[]
Phi2_hist = Float64[]
Phi3_hist = Float64[]

# Progress indicator
progress_interval = N_events ÷ 20

println("Starting Monte Carlo simulation with $N_events events...")

for i in 1:N_events
    if i % progress_interval == 0
        println("  Progress: $(100*i÷N_events)%")
    end
    
    # Generate two nuclei
    nuc_A = generate_nucleus(A)
    nuc_B = generate_nucleus(A)
    
    # Sample impact parameter and reaction plane angle
    b = sample_impact_parameter(bmax)
    psi = sample_reaction_plane_angle()
    
    # Simulate collision
    N_part, N_coll, part_A, part_B, nuc_B_shifted = simulate_collision(nuc_A, nuc_B, b, psi)
    
    # Only store events with at least one collision
    if N_coll > 0
        push!(N_coll_hist, N_coll)
        push!(N_part_hist, N_part)
        push!(b_hist, b)
        push!(psi_hist, psi)
        
        # Calculate eccentricities if there are at least 2 participants
        if N_part >= 2
            ecc_results = calculate_all_eccentricities(nuc_A, nuc_B_shifted, part_A, part_B)
            
            push!(eps2_hist, ecc_results["eps2"])
            push!(eps3_hist, ecc_results["eps3"])
            # push!(eps4_hist, ecc_results["eps4"])
            # push!(eps5_hist, ecc_results["eps5"])
            push!(Phi2_hist, ecc_results["Phi2"])
            push!(Phi3_hist, ecc_results["Phi3"])
        end
    end
end

println("Total events with N_coll > 0: $(length(N_coll_hist))")
println("Total events with eccentricity data: $(length(eps2_hist))")

# ============================================
# 7. Summary Statistics
# ============================================
println("SUMMARY STATISTICS")
println("N_coll:")
println("  Mean: $(round(mean(N_coll_hist), digits=2))")
println("  Max:  $(maximum(N_coll_hist))")
println("\nN_part:")
println("  Mean: $(round(mean(N_part_hist), digits=2))")
println("  Max:  $(maximum(N_part_hist))")
println("\nImpact parameter:")
println("  Mean: $(round(mean(b_hist), digits=2)) fm")
println("\nEccentricities:")
println("  <eps_2> = $(round(mean(eps2_hist), digits=3))")
println("  <eps_3> = $(round(mean(eps3_hist), digits=3))")
# println("  <eps_4> = $(round(mean(eps4_hist), digits=3))")
# println("  <eps_5> = $(round(mean(eps5_hist), digits=3))")

# ============================================
# 8. Basic Distributions (Figs 4-5)
# ============================================

# Figure 4: N_coll histogram
p4 = histogram(N_coll_hist, bins=100, 
               xlabel=L"N_{coll}", ylabel="Counts",
               title="Number of Binary Collisions",
               label="", yscale=:log10, linewidth=0,
               color=:blue, alpha=0.7, dpi=300)

display(p4)
savefig(p4, "fig4_ncoll_distribution.png")

# Figure 5: N_part histogram
p5 = histogram(N_part_hist, bins=100,
               xlabel=L"N_{part}", ylabel="Counts",
               title="Number of Participants",
               label="", yscale=:log10, linewidth=0,
               color=:red, alpha=0.7, dpi=300)

display(p5)
savefig(p5, "fig5_npart_distribution.png")

# ============================================
# 9. N_part and N_coll vs Impact Parameter
# ============================================


# Bin the data by impact parameter
b_bin_edges = range(0, bmax, length=21)  # 20 bins
b_bin_centers_glauber = (b_bin_edges[1:end-1] .+ b_bin_edges[2:end]) ./ 2

# Initialize arrays for binned statistics
N_part_mean = zeros(length(b_bin_centers_glauber))
N_part_std = zeros(length(b_bin_centers_glauber))
N_coll_mean = zeros(length(b_bin_centers_glauber))
N_coll_std = zeros(length(b_bin_centers_glauber))

# Calculate mean and std for each bin
for (i, (b_low, b_high)) in enumerate(zip(b_bin_edges[1:end-1], b_bin_edges[2:end]))
    mask = (b_hist .>= b_low) .& (b_hist .< b_high)
    
    if sum(mask) > 0
        N_part_mean[i] = mean(N_part_hist[mask])
        N_part_std[i] = std(N_part_hist[mask])
        N_coll_mean[i] = mean(N_coll_hist[mask])
        N_coll_std[i] = std(N_coll_hist[mask])
    end
end

# Figure 6: N_part vs b
p6 = plot(b_bin_centers_glauber, N_part_mean, 
          ribbon=N_part_std,
          xlabel="Impact parameter b (fm)", 
          ylabel=L"N_{part}",
          title="Number of Participants vs Impact Parameter",
          label="Mean ± σ",
          linewidth=2, color=:blue,
          fillalpha=0.3, legend=:topright, dpi=300)

display(p6)
savefig(p6, "fig6_npart_vs_b.png")

# Figure 7: N_coll vs b
p7 = plot(b_bin_centers_glauber, N_coll_mean,
          ribbon=N_coll_std,
          xlabel="Impact parameter b (fm)",
          ylabel=L"N_{coll}",
          title="Number of Binary Collisions vs Impact Parameter",
          label="Mean ± σ",
          linewidth=2, color=:red,
          fillalpha=0.3, legend=:topright, dpi=300)

display(p7)
savefig(p7, "fig7_ncoll_vs_b.png")

# Figure 8: Combined plot (N_part and N_coll vs b)
p8 = plot(b_bin_centers_glauber, N_part_mean,
          ribbon=N_part_std,
          xlabel="Impact parameter b (fm)",
          ylabel="Number",
          title="Participants and Collisions vs Impact Parameter",
          label=L"N_{part}",
          linewidth=2, color=:blue, fillalpha=0.3, legend=:topright, dpi=300)
plot!(p8, b_bin_centers_glauber, N_coll_mean,
      ribbon=N_coll_std,
      label=L"N_{coll}",
      linewidth=2, color=:red, fillalpha=0.3)

display(p8)
savefig(p8, "fig8_npart_ncoll_vs_b.png")

# ============================================
# 10. 2D Histogram: N_part vs b
# ============================================

# Figure 9: 2D histogram
p9 = histogram2d(b_hist, N_part_hist,
                 bins=(50, 50),
                 xlabel="Impact parameter b (fm)",
                 ylabel=L"N_{part}",
                 title="Distribution of Participants vs Impact Parameter",
                 colorbar_title="Event Count",
                 colorbar_titlefontsize=8,
                 color=:plasma, dpi=300)

# Add annotations
plot!(p9, [0], [416], 
      marker=:star, markersize=10, color=:red, 
      label="Maximum (N_part = 416)", legend=:bottomleft)

display(p9)
savefig(p9, "fig9_npart_vs_b_2dhist.png")

# Calculate statistics
near_central_fraction = sum(N_part_hist .> 400) / length(N_part_hist) * 100
println("  Fraction of events with N_part > 400: $(round(near_central_fraction, digits=2))%")

b_central_10percent = 0.1 * bmax
central_fraction = sum(b_hist .< b_central_10percent) / length(b_hist) * 100
println("  Fraction of events with b < $(round(b_central_10percent, digits=2)) fm: $(round(central_fraction, digits=2))%")

# ============================================
# 11. Eccentricity Analysis Plots
# ============================================

# Calculate means for later use
mean_eps2 = mean(eps2_hist)
mean_eps3 = mean(eps3_hist)

# Calculate correlation coefficients
corr_eps2_b = cor(eps2_hist, b_hist)
corr_eps3_b = cor(eps3_hist, b_hist)

# Figure 10: eps_2 vs b
p10 = scatter(b_hist, eps2_hist, alpha=0.05, markersize=2,
             xlabel="Impact parameter b (fm)", ylabel=L"\varepsilon_2",
             title=L"\varepsilon_2" * " vs b",
             label="", color=:blue, markerstrokewidth=0, dpi=300)

# Binned average
b_bins = 0:1:20
eps2_b_binned = Float64[]
b_bin_centers = Float64[]
for i in 1:(length(b_bins)-1)
    mask = (b_hist .>= b_bins[i]) .& (b_hist .< b_bins[i+1])
    if sum(mask) > 10
        push!(eps2_b_binned, mean(eps2_hist[mask]))
        push!(b_bin_centers, (b_bins[i] + b_bins[i+1]) / 2)
    end
end
plot!(p10, b_bin_centers, eps2_b_binned, linewidth=3, color=:red,
      label="Binned avg (corr = $(round(corr_eps2_b, digits=3)))", marker=:circle, markersize=6)

display(p10)
savefig(p10, "fig10_eps2_vs_b.png")

# Figure 11: eps_3 vs b
p11 = scatter(b_hist, eps3_hist, alpha=0.05, markersize=2,
             xlabel="Impact parameter b (fm)", ylabel=L"\varepsilon_3",
             title=L"\varepsilon_3" * " vs b",
             label="", color=:red, markerstrokewidth=0, dpi=300)

# Binned average
eps3_b_binned = Float64[]
b_bin_centers_3 = Float64[]
for i in 1:(length(b_bins)-1)
    mask = (b_hist .>= b_bins[i]) .& (b_hist .< b_bins[i+1])
    if sum(mask) > 10
        push!(eps3_b_binned, mean(eps3_hist[mask]))
        push!(b_bin_centers_3, (b_bins[i] + b_bins[i+1]) / 2)
    end
end
plot!(p11, b_bin_centers_3, eps3_b_binned, linewidth=3, color=:darkred,
      label="Binned avg (corr = $(round(corr_eps3_b, digits=3)))", marker=:circle, markersize=6)

display(p11)
savefig(p11, "fig11_eps3_vs_b.png")

# ============================================
# 12. Eccentricity vs N_part (Scatter plots with binned averages)
# ============================================

# Match up the data
valid_indices = 1:min(length(N_part_hist), length(eps2_hist))
Npart_vals = N_part_hist[valid_indices]
eps2_vals = eps2_hist[valid_indices]
eps3_vals = eps3_hist[valid_indices]

# Calculate correlations
corr_eps2_Npart = cor(eps2_vals, Npart_vals)
corr_eps3_Npart = cor(eps3_vals, Npart_vals)

# Figure 12: eps_2 vs N_part (scatter with binned average)
p12 = scatter(Npart_vals, eps2_vals, alpha=0.05, markersize=2,
             xlabel=L"N_{part}", ylabel=L"\varepsilon_2",
             title=L"\varepsilon_2" * " vs N_part",
             label="", color=:blue, markerstrokewidth=0, dpi=300)

# Binned average
Npart_bins_scatter = 0:20:420
eps2_Npart_scatter = Float64[]
Npart_centers_scatter = Float64[]
for i in 1:(length(Npart_bins_scatter)-1)
    mask = (Npart_vals .>= Npart_bins_scatter[i]) .& (Npart_vals .< Npart_bins_scatter[i+1])
    if sum(mask) > 10
        push!(eps2_Npart_scatter, mean(eps2_vals[mask]))
        push!(Npart_centers_scatter, (Npart_bins_scatter[i] + Npart_bins_scatter[i+1]) / 2)
    end
end
plot!(p12, Npart_centers_scatter, eps2_Npart_scatter, linewidth=3, color=:red,
      label="Binned avg (corr = $(round(corr_eps2_Npart, digits=3)))", marker=:circle, markersize=6)

display(p12)
savefig(p12, "fig12_eps2_vs_npart.png")

# Figure 13: eps_3 vs N_part (scatter with binned average)
p13 = scatter(Npart_vals, eps3_vals, alpha=0.05, markersize=2,
             xlabel=L"N_{part}", ylabel=L"\varepsilon_3",
             title=L"\varepsilon_3" * " vs N_part",
             label="", color=:red, markerstrokewidth=0, dpi=300)

# Binned average
eps3_Npart_scatter = Float64[]
Npart_centers_scatter_3 = Float64[]
for i in 1:(length(Npart_bins_scatter)-1)
    mask = (Npart_vals .>= Npart_bins_scatter[i]) .& (Npart_vals .< Npart_bins_scatter[i+1])
    if sum(mask) > 10
        push!(eps3_Npart_scatter, mean(eps3_vals[mask]))
        push!(Npart_centers_scatter_3, (Npart_bins_scatter[i] + Npart_bins_scatter[i+1]) / 2)
    end
end
plot!(p13, Npart_centers_scatter_3, eps3_Npart_scatter, linewidth=3, color=:darkred,
      label="Binned avg (corr = $(round(corr_eps3_Npart, digits=3)))", marker=:circle, markersize=6)

display(p13)
savefig(p13, "fig13_eps3_vs_npart.png")

# Figure 14: Phi_2 and Phi_3 distributions
p14 = histogram(Phi2_hist, bins=50, normalize=:pdf,
                xlabel=L"\Phi_n", ylabel="Probability Density",
                title="Participant Plane Angles",
                label=L"\Phi_2", alpha=0.6, color=:blue, dpi=300)
histogram!(p14, Phi3_hist, bins=50, normalize=:pdf,
           label=L"\Phi_3", alpha=0.6, color=:red)

# Add expected uniform distribution
plot!(p14, [-π, π], [1/(2π), 1/(2π)], linewidth=2, color=:black,
      linestyle=:dash, label="Uniform")

display(p14)
savefig(p14, "fig14_participant_plane_angles.png")

# ============================================
# 15. N_coll vs b 2D Histogram
# ============================================

# Figure 15: 2D histogram for N_coll vs b
p15 = histogram2d(b_hist, N_coll_hist,
                 bins=(50, 50),
                 xlabel="Impact parameter b (fm)",
                 ylabel=L"N_{coll}",
                 title="Distribution of Binary Collisions vs Impact Parameter",
                 colorbar_title="Event Count",
                 colorbar_titlefontsize=8,
                 color=:plasma, dpi=300)

display(p15)
savefig(p15, "fig15_ncoll_vs_b_2dhist.png")

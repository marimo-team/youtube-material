# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "ai"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return

@app.cell
def _():
    # Define the range of points to allocate
    points_range = np.arange(0, 101)
    
    # Calculate damage for each combination of wits and finesse
    # Damage = Base Damage * (1 + Finesse Bonus) * (1 + Crit Chance * Crit Multiplier)
    # Crit Multiplier is 1 (base) + 1 (for double damage)
    
    base_damage = 100  # Assume a base damage of 100 for simplicity
    crit_multiplier = 2
    
    # Create a grid of wits and finesse points
    wits_points, finesse_points = np.meshgrid(points_range, points_range)
    
    # Calculate crit chance and finesse bonus
    crit_chance = wits_points * 0.01
    finesse_bonus = finesse_points * 0.05
    
    # Calculate total damage
    # Total Damage = Base Damage * (1 + Finesse Bonus) * (1 + Crit Chance * Crit Multiplier)
    total_damage = base_damage * (1 + finesse_bonus) * (1 + crit_chance * crit_multiplier)
    return

@app.cell
def _():
    # Plot the results
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    
    # Create a contour plot with iso lines
    contour = plt.contourf(wits_points, finesse_points, total_damage, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Total Damage")
    
    # Add iso lines
    iso_lines = plt.contour(wits_points, finesse_points, total_damage, levels=20, colors="black", linewidths=0.5)
    plt.clabel(iso_lines, inline=True, fontsize=8)
    
    # Highlight the optimal allocation line
    optimal_allocation = np.argmax(total_damage, axis=0)
    plt.plot(points_range, optimal_allocation, color="red", linewidth=2, label="Optimal Allocation")
    
    # Add labels and title
    plt.title("Optimal Allocation of Points between Wits and Finesse")
    plt.xlabel("Wits Points")
    plt.ylabel("Finesse Points")
    plt.legend()
    
    plt.gca()
    return

if __name__ == "__main__":
    app.run()

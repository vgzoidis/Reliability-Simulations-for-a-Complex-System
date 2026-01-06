import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

# Add config parameters
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    COMPONENTS_DATA as components_data,
    Tc, DT, N_SIMS,
    ensure_output_dir,
    print_header)

# Executes N simulations with MTTR for the system, calculating:
# - Experimental failure rate λ
# - Experimental reliability R(Tc)
def run_monte_carlo_simulations(component_name, mttf, duty_cycle, duration, dt, n_sims):
    failure_times = []
    failures_occurred = 0
    
    print(f"\nMTTF = {mttf} hours, Duty Cycle = {duty_cycle}, Tc = {duration} hours")
    print(f"{'='*70}")
    
    for i in range(n_sims):
        _, _, failure_time = simulate_component_failures(component_name, mttf, duty_cycle, duration, dt)
        
        if failure_time is not None:
            failure_times.append(failure_time)
            failures_occurred += 1
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_sims} simulations...")
    
    # Reliability R_system(Tc)
    reliability_experimental = (n_sims - failures_occurred) / n_sims
    
    # Theoretical Reliability
    lam_theoretical = 1.0 / mttf
    t_effective = duty_cycle * duration
    reliability_theoretical = np.exp(-lam_theoretical * t_effective)
    
    # Failure Rate λ
    # From mean time between faults
    if len(failure_times) > 0:
        mean_failure_time = np.mean(failure_times)
        effective_mttf = mean_failure_time / duty_cycle if duty_cycle > 0 else float('inf')
        lambda_experimental_method1 = 1.0 / effective_mttf if effective_mttf > 0 else 0
    else:
        lambda_experimental_method1 = 0
        mean_failure_time = None
    
    # From reliability R(Tc)
    if reliability_experimental > 0 and duty_cycle > 0:
        lambda_experimental_method2 = -np.log(reliability_experimental) / t_effective
    else:
        lambda_experimental_method2 = 0
    
    print_header(f"RESULTS for {component_name}:", "=", 70)
    print(f"Number of failures: {failures_occurred}/{n_sims}")
    print(f"\nReliability R(Tc={duration}):")
    print(f"Experimental:    R = {reliability_experimental:.6f}")
    print(f"Theoretical:     R = {reliability_theoretical:.6f}")
    print(f"Relative Error:  {abs(reliability_experimental - reliability_theoretical) / reliability_theoretical * 100:.2f}%")
    
    print(f"\nFault Rate λ:")
    print(f"Theoretical:        λ = {lam_theoretical:.6f} failures/hour")
    print(f"Experimental (MTTF): λ = {lambda_experimental_method1:.6f} failures/hour")
    print(f"Experimental (R):    λ = {lambda_experimental_method2:.6f} failures/hour")
    
    if mean_failure_time is not None:
        print(f"\nMean time to first failure: {mean_failure_time:.4f} hours")
        print(f"Effective MTTF (adjusted for DC): {effective_mttf:.4f} hours")
    
    return {
        'component': component_name,
        'failures': failures_occurred,
        'total_sims': n_sims,
        'reliability_exp': reliability_experimental,
        'reliability_theo': reliability_theoretical,
        'lambda_theo': lam_theoretical,
        'lambda_exp_method1': lambda_experimental_method1,
        'lambda_exp_method2': lambda_experimental_method2,
        'failure_times': failure_times,
        'mean_failure_time': mean_failure_time
    }

# Component states:
# - 2: Operational
# - 1: Non-operational due to duty cycle
# - 0: Under repair
def simulate_component_failures(component_name, mttf, duty_cycle, duration, dt):
    time_axis = np.arange(0, duration, dt)
    status_history = []
    failure_time = None
    
   
    lam = 1.0 / mttf   # Calculate λ parameter
    current_status = 1 # Operational initially
    
    for t in time_axis:
        if current_status == -1:
            status_history.append(-1)
        else:
            is_active = np.random.rand() < duty_cycle
            
            if is_active:
                prob_fail = 1 - np.exp(-lam * dt)
                
                if np.random.rand() < prob_fail:
                    current_status = -1
                    failure_time = t
                    status_history.append(-1)
                else:
                    current_status = 1
                    status_history.append(1)
            else:
                current_status = 0
                status_history.append(0)
    
    return time_axis, np.array(status_history), failure_time

def visualize_single_simulation(component_name, mttf, duty_cycle, duration, dt):
    time_axis, status_history, failure_time = simulate_component_failures(component_name, mttf, duty_cycle, duration, dt)
    
    plt.figure(figsize=(12, 4))
    
    colors = []
    for status in status_history:
        if status == 1:
            colors.append('green')  # Operational
        elif status == 0:
            colors.append('yellow') # Not operational
        else:
            colors.append('red')    # Fault
    
    plt.scatter(time_axis, status_history, c=colors, s=1, alpha=0.5)
    if failure_time is not None:
        plt.axvline(x=failure_time, color='red', linestyle='--', label=f'Failure at t={failure_time:.2f}h')
        plt.legend()
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.title(f'Failure Simulation - {component_name} (MTTF={mttf}h, DC={duty_cycle})', fontsize=14)
    plt.yticks([-1, 0, 1], ['Failure', 'Non-Operational', 'Operational'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'{component_name}_simulation_example.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved: {output_path}")

def create_summary_plots(results_list):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = [r['component'] for r in results_list]
    
    ax1 = axes[0, 0]
    rel_exp = [r['reliability_exp'] for r in results_list]
    rel_theo = [r['reliability_theo'] for r in results_list]
    x = np.arange(len(components))
    width = 0.35
    ax1.bar(x - width/2, rel_exp, width, label='Experimental', alpha=0.8)
    ax1.bar(x + width/2, rel_theo, width, label='Theoretical', alpha=0.8)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Reliability R(Tc)')
    ax1.set_title(f'Reliability at Tc={Tc} hours')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Compare failure rate
    ax2 = axes[0, 1]
    lam_exp = [r['lambda_exp_method2'] for r in results_list]
    lam_theo = [r['lambda_theo'] for r in results_list]
    ax2.bar(x - width/2, lam_exp, width, label='Experimental', alpha=0.8)
    ax2.bar(x + width/2, lam_theo, width, label='Theoretical', alpha=0.8)
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Failure Rate λ (failures/hour)')
    ax2.set_title('Failure Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Percentage of faults
    ax3 = axes[1, 0]
    failure_rates = [r['failures'] / r['total_sims'] * 100 for r in results_list]
    ax3.bar(components, failure_rates, alpha=0.8, color='coral')
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Percentage of Simulations with Failure (%)')
    ax3.set_title('Failure Percentage')
    ax3.grid(True, alpha=0.3)
    
    # Fault time distribution
    ax4 = axes[1, 1]
    if len(results_list[0]['failure_times']) > 0:
        ax4.hist(results_list[0]['failure_times'], bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Failure Time (hours)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Failure Time Distribution - {results_list[0]["component"]}')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'failure_simulation_summary.png')
    plt.savefig(output_path, dpi=150)

if __name__ == "__main__":
    OUTPUT_DIR = ensure_output_dir('part1')
    
    # Run simulation for all components
    all_results = []
    for comp_name, specs in components_data.items():
        results = run_monte_carlo_simulations(
            comp_name, 
            specs['MTTF'], 
            specs['DC'], 
            Tc, 
            DT, 
            N_SIMS
        )
        all_results.append(results)
        
        # Create graph for simulation example
        if comp_name in ['C1', 'C7']:
            visualize_single_simulation(comp_name, specs['MTTF'], specs['DC'], Tc, DT)
    
    # Create results summary graph
    create_summary_plots(all_results)
    print(f"Plots saved to folder: {OUTPUT_DIR}/")

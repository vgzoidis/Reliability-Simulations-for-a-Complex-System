import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

# Add config parameters
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    COMPONENTS_DATA as components_data,
    Ts, DT, N_SIMS,
    ensure_output_dir,
    print_header)

# Executes N simulations for the system, calculating:
# - Experimental failure rate λ
# - Experimental reliability R_system(Ts)
# - Experimental MTTF_system
def run_system_monte_carlo(components_db, duration, dt, n_sims):
    system_failure_times = []
    system_failures_count = 0

    # Samples for graph
    sample_time = None
    sample_history = None
    sample_components = None
    
    for i in range(n_sims):
        time_axis, system_history, comp_states, failure_time = simulate_system_reliability(components_db, duration, dt)
        
        # Save first results
        if i == 0:
            sample_time = time_axis
            sample_history = system_history
            sample_components = comp_states
        
        # Save the rest of the results
        if failure_time is not None:
            system_failure_times.append(failure_time)
            system_failures_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_sims} simulations...")
    
    # 1. Reliability R_system(Ts)
    reliability_system_exp = (n_sims - system_failures_count) / n_sims
    
    print(f"\nReliability R_system(Ts={duration}h):")
    print(f"Experimental:      R = {reliability_system_exp:.6f}")
    print(f"Number of failures: {system_failures_count}/{n_sims}")
    print(f"Success rate:      {reliability_system_exp * 100:.2f}%")
    
    # MTTF
    if len(system_failure_times) > 0:
        mttf_system_exp = np.mean(system_failure_times)
        mttf_std = np.std(system_failure_times)
        mttf_median = np.median(system_failure_times)
        
        print(f"\nMean Time To Failure:")
        print(f"Mean:            {mttf_system_exp:.4f} hours")
        print(f"Std deviation:   {mttf_std:.4f} hours")
        print(f"Median:          {mttf_median:.4f} hours")
        print(f"Minimum:         {min(system_failure_times):.4f} hours")
        print(f"Maximum:         {max(system_failure_times):.4f} hours")
    else:
        mttf_system_exp = float('inf')
        print(f"\nMean Time To Failure:")
        print(f"No failures observed in {n_sims} simulations!")
        print(f"MTTF > {duration} hours")
    
    # Failure Rate λ
    # From MTTF
    if mttf_system_exp != float('inf'):
        lambda_system_method1 = 1.0 / mttf_system_exp
    else:
        lambda_system_method1 = 0
    
    # From R(Ts)
    if reliability_system_exp > 0:
        lambda_system_method2 = -np.log(reliability_system_exp) / duration
    else:
        lambda_system_method2 = float('inf')
    
    print(f"\nFailure Rate λ:")
    print(f"Method 1 (MTTF):  λ = {lambda_system_method1:.6f} failures/hour")
    print(f"Method 2 (R):     λ = {lambda_system_method2:.6f} failures/hour")
    
    # Theoretical Results
    print_header("Theoretical Results", "=", 70)
    
    # For every component: R_i(t) = exp(-λ_i * DC_i * t)
    def component_reliability(mttf, dc, t):
        lam = 1.0 / mttf
        return np.exp(-lam * dc * t)
    
    R_C1 = component_reliability(components_db['C1']['MTTF'], components_db['C1']['DC'], duration)
    R_C2 = component_reliability(components_db['C2']['MTTF'], components_db['C2']['DC'], duration)
    R_C3 = component_reliability(components_db['C3']['MTTF'], components_db['C3']['DC'], duration)
    R_C4 = component_reliability(components_db['C4']['MTTF'], components_db['C4']['DC'], duration)
    R_C5 = component_reliability(components_db['C5']['MTTF'], components_db['C5']['DC'], duration)
    R_C6 = component_reliability(components_db['C6']['MTTF'], components_db['C6']['DC'], duration)
    R_C7 = component_reliability(components_db['C7']['MTTF'], components_db['C7']['DC'], duration)
    
    # Parallel block reliability:
    R_block2 = 1 - (1-R_C2)*(1-R_C3)*(1-R_C4)
    R_block3 = 1 - (1-R_C5)*(1-R_C6)
    
    # Overall Reliability:
    R_system_theo = R_C1 * R_block2 * R_block3 * R_C7
    
    print(f"\nComponent reliabilities at Ts={duration}h:")
    print(f"  R_C1 = {R_C1:.6f}")
    print(f"  R_C2 = {R_C2:.6f}, R_C3 = {R_C3:.6f}, R_C4 = {R_C4:.6f}")
    print(f"  R_C5 = {R_C5:.6f}, R_C6 = {R_C6:.6f}")
    print(f"  R_C7 = {R_C7:.6f}")
    
    print(f"\nBlock reliabilities:")
    print(f"  R_block2 (C2‖C3‖C4) = {R_block2:.6f}")
    print(f"  R_block3 (C5‖C6)    = {R_block3:.6f}")
    
    print(f"\nSystem reliability:")
    print(f"  Theoretical:   R_system = {R_system_theo:.6f}")
    print(f"  Experimental: R_system = {reliability_system_exp:.6f}")
    print(f"  Relative error: {abs(R_system_theo - reliability_system_exp) / R_system_theo * 100:.2f}%")
    
    if R_system_theo > 0:
        lambda_system_theo = -np.log(R_system_theo) / duration
        print(f"\n  Theoretical λ_system = {lambda_system_theo:.6f} failures/hour")
    
    return {
        'reliability_exp': reliability_system_exp,
        'reliability_theo': R_system_theo,
        'mttf_exp': mttf_system_exp,
        'lambda_method1': lambda_system_method1,
        'lambda_method2': lambda_system_method2,
        'failure_times': system_failure_times,
        'failures_count': system_failures_count,
        'total_sims': n_sims,
        'sample_time': sample_time,
        'sample_history': sample_history,
        'sample_components': sample_components
    }

# Component states:
# - 2: Operational
# - 1: Non-operational due to duty cycle
# - 0: Under repair
def simulate_system_reliability(components_db, duration, dt):
    time_axis = np.arange(0, duration, dt)
    
    # Initialize operational states
    component_states = {name: [] for name in components_db}
    current_status = {name: 1 for name in components_db}
    failed_components = {name: False for name in components_db}
    system_history = []
    system_failure_time = None
    system_failed = False
    
    for t in time_axis:
        for name, specs in components_db.items():
            if failed_components[name]: # Componend already failed
                current_status[name] = -1
            else: # Duty cyvle check
                is_active = np.random.rand() < specs['DC']
                
                if is_active: # Component is active, check for fault
                    lam = 1.0 / specs['MTTF']
                    prob_fail = 1 - np.exp(-lam * dt)
                    
                    if np.random.rand() < prob_fail:
                        current_status[name] = -1
                        failed_components[name] = True
                    else:
                        current_status[name] = 1
                else:
                    current_status[name] = 0
            
            component_states[name].append(current_status[name])
        
        # Calculate system status
        c1_ok = (current_status['C1'] != -1)
        c7_ok = (current_status['C7'] != -1)
        block2_ok = (current_status['C2'] != -1) or \
                    (current_status['C3'] != -1) or \
                    (current_status['C4'] != -1)
        block3_ok = (current_status['C5'] != -1) or \
                    (current_status['C6'] != -1)
        
        system_operational = c1_ok and block2_ok and block3_ok and c7_ok
        system_history.append(1 if system_operational else 0)
        
        if not system_operational and not system_failed:
            system_failure_time = t
            system_failed = True
    
    return time_axis, np.array(system_history), component_states, system_failure_time

def visualize_system_simulation(results):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    time = results['sample_time']
    history = results['sample_history']
    
    ax1.fill_between(time, 0, history, where=(history==1), color='green', alpha=0.3, label='Operational')
    ax1.fill_between(time, 0, history, where=(history==0), color='red', alpha=0.3, label='Failure')
    ax1.plot(time, history, 'k-', linewidth=0.5, alpha=0.5)
    
    failures = np.where(history == 0)[0]
    if len(failures) > 0:
        first_failure = time[failures[0]]
        ax1.axvline(x=first_failure, color='red', linestyle='--', linewidth=2, label=f'First failure: {first_failure:.2f}h')
    
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('System State', fontsize=11)
    ax1.set_title('Sample Simulation - System State', fontsize=13, fontweight='bold')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Failure', 'Operational'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    ax2 = fig.add_subplot(gs[1, :])
    comp_states = results['sample_components']
    colors_map = {1: 'green', 0: 'yellow', -1: 'red'}
    
    for i, (comp_name, states) in enumerate(comp_states.items()):
        states_array = np.array(states)
        for state in [-1, 0, 1]:
            mask = (states_array == state)
            if np.any(mask):
                ax2.fill_between(time, i-0.4, i+0.4, where=mask, color=colors_map[state], alpha=0.6)
    
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('Component', fontsize=11)
    ax2.set_title('Sample Simulation - Component States', fontsize=13, fontweight='bold')
    ax2.set_yticks(range(len(comp_states)))
    ax2.set_yticklabels(list(comp_states.keys()))
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Operational'),
        Patch(facecolor='yellow', alpha=0.6, label='Non-Operational (DC)'),
        Patch(facecolor='red', alpha=0.6, label='Failure')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    ax3 = fig.add_subplot(gs[2, 0])
    if len(results['failure_times']) > 0:
        ax3.hist(results['failure_times'], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(x=results['mttf_exp'], color='red', linestyle='--', linewidth=2, label=f'MTTF={results["mttf_exp"]:.2f}h')
        ax3.set_xlabel('System Failure Time (hours)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('System Failure Time Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No failures observed', ha='center', va='center', fontsize=14)
        ax3.set_title('Failure Time Distribution', fontsize=12, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[2, 1])
    categories = ['Experimental', 'Theoretical']
    values = [results['reliability_exp'], results['reliability_theo']]
    colors = ['steelblue', 'coral']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Reliability R(Ts)', fontsize=11)
    ax4.set_title(f'System Reliability at Ts={Ts}h', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.1 * max(values)])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    output_path = os.path.join(OUTPUT_DIR, 'system_reliability_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    OUTPUT_DIR = ensure_output_dir('part2')
    
    results = run_system_monte_carlo(components_data, Ts, DT, N_SIMS)
    
    visualize_system_simulation(results)
    print(f"Created timeline diagram in: {OUTPUT_DIR}")

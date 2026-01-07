import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import COMPONENTS_DATA, Tc, Ts, DT, N_SIMS, ensure_output_dir, print_header

# Function to simulate the system
# Returns the system failure time or "None"
def simulate_system(components_db, duration, dt):

    # Initialize states of all components
    time_axis = np.arange(0, duration, dt)
    comp_states = {name: [] for name in components_db}
    comp_failure_times = {name: None for name in components_db}  # Track component failure times
    failed = {name: False for name in components_db}  # Which have failed
    current = {name: 1 for name in components_db}     # Current state
    system_history = []
    system_failure_time = None
    system_failed = False
    
    # Simulation loop for each time step
    for t in time_axis:
        # Update state of each component like in simulate_component
        for name, specs in components_db.items():
            if failed[name]:
                # If failed, remains in failed state
                current[name] = -1
            else:
                # Check duty cycle
                is_active = np.random.rand() < specs['DC']
                if is_active:
                    # Calculate failure probability: P = 1 - e^(-λ·dt)
                    lam = 1.0 / specs['MTTF']
                    if np.random.rand() < (1 - np.exp(-lam * dt)):
                        failed[name] = True
                        current[name] = -1
                        comp_failure_times[name] = t  # Record component failure time
                    else:
                        current[name] = 1
                else:
                    current[name] = 0  # Non-operational (DC)
            
            # Record state
            comp_states[name].append(current[name])
        
        # Check system state based on the structure: C1 → [C2||C3||C4] → [C5||C6] → C7
        # - Series connection: All blocks must be operational
        # - Parallel blocks: At least one component must be operational
        
        # C1 and C7 must be operational
        c1_ok = current['C1'] != -1
        c7_ok = current['C7'] != -1
        
        # Block2: At least one of C2, C3, C4 must be operational
        block2_ok = current['C2'] != -1 or current['C3'] != -1 or current['C4'] != -1
        
        # Block3: At least one of C5, C6 must be operational
        block3_ok = current['C5'] != -1 or current['C6'] != -1
        
        # System operational if all blocks are operational
        system_ok = c1_ok and block2_ok and block3_ok and c7_ok
        system_history.append(1 if system_ok else 0)
        
        # Record time of first system failure
        if not system_ok and not system_failed:
            system_failure_time = t
            system_failed = True
    
    return time_axis, np.array(system_history), comp_states, system_failure_time, comp_failure_times

# Combined function to analyze both component and system reliability
# Extracts component metrics over full Tc duration and system metrics over first Ts duration
def run_combined_analysis(components_db, Tc, Ts, dt, n_sims):
    print(f"\nRunning Combined Analysis (Tc={Tc}h, Ts={Ts}h)...")
    
    # Storage for component failure times (for component analysis at Tc)
    comp_failure_times_Tc = {name: [] for name in components_db}
    
    # Storage for system failure times (for system analysis at Ts)
    system_failure_times_Ts = []
    
    # Sample data for visualization
    sample_data = None
    
    # Time step count for Ts cutoff
    Ts_steps = int(Ts / dt)
    
    # Execute N independent simulations for full Tc duration
    for i in range(n_sims):
        time_axis, sys_hist, comp_states, sys_ft, comp_ft = simulate_system(components_db, Tc, dt)
        
        # Progress tracking
        if (i + 1) % 100 == 0:
            print(f"\r(Completed {i+1}/{n_sims} simulations)", end='', flush=True)
        
        # Store sample data from first simulation
        if i == 0:
            # Store full data for component timeline, but truncate system data to Ts for system plot
            sample_data = (time_axis[:Ts_steps], sys_hist[:Ts_steps], 
                          {name: states[:Ts_steps] for name, states in comp_states.items()})
        
        # Collect component failure times (for Tc analysis)
        for name, ft in comp_ft.items():
            if ft is not None:
                comp_failure_times_Tc[name].append(ft)
        
        # Collect system failure time only if it occurred within Ts
        if sys_ft is not None and sys_ft < Ts:
            system_failure_times_Ts.append(sys_ft)
    
    # --- Calculate Component Metrics ---
    print(f"\n\nCOMPONENT RELIABILITY:")
    comp_results = []
    
    for comp_name, specs in components_db.items():
        mttf, dc = specs['MTTF'], specs['DC']
        failure_times = comp_failure_times_Tc[comp_name]
        
        # Experimental Reliability R(Tc): Percentage of simulations without failure
        failures = len(failure_times)
        R_exp = (n_sims - failures) / n_sims
        
        # Theoretical Reliability: R(t) = e^(-λ·DC·t)
        R_theo = np.exp(-dc * Tc / mttf)
        
        # Theoretical Failure rate: λ = 1/MTTF
        lam_theo = 1.0 / mttf
        
        # Experimental Failure rate: R(t) = e^(-λ·DC·t) → λ = -ln(R)/(DC·t)
        lam_exp = -np.log(R_exp) / (dc * Tc) if R_exp > 0 and dc > 0 else 0
        
        print(f"\n{comp_name}: MTTF={mttf}h, DC={dc}")
        print(f"  R: \tExp={R_exp:.4f}, \tTheo={R_theo:.4f}, \tError={abs(R_exp-R_theo)/R_theo*100:.1f}%")
        print(f"  λ: \tExp={lam_exp:.4f}, \tTheo={lam_theo:.4f}, \tError={abs(lam_exp-lam_theo)/lam_theo*100:.1f}%")
        
        comp_results.append({
            'component': comp_name, 'R_exp': R_exp, 'R_theo': R_theo,
            'lambda_exp': lam_exp, 'lambda_theo': lam_theo,
            'failures': failures, 'failure_times': failure_times
        })
    
    # --- Calculate System Metrics ---
    print(f"\nSYSTEM RELIABILITY:")
    
    # Experimental Reliability R(Ts): Percentage of simulations without failure within Ts
    failures = len(system_failure_times_Ts)
    R_exp = (n_sims - failures) / n_sims
    
    # Experimental MTTF: Mean of all failure times
    MTTF_exp = np.mean(system_failure_times_Ts) if system_failure_times_Ts else float('inf')
    
    # Experimental Failure rate: λ = -ln(R)/t
    lam_exp = -np.log(R_exp) / Ts if R_exp > 0 else float('inf')
    
    # Theoretical System Reliability
    def R_comp(mttf, dc, t):
        return np.exp(-dc * t / mttf)
    R_C = {name: R_comp(s['MTTF'], s['DC'], Ts) for name, s in components_db.items()}
    R_block2 = 1 - (1-R_C['C2']) * (1-R_C['C3']) * (1-R_C['C4'])
    R_block3 = 1 - (1-R_C['C5']) * (1-R_C['C6'])
    R_theo = R_C['C1'] * R_block2 * R_block3 * R_C['C7']
    
    # Theoretical Failure rate: λ = -ln(R)/t
    lam_theo = -np.log(R_theo) / Ts if R_theo > 0 else float('inf')
    
    # Theoretical MTTF: MTTF = 1/λ
    MTTF_theo = 1 / lam_theo if lam_theo != float('inf') and lam_theo > 0 else float('inf')
    
    print(f"\nSystem R: \tExp={R_exp:.4f}, \tTheo={R_theo:.4f}, \tError={abs(R_exp-R_theo)/R_theo*100:.1f}%")
    print(f"System λ: \tExp={lam_exp:.4f}, \tTheo={lam_theo:.4f}, \tError={abs(lam_exp-lam_theo)/lam_theo*100:.1f}%")
    print(f"System MTTF: \tExp={MTTF_exp:.2f}h, \tTheo={MTTF_theo:.2f}h, \tError={abs(MTTF_exp-MTTF_theo)/MTTF_theo*100:.1f}%" if MTTF_exp != float('inf') else f"System MTTF: Theo={MTTF_theo:.2f}h (No failures observed)")
    
    sys_results = {
        'R_exp': R_exp, 'R_theo': R_theo, 
        'MTTF_exp': MTTF_exp, 'MTTF_theo': MTTF_theo,
        'lambda_exp': lam_exp, 'lambda_theo': lam_theo,
        'failure_times': system_failure_times_Ts, 'sample_data': sample_data
    }
    
    return comp_results, sys_results


# VISUALIZATION
def create_component_plots(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    components = [r['component'] for r in results]
    x = np.arange(len(components))
    width = 0.35
    
    # Reliability comparison
    ax1 = axes[0]
    ax1.bar(x - width/2, [r['R_exp'] for r in results], width, label='Experimental', alpha=0.8)
    ax1.bar(x + width/2, [r['R_theo'] for r in results], width, label='Theoretical', alpha=0.8)
    ax1.set_xlabel('Component')
    ax1.set_ylabel(f'Reliability R(Tc)')
    ax1.set_title(f'Component Reliability at Tc={Tc}h')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Lambda comparison
    ax2 = axes[1]
    ax2.bar(x - width/2, [r['lambda_exp'] for r in results], width, label='Experimental', alpha=0.8)
    ax2.bar(x + width/2, [r['lambda_theo'] for r in results], width, label='Theoretical', alpha=0.8)
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Failure Rate λ (failures/h)')
    ax2.set_title('Component Failure Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_reliability.png'), dpi=150)
    plt.close()

def create_system_plots(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability bar chart
    ax1 = axes[0]
    bars = ax1.bar(['Experimental', 'Theoretical'], [results['R_exp'], results['R_theo']], color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel(f'Reliability R(Ts)')
    ax1.set_title(f'System Reliability at Ts={Ts}h')
    ax1.set_ylim([0, 1.1 * max(results['R_exp'], results['R_theo'])])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Failure time histogram
    ax2 = axes[1]
    if results['failure_times']:
        ax2.hist(results['failure_times'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(results['MTTF_exp'], color='red', linestyle='--', linewidth=2, label=f'MTTF Exp={results["MTTF_exp"]:.2f}h')
        ax2.axvline(results['MTTF_theo'], color='orange', linestyle='-.', linewidth=2, label=f'MTTF Theo={results["MTTF_theo"]:.2f}h')
        ax2.legend()
    ax2.set_xlabel('System Failure Time (hours)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('System Failure Time Distribution (up to)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_reliability.png'), dpi=150)
    plt.close()

def create_timeline_plot(sample_data, output_dir):
    time, sys_hist, comp_states = sample_data
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 2])
    
    # System timeline
    ax1 = axes[0]
    ax1.fill_between(time, 0, sys_hist, where=(sys_hist==1), color='green', alpha=0.6, label='Operational')
    ax1.fill_between(time, 0, sys_hist, where=(sys_hist==0), color='red', alpha=0.6, label='Failed')
    ax1.set_ylabel('System State')
    ax1.set_title('Sample Simulation - System State Over Time')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Failed', 'OK'])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Component timeline
    ax2 = axes[1]
    colors = {1: 'green', 0: 'yellow', -1: 'red'}
    components = list(comp_states.keys())
    
    for i, (name, states) in enumerate(comp_states.items()):
        states_arr = np.array(states)
        for state, color in colors.items():
            mask = states_arr == state
            if np.any(mask):
                ax2.fill_between(time, i-0.4, i+0.4, where=mask, color=color, alpha=0.6)
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Component')
    ax2.set_title('Component States Over Time')
    ax2.set_yticks(range(len(components)))
    ax2.set_yticklabels(components)
    ax2.grid(True, alpha=0.3, axis='x')
    
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Operational'),
        Patch(facecolor='yellow', alpha=0.6, label='Non-Op (DC)'),
        Patch(facecolor='red', alpha=0.6, label='Failed')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeline_no_repair.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    OUTPUT_DIR = ensure_output_dir('no_repair')
    
    # Combined analysis - runs simulations once for Tc and extracts both component and system metrics
    comp_results, sys_results = run_combined_analysis(COMPONENTS_DATA, Tc, Ts, DT, N_SIMS)
    
    # Create plots
    create_component_plots(comp_results, OUTPUT_DIR)
    create_system_plots(sys_results, OUTPUT_DIR)
    create_timeline_plot(sys_results['sample_data'], OUTPUT_DIR)
    
    print(f"\nPlots saved to: {OUTPUT_DIR}/")

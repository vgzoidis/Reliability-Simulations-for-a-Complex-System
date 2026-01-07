import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import COMPONENTS_DATA, Tc, Ts, DT, N_SIMS, ensure_output_dir, print_header

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

# Function to simulate a component until first failure (without repair)
# Returns the failure time or "None" if it didn't fail
def simulate_component(mttf, duty_cycle, duration, dt):
    # Calculate failure rate λ from MTTF (λ = 1/MTTF)
    lam = 1.0 / mttf
    
    # Create time axis with step dt
    time_axis = np.arange(0, duration, dt)
    status_history = []
    failure_time = None
    failed = False
    
    # Simulation loop for each time step
    for t in time_axis:
        if failed:
            # If already failed, remains in failed state (-1)
            status_history.append(-1)
        else:
            # Check if component should be active (duty cycle)
            # Random number [0,1) < DC → active
            is_active = np.random.rand() < duty_cycle
            
            if is_active:
                # Calculate failure probability in interval dt
                # From exponential distribution: P(fail in dt) = 1 - e^(-λ·dt)
                if np.random.rand() < (1 - np.exp(-lam * dt)):
                    # Failure! Record time and change state
                    failed = True
                    failure_time = t
                    status_history.append(-1)  # State: Failed
                else:
                    status_history.append(1)   # State: Operational
            else:
                # Non-operational due to duty cycle (not failure)
                status_history.append(0)
    
    return time_axis, np.array(status_history), failure_time

# Function to simulate the system
# Returns the system failure time or "None"
def simulate_system(components_db, duration, dt):

    # Initialize states of all components
    time_axis = np.arange(0, duration, dt)
    comp_states = {name: [] for name in components_db}
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
    
    return time_axis, np.array(system_history), comp_states, system_failure_time

# =============================================================================
# ANALYSIS AND PLOTTING FUNCTIONS
# =============================================================================

# Function to analyze each component's reliability
# Calculates R(Tc) and λ Theoretical and Experimental for each component
def run_component_analysis(components_db, duration, dt, n_sims):

    print_header("COMPONENT RELIABILITY ANALYSIS (No Repair)", "=", 70)
    results = []
    
    for comp_name, specs in components_db.items():
        mttf, dc = specs['MTTF'], specs['DC']
        failure_times = []
        
        # Execute N independent simulations
        for i in range(n_sims):
            _, _, ft = simulate_component(mttf, dc, duration, dt)
            if ft is not None:
                failure_times.append(ft)
        
        # --- Calculate Metrics ---

        # Experimental Reliability R(Tc): Percentage of simulations without failure
        failures = len(failure_times)
        R_exp = (n_sims - failures) / n_sims
        
        # Theoretical Reliability: R(t) = e^(-λ·DC·t)
        R_theo = np.exp(-dc * duration / mttf)
        
        # Theoretical Failure rate: λ = 1/MTTF
        lam_theo = 1.0 / mttf
        
        # Experimental Failure rate: R(t) = e^(-λ·DC·t) → λ = -ln(R)/(DC·t)
        lam_exp = -np.log(R_exp) / (dc * duration) if R_exp > 0 and dc > 0 else 0
        
        print(f"\n{comp_name}: MTTF={mttf}h, DC={dc}")
        print(f"  R(Tc={duration}h): Exp={R_exp:.4f}, Theo={R_theo:.4f}, Error={abs(R_exp-R_theo)/R_theo*100:.1f}%")
        print(f"  λ: Exp={lam_exp:.6f}, Theo={lam_theo:.6f} failures/h, Error={abs(lam_exp-lam_theo)/lam_theo*100:.1f}%")
        
        results.append({
            'component': comp_name, 'R_exp': R_exp, 'R_theo': R_theo,
            'lambda_exp': lam_exp, 'lambda_theo': lam_theo,
            'failures': failures, 'failure_times': failure_times
        })
    
    return results

# Function to analyze system reliability
# Calculates R(Ts), λ, and MTTF (Theoretical and Experimental) for the system
def run_system_analysis(components_db, duration, dt, n_sims):
    print_header("SYSTEM RELIABILITY ANALYSIS (No Repair)", "=", 70)
    
    failure_times = []
    sample_data = None
    
    # Execute N independent system simulations
    for i in range(n_sims):
        time_axis, sys_hist, comp_states, ft = simulate_system(components_db, duration, dt)
        if i == 0:
            sample_data = (time_axis, sys_hist, comp_states)
        if ft is not None:
            failure_times.append(ft)
        if (i + 1) % 200 == 0:
            print(f"Completed {i+1}/{n_sims} simulations...")
    
    # --- Calculate Metrics ---
    
    # Experimental Reliability R(Ts): Percentage of simulations without failure
    failures = len(failure_times)
    R_exp = (n_sims - failures) / n_sims
    
    # Experimental MTTF: Mean of all failure times
    MTTF_exp = np.mean(failure_times) if failure_times else float('inf')
    
    # Experimental Failure rate: λ = -ln(R)/t
    lam_exp = -np.log(R_exp) / duration if R_exp > 0 else float('inf')
    
    #Theoretical System Reliability
    def R_comp(mttf, dc, t):
        return np.exp(-dc * t / mttf)
    R_C = {name: R_comp(s['MTTF'], s['DC'], duration) for name, s in components_db.items()}
    R_block2 = 1 - (1-R_C['C2']) * (1-R_C['C3']) * (1-R_C['C4'])
    R_block3 = 1 - (1-R_C['C5']) * (1-R_C['C6'])
    R_theo = R_C['C1'] * R_block2 * R_block3 * R_C['C7']
    
    # Theoretical Failure rate: λ = -ln(R)/t
    lam_theo = -np.log(R_theo) / duration if R_theo > 0 else float('inf')
    
    # Theoretical MTTF: MTTF = 1/λ
    MTTF_theo = 1 / lam_theo if lam_theo != float('inf') and lam_theo > 0 else float('inf')
    
    print(f"\nSystem R(Ts={duration}h): Exp={R_exp:.4f}, Theo={R_theo:.4f}, Error={abs(R_exp-R_theo)/R_theo*100:.1f}%")
    print(f"System λ: Exp={lam_exp:.6f}, Theo={lam_theo:.6f} failures/h, Error={abs(lam_exp-lam_theo)/lam_theo*100:.1f}%")
    print(f"System MTTF: Exp={MTTF_exp:.2f}h, Theo={MTTF_theo:.2f}h, Error={abs(MTTF_exp-MTTF_theo)/MTTF_theo*100:.1f}%" if MTTF_exp != float('inf') else f"System MTTF: Theo={MTTF_theo:.2f}h (No failures observed)")
    
    return {
        'R_exp': R_exp, 'R_theo': R_theo, 
        'MTTF_exp': MTTF_exp, 'MTTF_theo': MTTF_theo,
        'lambda_exp': lam_exp, 'lambda_theo': lam_theo,
        'failure_times': failure_times, 'sample_data': sample_data
    }

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
    ax2.set_title('System Failure Time Distribution')
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
    
    # Component analysis
    comp_results = run_component_analysis(COMPONENTS_DATA, Tc, DT, N_SIMS)
    create_component_plots(comp_results, OUTPUT_DIR)
    
    # System analysis
    sys_results = run_system_analysis(COMPONENTS_DATA, Ts, DT, N_SIMS)
    create_system_plots(sys_results, OUTPUT_DIR)
    create_timeline_plot(sys_results['sample_data'], OUTPUT_DIR)
    
    print(f"\nPlots saved to: {OUTPUT_DIR}/")

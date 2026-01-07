import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import COMPONENTS_DATA, Tc, DT, N_SIMS, ensure_output_dir, print_header

# Component States:
#  1 = Operational (active)
#  0 = Non-operational (due to duty cycle)
# -1 = Failed (waiting for repair)
# -2 = Under repair

# Function to simulate the system with repair
# When a component fails, it undergoes repair for a random time tr ~ Exp(1/MTTR).
# After repair, the component returns to operational state.
def simulate_system_with_repair(components_db, duration, dt):
    
    time_axis = np.arange(0, duration, dt)
    n_steps = len(time_axis)
    
    # Component tracking
    comp_states = {name: [] for name in components_db}
    current = {name: 1 for name in components_db}      # Current state
    failed = {name: False for name in components_db}   # Is currently failed/under repair
    repair_end_time = {name: None for name in components_db}  # When repair will complete
    
    # Component metrics tracking
    comp_failure_times = {name: [] for name in components_db}  # List of all failure times
    comp_repair_times = {name: [] for name in components_db}   # List of all repair durations
    comp_up_times = {name: [] for name in components_db}       # List of all up times between failures
    last_up_start = {name: 0 for name in components_db}        # When current up period started
    
    # System tracking
    system_history = []
    system_failed = False
    system_failure_times = []
    system_repair_times = []
    system_up_times = []
    last_system_up_start = 0
    system_down_start = None
    
    # Simulation loop
    for step, t in enumerate(time_axis):
        # Update each component
        for name, specs in components_db.items():
            mttf = specs['MTTF']
            mttr = specs['MTTR']
            dc = specs['DC']
            
            if failed[name]:
                # Component is failed or under repair
                if repair_end_time[name] is not None and t >= repair_end_time[name]:
                    # Repair complete - component returns to operational
                    failed[name] = False
                    repair_end_time[name] = None
                    current[name] = 1
                    last_up_start[name] = t  # Start new up period
                else:
                    # Still under repair
                    current[name] = -2
            else:
                # Component is operational - check for failure
                is_active = np.random.rand() < dc
                if is_active:
                    # Calculate failure probability: P = 1 - e^(-λ·dt)
                    lam = 1.0 / mttf
                    if np.random.rand() < (1 - np.exp(-lam * dt)):
                        # Failure occurred
                        failed[name] = True
                        current[name] = -1
                        
                        # Record up time (time since last repair or start)
                        up_time = t - last_up_start[name]
                        if up_time > 0:
                            comp_up_times[name].append(up_time)
                        
                        # Generate repair time from exponential distribution
                        tr = np.random.exponential(mttr)
                        repair_end_time[name] = t + tr
                        comp_failure_times[name].append(t)
                        comp_repair_times[name].append(tr)
                    else:
                        current[name] = 1
                else:
                    current[name] = 0  # Non-operational (DC)
            
            comp_states[name].append(current[name])
        
        # Check system state based on RBD: C1 → [C2||C3||C4] → [C5||C6] → C7
        c1_ok = current['C1'] >= 0
        c7_ok = current['C7'] >= 0
        block2_ok = current['C2'] >= 0 or current['C3'] >= 0 or current['C4'] >= 0
        block3_ok = current['C5'] >= 0 or current['C6'] >= 0
        
        system_ok = c1_ok and block2_ok and block3_ok and c7_ok
        system_history.append(1 if system_ok else 0)
        
        # Track system failures and repairs
        if not system_ok and not system_failed:
            # System just failed
            system_failed = True
            system_failure_times.append(t)
            system_down_start = t
            
            # Record system up time
            up_time = t - last_system_up_start
            if up_time > 0:
                system_up_times.append(up_time)
                
        elif system_ok and system_failed:
            # System just recovered
            system_failed = False
            last_system_up_start = t
            
            # Record system repair time
            if system_down_start is not None:
                repair_time = t - system_down_start
                system_repair_times.append(repair_time)
                system_down_start = None
    
    # Handle final up period for components (if still operational at end)
    for name in components_db:
        if not failed[name]:
            final_up_time = duration - last_up_start[name]
            if final_up_time > 0:
                comp_up_times[name].append(final_up_time)
    
    # Handle final system up period
    if not system_failed:
        final_up_time = duration - last_system_up_start
        if final_up_time > 0:
            system_up_times.append(final_up_time)
    
    # Compile component metrics
    comp_metrics = {}
    for name in components_db:
        comp_metrics[name] = {
            'failure_times': comp_failure_times[name],
            'repair_times': comp_repair_times[name],
            'up_times': comp_up_times[name],
            'n_failures': len(comp_failure_times[name])
        }
    
    # Compile system metrics
    sys_metrics = {
        'failure_times': system_failure_times,
        'repair_times': system_repair_times,
        'up_times': system_up_times,
        'n_failures': len(system_failure_times)
    }
    
    return time_axis, np.array(system_history), comp_states, comp_metrics, sys_metrics

# Function to analyze the components and system reliability
# Extracts metrics for MTBF, MUT, MTTR, and Availability
def run_availability_analysis(components_db, Tc, dt, n_sims):

    print(f"\nRunning Availability Analysis with Repair (Tc={Tc}h)...")
    
    # Aggregate metrics across simulations
    comp_all_failures = {name: 0 for name in components_db}
    comp_all_repair_times = {name: [] for name in components_db}
    comp_all_up_times = {name: [] for name in components_db}
    comp_total_up = {name: 0 for name in components_db}
    comp_total_down = {name: 0 for name in components_db}
    
    sys_all_failures = 0
    sys_all_repair_times = []
    sys_all_up_times = []
    sys_total_up = 0
    sys_total_down = 0
    
    sample_data = None
    
    for i in range(n_sims):
        time_axis, sys_hist, comp_states, comp_metrics, sys_metrics = simulate_system_with_repair(
            components_db, Tc, dt
        )
        
        if (i + 1) % 100 == 0:
            print(f"\r(Completed {i+1}/{n_sims} simulations)", end='', flush=True)
        
        # Store sample from first simulation
        if i == 0:
            sample_data = (time_axis, sys_hist, comp_states)
        
        # Aggregate component metrics
        for name in components_db:
            comp_all_failures[name] += comp_metrics[name]['n_failures']
            comp_all_repair_times[name].extend(comp_metrics[name]['repair_times'])
            comp_all_up_times[name].extend(comp_metrics[name]['up_times'])
            
            # Calculate up/down time from states
            states = np.array(comp_states[name])
            up_steps = np.sum(states >= 0)
            down_steps = np.sum(states < 0)
            comp_total_up[name] += up_steps * dt
            comp_total_down[name] += down_steps * dt
        
        # Aggregate system metrics
        sys_all_failures += sys_metrics['n_failures']
        sys_all_repair_times.extend(sys_metrics['repair_times'])
        sys_all_up_times.extend(sys_metrics['up_times'])
        
        # Calculate system up/down time
        sys_up_steps = np.sum(sys_hist == 1)
        sys_down_steps = np.sum(sys_hist == 0)
        sys_total_up += sys_up_steps * dt
        sys_total_down += sys_down_steps * dt
    
    print("\n")
    
    # Calculate component metrics
    print_header("COMPONENT AVAILABILITY METRICS", "-", 70)
    comp_results = []
    
    for comp_name, specs in components_db.items():
        mttf_theo = specs['MTTF']
        mttr_theo = specs['MTTR']
        dc = specs['DC']
        
        n_failures = comp_all_failures[comp_name]
        total_time = n_sims * Tc
        
        # Experimental MUT: Average of all up times
        if comp_all_up_times[comp_name]:
            MUT_exp = np.mean(comp_all_up_times[comp_name])
        else:
            MUT_exp = Tc  # No failures means full up time
        
        # Experimental MTTR: Average of all repair times
        if comp_all_repair_times[comp_name]:
            MTTR_exp = np.mean(comp_all_repair_times[comp_name])
        else:
            MTTR_exp = 0
        
        # Experimental MTBF: MTBF = MUT + MTTR (mean time between failures)
        MTBF_exp = MUT_exp + MTTR_exp
        
        # Experimental Availability: Total up time / Total time
        A_exp = comp_total_up[comp_name] / total_time
        
        # Theoretical values
        MTTF_eff = mttf_theo / dc if dc > 0 else float('inf') # Effective MTTF considering DC: MTTF_eff = MTTF / DC
        MTBF_theo = MTTF_eff + mttr_theo  # MTBF = MTTF + MTTR
        MUT_theo = MTTF_eff  # Mean Up Time = MTTF (effective)
        MTTR_theo_val = mttr_theo
        
        # Availability A = MTTF / (MTTF + MTTR) = MUT / MTBF
        A_theo = MTTF_eff / (MTTF_eff + mttr_theo) if (MTTF_eff + mttr_theo) > 0 else 1.0
        
        print(f"\n{comp_name}: MTTF={mttf_theo}h, MTTR={mttr_theo}h, DC={dc}")
        print(f"  MTBF:\t\tExp={MTBF_exp:.2f}h,\tTheo={MTBF_theo:.2f}h,\tError={abs(MTBF_exp-MTBF_theo)/MTBF_theo*100:.1f}%")
        print(f"  MUT:\t\tExp={MUT_exp:.2f}h,\tTheo={MUT_theo:.2f}h,\tError={abs(MUT_exp-MUT_theo)/MUT_theo*100:.1f}%")
        print(f"  MTTR:\t\tExp={MTTR_exp:.2f}h,\tTheo={MTTR_theo_val:.2f}h,\tError={abs(MTTR_exp-MTTR_theo_val)/MTTR_theo_val*100:.1f}%" if MTTR_theo_val > 0 else f"  MTTR:\t\tExp={MTTR_exp:.2f}h,\tTheo={MTTR_theo_val:.2f}h")
        print(f"  A:\t\tExp={A_exp:.4f},\tTheo={A_theo:.4f},\tError={abs(A_exp-A_theo)/A_theo*100:.1f}%")
        
        comp_results.append({
            'component': comp_name,
            'MTBF_exp': MTBF_exp, 'MTBF_theo': MTBF_theo,
            'MUT_exp': MUT_exp, 'MUT_theo': MUT_theo,
            'MTTR_exp': MTTR_exp, 'MTTR_theo': MTTR_theo_val,
            'A_exp': A_exp, 'A_theo': A_theo,
            'n_failures': n_failures
        })
    
    # Calculate system metrics
    print_header("SYSTEM AVAILABILITY METRICS", "-", 70)
    
    total_time = n_sims * Tc
    
    # Experimental system MUT
    if sys_all_up_times:
        sys_MUT_exp = np.mean(sys_all_up_times)
    else:
        sys_MUT_exp = Tc
    
    # Experimental system MTTR
    if sys_all_repair_times:
        sys_MTTR_exp = np.mean(sys_all_repair_times)
    else:
        sys_MTTR_exp = 0
    
    # Experimental system MTBF: MTBF = MUT + MTTR (mean time between failures)
    sys_MTBF_exp = sys_MUT_exp + sys_MTTR_exp
    
    # Experimental system Availability
    sys_A_exp = sys_total_up / total_time
    
    # Theoretical system availability (approximate using component availabilities)
    def comp_availability(mttf, dc, mttr):
        mttf_eff = mttf / dc if dc > 0 else float('inf')
        return mttf_eff / (mttf_eff + mttr)
    
    A_C = {name: comp_availability(s['MTTF'], s['DC'], s['MTTR']) 
           for name, s in components_db.items()}
    
    # System availability based on RBD structure
    A_block2 = 1 - (1 - A_C['C2']) * (1 - A_C['C3']) * (1 - A_C['C4'])
    A_block3 = 1 - (1 - A_C['C5']) * (1 - A_C['C6'])
    sys_A_theo = A_C['C1'] * A_block2 * A_block3 * A_C['C7']
    
    # Component effective failure rates (considering duty cycle)
    def comp_failure_rate(mttf, dc):
        return dc / mttf
    
    def comp_mttf_eff(mttf, dc):
        return mttf / dc if dc > 0 else float('inf')
    
    # Get component effective MTTFs and MTTRs
    mttf_eff = {name: comp_mttf_eff(s['MTTF'], s['DC']) for name, s in components_db.items()}
    mttr_comp = {name: s['MTTR'] for name, s in components_db.items()}
    
    # Series component failure rates
    lambda_C1 = 1.0 / mttf_eff['C1']
    lambda_C7 = 1.0 / mttf_eff['C7']
    
    # Parallel block 2 (C2, C3, C4): All must fail for block to fail
    lambda_C2 = 1.0 / mttf_eff['C2']
    lambda_C3 = 1.0 / mttf_eff['C3']
    lambda_C4 = 1.0 / mttf_eff['C4']
    mu_C2, mu_C3, mu_C4 = 1.0/mttr_comp['C2'], 1.0/mttr_comp['C3'], 1.0/mttr_comp['C4']
    lambda_block2 = (lambda_C2 * lambda_C3 * lambda_C4) / (mu_C2*mu_C3 + mu_C2*mu_C4 + mu_C3*mu_C4)
    
    # Parallel block 3 (C5, C6): Both must fail for block to fail
    lambda_C5 = 1.0 / mttf_eff['C5']
    lambda_C6 = 1.0 / mttf_eff['C6']
    mu_C5, mu_C6 = 1.0/mttr_comp['C5'], 1.0/mttr_comp['C6']
    lambda_block3 = (lambda_C5 * lambda_C6) / (mu_C5 + mu_C6)
    
    # System failure rate (series connection of all blocks)
    lambda_sys_theo = lambda_C1 + lambda_block2 + lambda_block3 + lambda_C7
    
    # Theoretical system MUT = 1 / λ_sys
    sys_MUT_theo = 1.0 / lambda_sys_theo if lambda_sys_theo > 0 else float('inf')
    
    # Theoretical system MTTR using MTTR = MUT * (1 - A) / A
    sys_MTTR_theo = sys_MUT_theo * (1 - sys_A_theo) / sys_A_theo if sys_A_theo > 0 else 0
    
    # Theoretical system MTBF = MUT + MTTR
    sys_MTBF_theo = sys_MUT_theo + sys_MTTR_theo
    
    print(f"\nSystem Metrics:")
    print(f"  MTBF:\t\tExp={sys_MTBF_exp:.2f}h,\tTheo={sys_MTBF_theo:.2f}h,\tError={abs(sys_MTBF_exp-sys_MTBF_theo)/sys_MTBF_theo*100:.1f}%")
    print(f"  MUT:\t\tExp={sys_MUT_exp:.2f}h,\tTheo={sys_MUT_theo:.2f}h,\tError={abs(sys_MUT_exp-sys_MUT_theo)/sys_MUT_theo*100:.1f}%")
    print(f"  MTTR:\t\tExp={sys_MTTR_exp:.2f}h,\tTheo={sys_MTTR_theo:.2f}h,\tError={abs(sys_MTTR_exp-sys_MTTR_theo)/sys_MTTR_theo*100:.1f}%" if sys_MTTR_theo > 0 else f"  MTTR:\t\tExp={sys_MTTR_exp:.2f}h,\tTheo={sys_MTTR_theo:.2f}h")
    print(f"  A:\t\tExp={sys_A_exp:.4f},\tTheo={sys_A_theo:.4f},\tError={abs(sys_A_exp-sys_A_theo)/sys_A_theo*100:.1f}%")
    
    sys_results = {
        'MTBF_exp': sys_MTBF_exp, 'MTBF_theo': sys_MTBF_theo,
        'MUT_exp': sys_MUT_exp, 'MUT_theo': sys_MUT_theo,
        'MTTR_exp': sys_MTTR_exp, 'MTTR_theo': sys_MTTR_theo,
        'A_exp': sys_A_exp, 'A_theo': sys_A_theo,
        'n_failures': sys_all_failures,
        'sample_data': sample_data
    }
    
    return comp_results, sys_results


# VISUALIZATION

def create_component_availability_plots(results, output_dir):
    """Create plots comparing component availability metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = [r['component'] for r in results]
    x = np.arange(len(components))
    width = 0.35
    
    # MTBF comparison
    ax1 = axes[0, 0]
    mtbf_exp = [r['MTBF_exp'] if r['MTBF_exp'] != float('inf') else 0 for r in results]
    mtbf_theo = [r['MTBF_theo'] for r in results]
    ax1.bar(x - width/2, mtbf_exp, width, label='Experimental', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, mtbf_theo, width, label='Theoretical', alpha=0.8, color='coral')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('MTBF (hours)')
    ax1.set_title('Mean Time Between Failures (MTBF)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # MUT comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, [r['MUT_exp'] for r in results], width, label='Experimental', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, [r['MUT_theo'] for r in results], width, label='Theoretical', alpha=0.8, color='coral')
    ax2.set_xlabel('Component')
    ax2.set_ylabel('MUT (hours)')
    ax2.set_title('Mean Up Time (MUT)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # MTTR comparison
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, [r['MTTR_exp'] for r in results], width, label='Experimental', alpha=0.8, color='steelblue')
    ax3.bar(x + width/2, [r['MTTR_theo'] for r in results], width, label='Theoretical', alpha=0.8, color='coral')
    ax3.set_xlabel('Component')
    ax3.set_ylabel('MTTR (hours)')
    ax3.set_title('Mean Time To Repair (MTTR)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(components)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Availability comparison
    ax4 = axes[1, 1]
    ax4.bar(x - width/2, [r['A_exp'] for r in results], width, label='Experimental', alpha=0.8, color='steelblue')
    ax4.bar(x + width/2, [r['A_theo'] for r in results], width, label='Theoretical', alpha=0.8, color='coral')
    ax4.set_xlabel('Component')
    ax4.set_ylabel('Availability')
    ax4.set_title('Component Availability (A)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(components)
    ax4.legend()
    ax4.set_ylim([0.9 * min([r['A_exp'] for r in results]), 1.0])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_availability.png'), dpi=150)
    plt.close()


def create_system_availability_plots(results, output_dir):
    """Create plots for system availability metrics with experimental vs theoretical comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    width = 0.35
    x = np.arange(1)
    
    # MTBF comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, [results['MTBF_exp']], width, label='Experimental', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, [results['MTBF_theo']], width, label='Theoretical', alpha=0.8, color='coral')
    ax1.set_ylabel('MTBF (hours)')
    ax1.set_title('System Mean Time Between Failures (MTBF)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['System'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}h', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}h', ha='center', va='bottom', fontsize=9)
    
    # MUT comparison
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, [results['MUT_exp']], width, label='Experimental', alpha=0.8, color='steelblue')
    bars2 = ax2.bar(x + width/2, [results['MUT_theo']], width, label='Theoretical', alpha=0.8, color='coral')
    ax2.set_ylabel('MUT (hours)')
    ax2.set_title('System Mean Up Time (MUT)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['System'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}h', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}h', ha='center', va='bottom', fontsize=9)
    
    # MTTR comparison
    ax3 = axes[1, 0]
    bars1 = ax3.bar(x - width/2, [results['MTTR_exp']], width, label='Experimental', alpha=0.8, color='steelblue')
    bars2 = ax3.bar(x + width/2, [results['MTTR_theo']], width, label='Theoretical', alpha=0.8, color='coral')
    ax3.set_ylabel('MTTR (hours)')
    ax3.set_title('System Mean Time To Repair (MTTR)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['System'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}h', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}h', ha='center', va='bottom', fontsize=9)
    
    # Availability comparison
    ax4 = axes[1, 1]
    bars1 = ax4.bar(x - width/2, [results['A_exp']], width, label='Experimental', alpha=0.8, color='steelblue')
    bars2 = ax4.bar(x + width/2, [results['A_theo']], width, label='Theoretical', alpha=0.8, color='coral')
    ax4.set_ylabel('Availability')
    ax4.set_title(f'System Availability at Tc={Tc}h')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['System'])
    ax4.legend()
    ax4.set_ylim([0.9 * min(results['A_exp'], results['A_theo']), 1.0])
    ax4.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_availability.png'), dpi=150)
    plt.close()


def create_timeline_plot_with_repair(sample_data, output_dir):
    """Create timeline plot showing component and system states including repair."""
    time, sys_hist, comp_states = sample_data
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 2])
    
    # Helper function to get contiguous segments
    def get_segments(states_arr, target_state):
        segments = []
        in_segment = False
        start = 0
        for i, state in enumerate(states_arr):
            if state == target_state and not in_segment:
                start = time[i]
                in_segment = True
            elif state != target_state and in_segment:
                segments.append((start, time[i] - start))
                in_segment = False
        if in_segment:
            # Extend to end of time + dt to cover last interval
            segments.append((start, time[-1] - start + dt))
        return segments
    
    # System timeline using broken_barh
    ax1 = axes[0]
    sys_hist_arr = np.array(sys_hist)
    
    # Draw operational segments (green)
    op_segments = get_segments(sys_hist_arr, 1)
    if op_segments:
        ax1.broken_barh(op_segments, (0, 1), facecolors='green', alpha=0.6, label='Operational')
    
    # Draw failed segments (red)
    fail_segments = get_segments(sys_hist_arr, 0)
    if fail_segments:
        ax1.broken_barh(fail_segments, (0, 1), facecolors='red', alpha=0.6, label='Failed')
    
    ax1.set_ylabel('System State')
    ax1.set_title('Sample Simulation - System State Over Time (With Repair)')
    ax1.set_yticks([0.25, 0.75])
    ax1.set_yticklabels(['Failed', 'OK'])
    ax1.set_ylim(0, 1)
    ax1.set_xlim(time[0], time[-1] + dt)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Component timeline with repair states using broken_barh
    ax2 = axes[1]
    # State colors: 1=Operational(green), 0=Non-Op DC(yellow), -1=Failed(red), -2=Under Repair(blue)
    colors = {1: 'green', 0: 'yellow', -1: 'red', -2: 'blue'}
    components = list(comp_states.keys())
    
    for i, (name, states) in enumerate(comp_states.items()):
        states_arr = np.array(states)
        for state, color in colors.items():
            segments = get_segments(states_arr, state)
            if segments:
                ax2.broken_barh(segments, (i - 0.4, 0.8), facecolors=color, alpha=0.6)
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Component')
    ax2.set_title('Component States Over Time (With Repair)')
    ax2.set_yticks(range(len(components)))
    ax2.set_yticklabels(components)
    ax2.set_xlim(time[0], time[-1] + dt)
    ax2.grid(True, alpha=0.3, axis='x')
    
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Operational'),
        Patch(facecolor='yellow', alpha=0.6, label='Non-Op (DC)'),
        Patch(facecolor='red', alpha=0.6, label='Failed'),
        Patch(facecolor='blue', alpha=0.6, label='Under Repair')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeline_with_repair.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    OUTPUT_DIR = ensure_output_dir('with_repair')
    
    # Run availability analysis with repair
    comp_results, sys_results = run_availability_analysis(COMPONENTS_DATA, Tc, DT, N_SIMS)
    
    # Create plots
    create_component_availability_plots(comp_results, OUTPUT_DIR)
    create_system_availability_plots(sys_results, OUTPUT_DIR)
    create_timeline_plot_with_repair(sys_results['sample_data'], OUTPUT_DIR)
    
    print(f"\nPlots saved to: {OUTPUT_DIR}/")

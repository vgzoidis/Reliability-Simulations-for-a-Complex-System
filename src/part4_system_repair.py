import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

# Add config parameters
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    COMPONENTS_DATA as components_data,
    Tc, DT, N_SIMS,
    ensure_output_dir)

# Executes N simulations with MTTR for the system, calculating:
# - Mean Time Between Failures
# - Mean Up Time
# - Mean Time To Repair
# - Availability
def run_system_monte_carlo_with_repair(components_db, duration, dt, n_sims):    
    all_system_up_times = []
    all_system_down_times = []
    total_up_time = 0
    total_down_time = 0
    
    # Samples for graph
    sample_time = None
    sample_system_history = None
    sample_component_histories = None
    
    for i in range(n_sims):
        time_axis, system_history, comp_histories, up_times, down_times = simulate_system_with_repair(components_db, duration, dt)
        
        # Save first results
        if i == 0:
            sample_time = time_axis
            sample_system_history = system_history
            sample_component_histories = comp_histories
        
        # Save the rest of the results
        all_system_up_times.extend(up_times)
        all_system_down_times.extend(down_times)
        
        # Overall operational/fault time
        up_time = np.sum(system_history == 1) * dt
        down_time = np.sum(system_history == 0) * dt
        total_up_time += up_time
        total_down_time += down_time
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_sims} simulations...")
    
    # 1. Mean Time Between Failures
    if len(all_system_up_times) > 0:
        mtbf_system = np.mean(all_system_up_times)
        mtbf_std = np.std(all_system_up_times)
        mtbf_median = np.median(all_system_up_times)
        
        print(f"\nMean Time Between Failures:")
        print(f"Mean & MUT:                     {mtbf_system:.4f} hours")
        print(f"Standard deviation:             {mtbf_std:.4f} hours")
        print(f"Median:                         {mtbf_median:.4f} hours")
        print(f"Minimum:                        {min(all_system_up_times):.4f} hours")
        print(f"Maximum:                        {max(all_system_up_times):.4f} hours")
        print(f"Total operational periods:      {len(all_system_up_times)}")
    else:
        mtbf_system = float('inf')
        mtbf_std = 0
        print(f"No failures detected")
    
    # 2. Mean Time To Repair
    if len(all_system_down_times) > 0:
        mttr_system = np.mean(all_system_down_times)
        mttr_std = np.std(all_system_down_times)
        mttr_median = np.median(all_system_down_times)
        
        print(f"\nMean Time To Repair:")
        print(f"Mean:                           {mttr_system:.4f} hours")
        print(f"Standard deviation:             {mttr_std:.4f} hours")
        print(f"Median:                         {mttr_median:.4f} hours")
        print(f"Minimum:                        {min(all_system_down_times):.4f} hours")
        print(f"Maximum:                        {max(all_system_down_times):.4f} hours")
        print(f"Total failure periods:          {len(all_system_down_times)}")
    else:
        mttr_system = 0
        mttr_std = 0
        print(f"No failures detected")
    
    # 3. Availability
    total_time = n_sims * duration
    availability_system = total_up_time / total_time if total_time > 0 else 0
    
    print(f"\nAvailability:")
    print(f"A = {availability_system:.6f} ({availability_system * 100:.2f}%)")
    print(f"Average operational time per simulation: {total_up_time/n_sims:.4f}h")
    print(f"Average fault time per simulation:      {total_down_time/n_sims:.4f}h")
    
    # Theoretical availability
    if mtbf_system != float('inf') and mtbf_system > 0:
        availability_theo = mtbf_system / (mtbf_system + mttr_system)
        print(f"Theoretical (from MTBF/(MTBF+MTTR)):    {availability_theo:.6f} ({availability_theo * 100:.2f}%)")
        print(f"Relative error:                         {abs(availability_system - availability_theo) / availability_theo * 100:.2f}%")
    
    # Average number of fault changes
    avg_failures = len(all_system_up_times) / n_sims
    print(f"\nFault Statistics:")
    print(f"Total number of system failures:      {len(all_system_up_times)}")
    print(f"Average faults per simulation:        {avg_failures:.2f}")
    
    if mtbf_system != float('inf') and mtbf_system > 0:
        failure_rate = 1 / mtbf_system
        print(f"Failure rate λ:                       {failure_rate:.6f} failures/hour")
    
    print(f"{'='*70}\n")
    
    return {
        'mtbf': mtbf_system,
        'mtbf_std': mtbf_std,
        'mut': mtbf_system,
        'mttr': mttr_system,
        'mttr_std': mttr_std,
        'availability': availability_system,
        'avg_failures': avg_failures,
        'all_up_times': all_system_up_times,
        'all_down_times': all_system_down_times,
        'sample_time': sample_time,
        'sample_system_history': sample_system_history,
        'sample_component_histories': sample_component_histories
    }

# Component states:
# - 2: Operational
# - 1: Non-operational due to duty cycle
# - 0: Under repair
def simulate_system_with_repair(components_db, duration, dt):
    time_axis = np.arange(0, duration, dt)
    
    # Initialize operational states
    current_status = {name: 2 for name in components_db}
    repair_timers = {name: 0.0 for name in components_db}
    
    component_histories = {name: [] for name in components_db}
    system_history = []
    system_up_times = []
    system_down_times = []
    current_system_state = None
    current_period_start = 0.0
    
    for dur, time in enumerate(time_axis):
        for name, specs in components_db.items():
            if current_status[name] == 0:
                repair_timers[name] -= dt # Component under repair
                
                if repair_timers[name] <= 0:
                    current_status[name] = 2 # Repair complete
                
                component_histories[name].append(0) 
            else:
                is_active = np.random.rand() < specs['DC'] # DC check
                
                if is_active:
                    lam = 1.0 / specs['MTTF']
                    prob_fail = 1 - np.exp(-lam * dt)
                    
                    if np.random.rand() < prob_fail: # Fault detected
                        current_status[name] = 0
                        repair_timers[name] = np.random.exponential(specs['MTTR'])
                        component_histories[name].append(0)
                    else:                            # Working status
                        current_status[name] = 2
                        component_histories[name].append(2)
                else:                                # Not working due to DC
                    current_status[name] = 1
                    component_histories[name].append(1)
        
        # Calculate system status
        c1_ok = (current_status['C1'] != 0)
        c7_ok = (current_status['C7'] != 0)
        block2_ok = (current_status['C2'] != 0) or \
                    (current_status['C3'] != 0) or \
                    (current_status['C4'] != 0)
        block3_ok = (current_status['C5'] != 0) or \
                    (current_status['C6'] != 0)
        
        system_operational = c1_ok and block2_ok and block3_ok and c7_ok
        system_state = 1 if system_operational else 0
        system_history.append(system_state)
        
        if current_system_state is None:
            current_system_state = system_state
            current_period_start = time
        elif current_system_state != system_state:
            period_duration = time - current_period_start
            
            if current_system_state == 1: # Finished operational period
                system_up_times.append(period_duration)
            else:                         # Finished fault period
                system_down_times.append(period_duration)
            
            current_system_state = system_state
            current_period_start = time
    
    # Capture last period
    final_period_duration = duration - current_period_start
    if final_period_duration > 0:
        if current_system_state == 1:
            system_up_times.append(final_period_duration)
        else:
            system_down_times.append(final_period_duration)
    
    return (time_axis, np.array(system_history), component_histories, system_up_times, system_down_times)

def visualize_system_timeline(results):
    time = results['sample_time']
    system_history = results['sample_system_history']
    comp_histories = results['sample_component_histories']
    
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_facecolor('white')
    
    color_map = {
        0: '#FF4444',  # Fault
        1: '#FFD700',  # Duty cycle
        2: '#44FF44',  # Operational
    }
    
    # Components list with system overall
    components = sorted(comp_histories.keys())
    all_items = ['SYSTEM'] + components
    n_items = len(all_items)
    
    row_height = 0.8
    y_position = n_items - 1
    states = system_history
    changes = np.where(states[:-1] != states[1:])[0] + 1
    boundaries = np.concatenate(([0], changes, [len(states)]))
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        state = states[start_idx]
        
        start_time = time[start_idx]
        if i < len(boundaries) - 2:
            end_time = time[boundaries[i + 1]]
        else:
            end_time = time[-1]

        width = end_time - start_time
        color = color_map[2] if state == 1 else color_map[0]
        rect = Rectangle((start_time, y_position - row_height/2), width, row_height,facecolor=color, edgecolor='darkgray', linewidth=0.8)
        ax.add_patch(rect)
    
    for idx, comp_name in enumerate(components):
        y_position = n_items - idx - 2
        states = np.array(comp_histories[comp_name])
        changes = np.where(states[:-1] != states[1:])[0] + 1
        boundaries = np.concatenate(([0], changes, [len(states)]))
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            state = states[start_idx]
            
            start_time = time[start_idx]
            if i < len(boundaries) - 2:
                end_time = time[boundaries[i + 1]]
            else:
                end_time = time[-1]
            width = end_time - start_time
            
            # Skip if width is too small
            if width < 0.001:
                continue
            
            rect = Rectangle((start_time, y_position - row_height/2), width, row_height, facecolor=color_map[state], edgecolor='none', linewidth=0)
            ax.add_patch(rect)
    
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([-0.5, n_items - 0.5])
    ax.set_xlabel('Operational Time (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Component / System', fontsize=13, fontweight='bold')
    ax.set_title('System and Component State Over Time', fontsize=15, fontweight='bold', pad=20)
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(all_items, fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', color='gray')
    ax.set_axisbelow(True)
    ax.axhline(y=n_items-1, color='black', linewidth=3, linestyle='-', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor= color_map[2], edgecolor='black', label='Operational'),
        Patch(facecolor= color_map[1], edgecolor='black', label='Has Fault'),
        Patch(facecolor= color_map[0], edgecolor='black', label='Under Repair'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'system_component_timeline.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')

def create_system_analysis_plots(results):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35, height_ratios=[1.2, 0.8, 1])

    # 1. MTBF Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    if len(results['all_up_times']) > 0:
        ax1.hist(results['all_up_times'], bins=50, alpha=0.75, color='#2ecc71', edgecolor='black', density=True, linewidth=1.2)
        ax1.axvline(x=results['mtbf'], color='#27ae60', linestyle='--', linewidth=3, label=f'MTBF={results["mtbf"]:.3f}h', alpha=0.9)
        
        # Theoretical
        if results['mtbf'] != float('inf'):
            x = np.linspace(0, max(results['all_up_times']), 200)
            theoretical = (1/results['mtbf']) * np.exp(-x/results['mtbf'])
            ax1.plot(x, theoretical, 'r-', linewidth=2.5, label='Theoretical Exponential', alpha=0.8)
        
        ax1.set_xlabel('System Operational Time (hours)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax1.set_title('System MTBF Distribution', fontsize=13, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, framealpha=0.95, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. MTTR Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if len(results['all_down_times']) > 0:
        ax2.hist(results['all_down_times'], bins=50, alpha=0.75, color='#e74c3c', edgecolor='black', density=True, linewidth=1.2)
        ax2.axvline(x=results['mttr'], color='#c0392b', linestyle='--', linewidth=3, label=f'MTTR={results["mttr"]:.3f}h', alpha=0.9)
        
        ax2.set_xlabel('System Failure Time (hours)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax2.set_title('System MTTR Distribution', fontsize=13, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, framealpha=0.95, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # System Metrics
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    metrics_text = f"""
    ========================================================================================
                                ΜΕΤΡΙΚΕΣ ΑΞΙΟΠΙΣΤΙΑΣ ΣΥΣΤΗΜΑΤΟΣ                            
    ========================================================================================
      Mean Time Between Failures: {results['mtbf']:>10.4f} ώρες  (σ = {results['mtbf_std']:>8.4f}h)
      Mean Up Time (MUT): {results['mut']:>10.4f} ώρες
      Mean Time To Repair (MTTR): {results['mttr']:>10.4f} ώρες  (σ = {results['mttr_std']:>8.4f}h)
      Availability: {results['availability']:>10.6f} ({results['availability']*100:>6.2f}%)
      Average Failures: {results['avg_failures']:>10.2f} ανά προσομοίωση
      Failure Rate (λ): {1/results['mtbf'] if results['mtbf'] != float('inf') else 0:>10.6f} βλάβες/ώρα
    """
    
    ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10.5, 
             verticalalignment='center', horizontalalignment='center', family='monospace', 
             bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', alpha=0.9, edgecolor='#34495e', linewidth=2))
    
    # Availability Pie Chart
    ax3 = fig.add_subplot(gs[2, :])
    availability = results['availability']
    unavailability = 1 - availability
    
    sizes = [availability * 100, unavailability * 100]
    labels = [f'Up Time\n{availability*100:.2f}%', f'Down Time\n{unavailability*100:.2f}%']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.08, 0.08)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
           shadow=True, startangle=90, 
           textprops={'fontsize': 13, 'weight': 'bold'},
           wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True})
    
    ax3.set_title('System Availability', fontsize=14, fontweight='bold', pad=20)
    
    output_path = os.path.join(OUTPUT_DIR, 'system_repair_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    OUTPUT_DIR = ensure_output_dir('part4')

    results = run_system_monte_carlo_with_repair(components_data, Tc, DT, N_SIMS)
    
    visualize_system_timeline(results)
    print(f"Created timeline diagram in: {OUTPUT_DIR}")
    create_system_analysis_plots(results)
    print(f"Created analysis graph in:   {OUTPUT_DIR}")

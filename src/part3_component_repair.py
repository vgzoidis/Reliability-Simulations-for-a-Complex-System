import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add config parameters
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    COMPONENTS_DATA as components_data,
    Tc, DT, N_SIMS,
    ensure_output_dir)

# Executes N simulations with MTTR for a component, calculating:
# - Mean Time Between Failures
# - Mean Up Time
# - Mean Time To Repair
# - Availability
def run_monte_carlo_with_repair(component_name, mttf, duty_cycle, mttr, duration, dt, n_sims):
    print(f"\n{component_name} -> MTTF = {mttf}h, DC = {duty_cycle}, MTTR = {mttr}h, Tc = {duration}h")
    print(f"{'='*70}")
    
    all_failure_times = []
    all_repair_durations = []
    all_up_times = []
    all_total_up_time = []
    all_total_down_time = []
    total_failures = 0
    
    # Samples for graph
    sample_time = None
    sample_history = None
    
    for i in range(n_sims):
        time_axis, status_history, failure_times, repair_durations, up_times = \
            simulate_component_with_repair(mttf, duty_cycle, mttr, duration, dt)
        
        # Save first results
        if i == 0:
            sample_time = time_axis
            sample_history = status_history
        
        # Save the rest of the results
        all_failure_times.extend(failure_times)
        all_repair_durations.extend(repair_durations)
        all_up_times.extend(up_times)
        total_failures += len(failure_times)
        
        # Overall operational/fault time
        up_time = np.sum(status_history == 2) * dt
        down_time = np.sum(status_history == 0) * dt
        all_total_up_time.append(up_time)
        all_total_down_time.append(down_time)
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_sims} simulations...")
    
    # Calculate effective MTTF (needed for comparisons)
    if duty_cycle > 0:
        effective_mttf = mttf / duty_cycle
    else:
        effective_mttf = float('inf')
    
    # 1. Mean Time Between Failures
    if len(all_up_times) > 0:
        mtbf_exp = np.mean(all_up_times)
        mtbf_std = np.std(all_up_times)

        print(f"\nMean Time Between Failures:")
        print(f"Experimental:        {mtbf_exp:.4f} hours (σ = {mtbf_std:.4f})")
        print(f"Theoretical (MTTF/DC): {effective_mttf:.4f} hours")
        print(f"Relative error:      {abs(mtbf_exp - effective_mttf) / effective_mttf * 100:.2f}%")
    else:
        mtbf_exp = float('inf')
        mtbf_std = 0
        print("No failures detected")
    
    # 2. Mean Up Time
    mut_exp = mtbf_exp
    mut_std = mtbf_std
    print(f"\nMean Up Time:")
    if mut_exp != float('inf'):
        print(f"Experimental: {mut_exp:.4f} hours (σ = {mut_std:.4f})")
    else:
        print("No failures detected")
    
    # 3. Mean Time To Repair
    if len(all_repair_durations) > 0:
        mttr_exp = np.mean(all_repair_durations)
        mttr_std = np.std(all_repair_durations)
        print(f"\nMean Time To Repair:")
        print(f"Theoretical: {mttr:.4f} hours")
        if mttr_exp > 0:
            print(f"Experimental: {mttr_exp:.4f} hours (σ = {mttr_std:.4f})")
            print(f"Relative error: {abs(mttr_exp - mttr) / mttr * 100:.2f}%")
        else:
            print("No failures detected")
    else:
        mttr_exp = 0
        mttr_std = 0
    
    # 4. Availability (A = Total Up Time / Total Time)
    total_up = sum(all_total_up_time)
    total_down = sum(all_total_down_time)
    total_time = n_sims * duration

    # Theoretical availability (using effective_mttf calculated earlier)
    if effective_mttf != float('inf'):
        availability_theo = effective_mttf / (effective_mttf + mttr)
    else:
        availability_theo = 1.0

    availability_exp = total_up / total_time if total_time > 0 else 0
    print(f"\nAvailability:")
    print(f"Experimental: A = {availability_exp:.6f} ({availability_exp * 100:.2f}%)")
    print(f"Theoretical:  A = {availability_theo:.6f} ({availability_theo * 100:.2f}%)")
    print(f"Relative error:  {abs(availability_exp - availability_theo) / availability_theo * 100:.2f}%")
    
    # Average number of faults per simulation
    print(f"\nFault Statistics:")
    print(f"Συνολικός αριθμός βλαβών: {total_failures}")
    print(f"Μέσος αριθμός βλαβών ανά προσομοίωση: {(total_failures / n_sims):.2f}")
    print(f"Συνολικός αριθμός up periods: {len(all_up_times)}")
    
    print(f"\nΜέσος χρόνος λειτουργίας ανά προσομοίωση: {np.mean(all_total_up_time):.4f}h")
    print(f"Μέσος χρόνος βλάβης ανά προσομοίωση: {np.mean(all_total_down_time):.4f}h")
    
    return {
        'component': component_name,
        'mtbf_exp': mtbf_exp,
        'mtbf_theo': effective_mttf,
        'mtbf_std': mtbf_std,
        'mut_exp': mut_exp,
        'mut_std': mut_std,
        'mttr_exp': mttr_exp,
        'mttr_theo': mttr,
        'mttr_std': mttr_std,
        'availability_exp': availability_exp,
        'availability_theo': availability_theo,
        'total_failures': total_failures,
        'avg_failures': (total_failures / n_sims),
        'sample_time': sample_time,
        'sample_history': sample_history,
        'all_up_times': all_up_times,
        'all_repair_durations': all_repair_durations
    }

# Component states:
# - 2: Operational
# - 1: Non-operational due to duty cycle
# - 0: Under repair
def simulate_component_with_repair(mttf, duty_cycle, mttr, duration, dt):
    time_axis = np.arange(0, duration, dt)

    status_history = []
    lam = 1.0 / mttf
    current_status = 2
    repair_timer = 0.0 
    failure_times = []
    repair_durations = []
    up_times = []
    current_up_start = 0.0
    
    for dur, time in enumerate(time_axis):
        if current_status == 0: # Component under repair
            repair_timer -= dt
            
            if repair_timer <= 0: # Repair complete
                current_status = 2
                current_up_start = time
            status_history.append(0)
        else:
            # Check duty cycle
            is_active = np.random.rand() < duty_cycle
            
            if is_active: # Component is operational
                prob_fail = 1 - np.exp(-lam * dt)
                
                if np.random.rand() < prob_fail: # Fault
                    failure_times.append(time)
                    
                    # Calculate operation time from last repair
                    up_time = time - current_up_start
                    up_times.append(up_time)
                    
                    # Set random repair time
                    repair_duration = np.random.exponential(mttr)
                    repair_durations.append(repair_duration)
                    repair_timer = repair_duration
                    current_status = 0 
                    status_history.append(0)
                else: # Operational
                    current_status = 2
                    status_history.append(2)
            else: # Not working due to duty cycle
                current_status = 1
                status_history.append(1)
    
    if len(failure_times) == 0:    # If the component never failed, save the whole time as operational
        up_times.append(duration)
    elif current_status in [1, 2]: # If the component is working at the end, save the last period
        final_up_time = duration - current_up_start
        if final_up_time > 0:
            up_times.append(final_up_time)
    
    return (time_axis, np.array(status_history), failure_times, repair_durations, up_times)

def visualize_component_with_repair(results):
    comp_name = results['component']
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)
    
    # Sample Simulation
    ax1 = fig.add_subplot(gs[0, :])
    time = results['sample_time']
    history = results['sample_history']
    
    # Color according to state
    for state, color, label in [(2, 'green', 'Operational'), (1, 'yellow', 'Non-Operational (DC)'), (0, 'red', 'Under Repair')]:
        mask = (history == state)
        if np.any(mask):
            ax1.fill_between(time, 0, 1, where=mask, color=color, alpha=0.6, label=label, step='mid')
    
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('State', fontsize=11)
    ax1.set_title(f'Sample Simulation - {comp_name}', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.set_yticks([])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # MTBF
    ax2 = fig.add_subplot(gs[1, 0])
    if len(results['all_up_times']) > 0:
        ax2.hist(results['all_up_times'], bins=40, alpha=0.7, color='green', edgecolor='black', density=True)
        ax2.axvline(x=results['mtbf_exp'], color='darkgreen', linestyle='--', linewidth=2, label=f'MTBF={results["mtbf_exp"]:.2f}h')
        
        # Theoretical distribution
        x = np.linspace(0, max(results['all_up_times']), 100)
        if results['mtbf_exp'] != float('inf'):
            theoretical = (1/results['mtbf_exp']) * np.exp(-x/results['mtbf_exp'])
            ax2.plot(x, theoretical, 'r-', linewidth=2, label='Theoretical Exponential')
        
        ax2.set_xlabel('Operational Time (hours)', fontsize=10)
        ax2.set_ylabel('Probability Density', fontsize=10)
        ax2.set_title('MTBF Distribution', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # MTTR
    ax3 = fig.add_subplot(gs[1, 1])
    if len(results['all_repair_durations']) > 0:
        ax3.hist(results['all_repair_durations'], bins=40, alpha=0.7, color='red', edgecolor='black', density=True)
        ax3.axvline(x=results['mttr_exp'], color='darkred', linestyle='--', linewidth=2, label=f'MTTR={results["mttr_exp"]:.2f}h')
        
        # Theoretical distribution
        x = np.linspace(0, max(results['all_repair_durations']), 100)
        theoretical = (1/results['mttr_theo']) * np.exp(-x/results['mttr_theo'])
        ax3.plot(x, theoretical, 'b-', linewidth=2, label='Theoretical Exponential')
        
        ax3.set_xlabel('Repair Time (hours)', fontsize=10)
        ax3.set_ylabel('Probability Density', fontsize=10)
        ax3.set_title('MTTR Distribution', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Compare MTBF
    ax4 = fig.add_subplot(gs[2, 0])
    if results['mtbf_exp'] != float('inf'):
        categories = ['Πειραματική', 'Θεωρητική']
        values = [results['mtbf_exp'], results['mtbf_theo']]
        colors = ['steelblue', 'coral']
        bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}h', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_ylabel('MTBF (ώρες)', fontsize=10)
        ax4.set_title('Σύγκριση MTBF', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Compare availability
    ax5 = fig.add_subplot(gs[2, 1])
    categories = ['Πειραματική', 'Θεωρητική']
    values = [results['availability_exp'], results['availability_theo']]
    colors = ['green', 'lightgreen']
    bars = ax5.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'{val:.4f}\n({val*100:.2f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax5.set_ylabel('Availability', fontsize=10)
    ax5.set_title('Availability Comparison', fontsize=11, fontweight='bold')
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    output_path = os.path.join(OUTPUT_DIR, f'{comp_name}_repair_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

def create_summary_comparison(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    components = [r['component'] for r in all_results]
    x = np.arange(len(components))
    width = 0.35
    
    # MTBF Comparison
    ax1 = axes[0, 0]
    mtbf_exp = [r['mtbf_exp'] if r['mtbf_exp'] != float('inf') else 0 for r in all_results]
    mtbf_theo = [r['mtbf_theo'] if r['mtbf_theo'] != float('inf') else 0 for r in all_results]
    
    ax1.bar(x - width/2, mtbf_exp, width, label='Experimental', alpha=0.8)
    ax1.bar(x + width/2, mtbf_theo, width, label='Theoretical', alpha=0.8)
    ax1.set_xlabel('Component', fontsize=11)
    ax1.set_ylabel('MTBF (hours)', fontsize=11)
    ax1.set_title('MTBF Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # MTTR Comparison
    ax2 = axes[0, 1]
    mttr_exp = [r['mttr_exp'] for r in all_results]
    mttr_theo = [r['mttr_theo'] for r in all_results]
    
    ax2.bar(x - width/2, mttr_exp, width, label='Experimental', alpha=0.8, color='coral')
    ax2.bar(x + width/2, mttr_theo, width, label='Theoretical', alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Component', fontsize=11)
    ax2.set_ylabel('MTTR (hours)', fontsize=11)
    ax2.set_title('MTTR Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Availability Comparison
    ax3 = axes[1, 0]
    avail_exp = [r['availability_exp'] for r in all_results]
    avail_theo = [r['availability_theo'] for r in all_results]
    
    ax3.bar(x - width/2, avail_exp, width, label='Πειραματική', alpha=0.8, color='green')
    ax3.bar(x + width/2, avail_theo, width, label='Θεωρητική', alpha=0.8, color='lightgreen')
    ax3.set_xlabel('Εξάρτημα', fontsize=11)
    ax3.set_ylabel('Availability', fontsize=11)
    ax3.set_title('Σύγκριση Διαθεσιμότητας', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(components)
    ax3.set_ylim([0, 1.1])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Average Failures per Simulation
    ax4 = axes[1, 1]
    avg_failures = [r['avg_failures'] for r in all_results]
    
    ax4.bar(components, avg_failures, alpha=0.8, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Εξάρτημα', fontsize=11)
    ax4.set_ylabel('Μέσος Αριθμός Βλαβών', fontsize=11)
    ax4.set_title(f'Μέσος Αριθμός Βλαβών ανά Προσομοίωση (Tc={Tc}h)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'repair_summary_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nΣυγκεντρωτικό γράφημα αποθηκεύτηκε: {output_path}")

if __name__ == "__main__":
    OUTPUT_DIR = ensure_output_dir('part3')
    
    # Run simulation for all components
    all_results = []
    for comp_name, specs in components_data.items():
        results = run_monte_carlo_with_repair(
            comp_name,
            specs['MTTF'],
            specs['DC'],
            specs['MTTR'],
            Tc,
            DT,
            N_SIMS
        )
        all_results.append(results)
        
        # Create individual graphs for component results 
        visualize_component_with_repair(results)
    
    # Create results summary graph
    create_summary_comparison(all_results)
    
    print("\nΑνάλυση με επιδιόρθωση ολοκληρώθηκε επιτυχώς!")
    print(f"Όλα τα γραφήματα αποθηκεύτηκαν στον φάκελο: {OUTPUT_DIR}/\n")

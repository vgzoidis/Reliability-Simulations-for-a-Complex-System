import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import COMPONENTS_DATA, Tc, DT, N_SIMS, ensure_output_dir, print_header

# CORE
def simulate_component(mttf, duty_cycle, mttr, duration, dt):
    lam = 1.0 / mttf
    time_axis = np.arange(0, duration, dt)
    status_history = []
    repair_timer = 0.0
    current_status = 2  # 2=operational, 1=non-op (DC), 0=under repair
    
    up_times = []
    repair_durations = []
    current_up_start = 0.0
    
    for t in time_axis:
        if current_status == 0:  # Under repair
            repair_timer -= dt
            if repair_timer <= 0:
                current_status = 2
                current_up_start = t
            status_history.append(0)
        else:
            is_active = np.random.rand() < duty_cycle
            if is_active:
                if np.random.rand() < (1 - np.exp(-lam * dt)):
                    # Failure occurred
                    up_times.append(t - current_up_start)
                    repair_duration = np.random.exponential(mttr)
                    repair_durations.append(repair_duration)
                    repair_timer = repair_duration
                    current_status = 0
                    status_history.append(0)
                else:
                    current_status = 2
                    status_history.append(2)
            else:
                current_status = 1
                status_history.append(1)
    
    # Capture final up period
    if current_status in [1, 2] and len(up_times) == 0:
        up_times.append(duration)
    elif current_status in [1, 2]:
        up_times.append(duration - current_up_start)
    
    return time_axis, np.array(status_history), up_times, repair_durations

def simulate_system(components_db, duration, dt):
    time_axis = np.arange(0, duration, dt)
    comp_histories = {name: [] for name in components_db}
    current = {name: 2 for name in components_db}
    repair_timers = {name: 0.0 for name in components_db}
    
    system_history = []
    up_times, down_times = [], []
    current_state = None
    period_start = 0.0
    
    for t in time_axis:
        for name, specs in components_db.items():
            if current[name] == 0:  # Under repair
                repair_timers[name] -= dt
                if repair_timers[name] <= 0:
                    current[name] = 2
                comp_histories[name].append(0)
            else:
                is_active = np.random.rand() < specs['DC']
                if is_active:
                    lam = 1.0 / specs['MTTF']
                    if np.random.rand() < (1 - np.exp(-lam * dt)):
                        current[name] = 0
                        repair_timers[name] = np.random.exponential(specs['MTTR'])
                        comp_histories[name].append(0)
                    else:
                        current[name] = 2
                        comp_histories[name].append(2)
                else:
                    current[name] = 1
                    comp_histories[name].append(1)
        
        # System logic: C1 -> [C2||C3||C4] -> [C5||C6] -> C7
        c1_ok = current['C1'] != 0
        c7_ok = current['C7'] != 0
        block2_ok = current['C2'] != 0 or current['C3'] != 0 or current['C4'] != 0
        block3_ok = current['C5'] != 0 or current['C6'] != 0
        
        system_ok = 1 if (c1_ok and block2_ok and block3_ok and c7_ok) else 0
        system_history.append(system_ok)
        
        # Track up/down periods
        if current_state is None:
            current_state = system_ok
            period_start = t
        elif current_state != system_ok:
            period_duration = t - period_start
            if current_state == 1:
                up_times.append(period_duration)
            else:
                down_times.append(period_duration)
            current_state = system_ok
            period_start = t
    
    # Capture final period
    final_duration = duration - period_start
    if final_duration > 0:
        if current_state == 1:
            up_times.append(final_duration)
        else:
            down_times.append(final_duration)
    
    return time_axis, np.array(system_history), comp_histories, up_times, down_times

# ANALYSIS
def run_component_analysis(components_db, duration, dt, n_sims):
    print_header("COMPONENT ANALYSIS WITH REPAIR", "=", 70)
    results = []
    
    for comp_name, specs in components_db.items():
        mttf, dc, mttr = specs['MTTF'], specs['DC'], specs['MTTR']
        all_up_times, all_repair_times = [], []
        total_up, total_down = 0, 0
        
        for i in range(n_sims):
            _, hist, up_times, repair_times = simulate_component(mttf, dc, mttr, duration, dt)
            all_up_times.extend(up_times)
            all_repair_times.extend(repair_times)
            total_up += np.sum((hist == 2) | (hist == 1)) * dt
            total_down += np.sum(hist == 0) * dt
        
        # Calculate metrics
        MTBF_exp = np.mean(all_up_times) if all_up_times else float('inf')
        MTBF_theo = mttf / dc if dc > 0 else float('inf')
        MTTR_exp = np.mean(all_repair_times) if all_repair_times else 0
        A_exp = total_up / (n_sims * duration)
        A_theo = MTBF_theo / (MTBF_theo + mttr) if MTBF_theo != float('inf') else 1.0
        
        print(f"\n{comp_name}: MTTF={mttf}h, DC={dc}, MTTR={mttr}h")
        print(f"  MTBF: Exp={MTBF_exp:.2f}h, Theo={MTBF_theo:.2f}h")
        print(f"  MUT:  {MTBF_exp:.2f}h")
        print(f"  MTTR: Exp={MTTR_exp:.2f}h, Theo={mttr:.2f}h")
        print(f"  A:    Exp={A_exp:.4f} ({A_exp*100:.2f}%), Theo={A_theo:.4f} ({A_theo*100:.2f}%)")
        
        results.append({
            'component': comp_name,
            'MTBF_exp': MTBF_exp, 'MTBF_theo': MTBF_theo,
            'MUT': MTBF_exp,
            'MTTR_exp': MTTR_exp, 'MTTR_theo': mttr,
            'A_exp': A_exp, 'A_theo': A_theo
        })
    
    return results

def run_system_analysis(components_db, duration, dt, n_sims):
    print_header("SYSTEM ANALYSIS WITH REPAIR", "=", 70)
    
    all_up_times, all_down_times = [], []
    total_up, total_down = 0, 0
    sample_data = None
    
    for i in range(n_sims):
        time_axis, sys_hist, comp_hist, up_times, down_times = simulate_system(components_db, duration, dt)
        if i == 0:
            sample_data = (time_axis, sys_hist, comp_hist)
        all_up_times.extend(up_times)
        all_down_times.extend(down_times)
        total_up += np.sum(sys_hist == 1) * dt
        total_down += np.sum(sys_hist == 0) * dt
        
        if (i + 1) % 200 == 0:
            print(f"Completed {i+1}/{n_sims} simulations...")
    
    MTBF = np.mean(all_up_times) if all_up_times else float('inf')
    MUT = MTBF
    MTTR = np.mean(all_down_times) if all_down_times else 0
    A = total_up / (n_sims * duration)
    
    print(f"\nSystem MTBF: {MTBF:.4f}h")
    print(f"System MUT:  {MUT:.4f}h")
    print(f"System MTTR: {MTTR:.4f}h")
    print(f"System A:    {A:.4f} ({A*100:.2f}%)")
    
    return {
        'MTBF': MTBF, 'MUT': MUT, 'MTTR': MTTR, 'A': A,
        'all_up_times': all_up_times, 'all_down_times': all_down_times,
        'sample_data': sample_data
    }

# VISUALIZATION
def create_component_plots(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = [r['component'] for r in results]
    x = np.arange(len(components))
    width = 0.35
    
    # MTBF
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, [r['MTBF_exp'] for r in results], width, label='Experimental', alpha=0.8)
    ax1.bar(x + width/2, [r['MTBF_theo'] for r in results], width, label='Theoretical', alpha=0.8)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('MTBF (hours)')
    ax1.set_title('Mean Time Between Failures')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # MTTR
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, [r['MTTR_exp'] for r in results], width, label='Experimental', alpha=0.8, color='coral')
    ax2.bar(x + width/2, [r['MTTR_theo'] for r in results], width, label='Theoretical', alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Component')
    ax2.set_ylabel('MTTR (hours)')
    ax2.set_title('Mean Time To Repair')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Availability
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, [r['A_exp'] for r in results], width, label='Experimental', alpha=0.8, color='green')
    ax3.bar(x + width/2, [r['A_theo'] for r in results], width, label='Theoretical', alpha=0.8, color='lightgreen')
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Availability')
    ax3.set_title('Availability Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(components)
    ax3.set_ylim([0, 1.05])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # MUT
    ax4 = axes[1, 1]
    ax4.bar(components, [r['MUT'] for r in results], alpha=0.8, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Component')
    ax4.set_ylabel('MUT (hours)')
    ax4.set_title('Mean Up Time')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_repair.png'), dpi=150)
    plt.close()

def create_system_plots(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MTBF histogram
    ax1 = axes[0]
    if results['all_up_times']:
        ax1.hist(results['all_up_times'], bins=40, alpha=0.7, color='green', edgecolor='black', density=True)
        ax1.axvline(results['MTBF'], color='darkgreen', linestyle='--', linewidth=2, label=f'MTBF={results["MTBF"]:.2f}h')
        ax1.legend()
    ax1.set_xlabel('Up Time (hours)')
    ax1.set_ylabel('Density')
    ax1.set_title('System MTBF Distribution')
    ax1.grid(True, alpha=0.3)
    
    # MTTR histogram
    ax2 = axes[1]
    if results['all_down_times']:
        ax2.hist(results['all_down_times'], bins=40, alpha=0.7, color='red', edgecolor='black', density=True)
        ax2.axvline(results['MTTR'], color='darkred', linestyle='--', linewidth=2, label=f'MTTR={results["MTTR"]:.2f}h')
        ax2.legend()
    ax2.set_xlabel('Down Time (hours)')
    ax2.set_ylabel('Density')
    ax2.set_title('System MTTR Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Availability pie
    ax3 = axes[2]
    sizes = [results['A'] * 100, (1 - results['A']) * 100]
    labels = [f'Up Time\n{results["A"]*100:.2f}%', f'Down Time\n{(1-results["A"])*100:.2f}%']
    colors = ['#2ecc71', '#e74c3c']
    ax3.pie(sizes, labels=labels, colors=colors, explode=(0.05, 0.05),
            shadow=True, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
    ax3.set_title('System Availability')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_repair.png'), dpi=150)
    plt.close()

def create_timeline_plot(sample_data, output_dir):
    time, sys_hist, comp_hist = sample_data
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    color_map = {0: '#FF4444', 1: '#FFD700', 2: '#44FF44'}
    components = sorted(comp_hist.keys())
    all_items = ['SYSTEM'] + components
    n_items = len(all_items)
    row_height = 0.8
    
    # Draw system row
    y_pos = n_items - 1
    states = sys_hist
    changes = np.where(states[:-1] != states[1:])[0] + 1
    boundaries = np.concatenate(([0], changes, [len(states)]))
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        state = states[start_idx]
        start_time = time[start_idx]
        end_time = time[boundaries[i + 1]] if i < len(boundaries) - 2 else time[-1]
        width = end_time - start_time
        color = color_map[2] if state == 1 else color_map[0]
        rect = Rectangle((start_time, y_pos - row_height/2), width, row_height, facecolor=color, edgecolor='darkgray', linewidth=0.8)
        ax.add_patch(rect)
    
    # Draw component rows
    for idx, comp_name in enumerate(components):
        y_pos = n_items - idx - 2
        states = np.array(comp_hist[comp_name])
        changes = np.where(states[:-1] != states[1:])[0] + 1
        boundaries = np.concatenate(([0], changes, [len(states)]))
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            state = states[start_idx]
            start_time = time[start_idx]
            end_time = time[boundaries[i + 1]] if i < len(boundaries) - 2 else time[-1]
            width = end_time - start_time
            if width < 0.001:
                continue
            rect = Rectangle((start_time, y_pos - row_height/2), width, row_height, facecolor=color_map[state], edgecolor='none')
            ax.add_patch(rect)
    
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([-0.5, n_items - 0.5])
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Component / System', fontsize=12, fontweight='bold')
    ax.set_title('System and Component States Over Time (With Repair)', fontsize=14, fontweight='bold')
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(all_items[::-1], fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    legend_elements = [
        Patch(facecolor=color_map[2], edgecolor='black', label='Operational'),
        Patch(facecolor=color_map[1], edgecolor='black', label='Non-Op (DC)'),
        Patch(facecolor=color_map[0], edgecolor='black', label='Under Repair'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeline_with_repair.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    OUTPUT_DIR = ensure_output_dir('with_repair')
    
    # Component analysis with repair
    comp_results = run_component_analysis(COMPONENTS_DATA, Tc, DT, N_SIMS)
    create_component_plots(comp_results, OUTPUT_DIR)
    
    # System analysis with repair
    sys_results = run_system_analysis(COMPONENTS_DATA, Tc, DT, N_SIMS)
    create_system_plots(sys_results, OUTPUT_DIR)
    create_timeline_plot(sys_results['sample_data'], OUTPUT_DIR)
    
    print(f"\nPlots saved to: {OUTPUT_DIR}/")

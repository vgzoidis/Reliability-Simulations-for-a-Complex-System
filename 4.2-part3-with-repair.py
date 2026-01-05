import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Παράμετροι Εξαρτημάτων (Από Πίνακα 2) ---
components_data = {
    'C1': {'MTTF': 30, 'DC': 0.3, 'MTTR': 12},
    'C2': {'MTTF': 24, 'DC': 1.0, 'MTTR': 12},
    'C3': {'MTTF': 23, 'DC': 1.0, 'MTTR': 12},
    'C4': {'MTTF': 24, 'DC': 1.0, 'MTTR': 10},
    'C5': {'MTTF': 27, 'DC': 1.0, 'MTTR': 10},
    'C6': {'MTTF': 28, 'DC': 1.0, 'MTTR': 8},
    'C7': {'MTTF': 33, 'DC': 0.4, 'MTTR': 12},
}

# --- Παράμετροι Προσομοίωσης ---
Tc = 30.0        # Χρόνος μελέτης (ώρες)
DT = 0.01        # Χρονικό βήμα (ώρες)
N_SIMS = 1000    # Αριθμός προσομοιώσεων Monte Carlo


def simulate_component_with_repair(component_name, mttf, duty_cycle, mttr, duration, dt):
    """
    Προσομοιώνει ένα εξάρτημα με βλάβες και επιδιορθώσεις.
    
    Καταστάσεις:
    - 2: Λειτουργικό (Operational)
    - 1: Μη λειτουργικό λόγω duty cycle (Non-operational due to DC)
    - 0: Σε επιδιόρθωση (Under repair)
    
    Returns:
        time_axis: Χρονικός άξονας
        status_history: Ιστορικό κατάστασης
        failure_times: Λίστα με χρόνους βλαβών
        repair_times: Λίστα με διάρκειες επιδιόρθωσης
        up_times: Λίστα με διάρκειες λειτουργίας μεταξύ βλαβών
    """
    time_axis = np.arange(0, duration, dt)
    status_history = []
    
    # Παράμετροι βλάβης και επιδιόρθωσης
    lam = 1.0 / mttf
    
    # Καταστάσεις
    current_status = 2  # Αρχικά λειτουργικό
    repair_timer = 0.0  # Χρόνος που απομένει για επιδιόρθωση
    
    # Καταγραφή γεγονότων
    failure_times = []
    repair_durations = []
    up_times = []
    
    last_repair_complete_time = 0.0  # Χρόνος τελευταίας επιδιόρθωσης
    current_up_start = 0.0  # Χρόνος έναρξης τρέχουσας λειτουργικής περιόδου
    
    for i, t in enumerate(time_axis):
        if current_status == 0:
            # Το εξάρτημα είναι σε επιδιόρθωση
            repair_timer -= dt
            
            if repair_timer <= 0:
                # Ολοκληρώθηκε η επιδιόρθωση
                current_status = 2
                last_repair_complete_time = t
                current_up_start = t
            
            status_history.append(0)
            
        else:
            # Το εξάρτημα δεν είναι σε επιδιόρθωση
            # Έλεγχος duty cycle
            is_active = np.random.rand() < duty_cycle
            
            if is_active:
                # Το εξάρτημα είναι ενεργό - έλεγχος για βλάβη
                prob_fail = 1 - np.exp(-lam * dt)
                
                if np.random.rand() < prob_fail:
                    # Βλάβη!
                    failure_times.append(t)
                    
                    # Υπολογισμός χρόνου λειτουργίας από την τελευταία επιδιόρθωση
                    up_time = t - current_up_start
                    up_times.append(up_time)
                    
                    # Δημιουργία τυχαίου χρόνου επιδιόρθωσης (εκθετική κατανομή)
                    repair_duration = np.random.exponential(mttr)
                    repair_durations.append(repair_duration)
                    repair_timer = repair_duration
                    
                    current_status = 0  # Μετάβαση σε κατάσταση επιδιόρθωσης
                    status_history.append(0)
                else:
                    # Λειτουργικό
                    current_status = 2
                    status_history.append(2)
            else:
                # Μη λειτουργικό λόγω duty cycle
                current_status = 1
                status_history.append(1)
    
    # Αν το εξάρτημα δεν απέτυχε ποτέ, καταγράφουμε όλο το χρόνο ως up time
    if len(failure_times) == 0:
        up_times.append(duration)
    # Αν το εξάρτημα ήταν λειτουργικό στο τέλος, καταγράφουμε και την τελευταία up περίοδο
    elif current_status in [1, 2]:
        final_up_time = duration - current_up_start
        if final_up_time > 0:
            up_times.append(final_up_time)
    
    return (time_axis, np.array(status_history), failure_times, 
            repair_durations, up_times)


def run_monte_carlo_with_repair(component_name, mttf, duty_cycle, mttr, duration, dt, n_sims):
    """
    Εκτελεί N προσομοιώσεις με επιδιόρθωση και υπολογίζει:
    - MTBF (Mean Time Between Failures)
    - MUT (Mean Up Time)
    - MTTR (Mean Time To Repair) - πειραματική
    - A (Availability)
    """
    print(f"\n{'='*70}")
    print(f"Προσομοίωση με Επιδιόρθωση: {component_name}")
    print(f"MTTF = {mttf}h, DC = {duty_cycle}, MTTR = {mttr}h, Tc = {duration}h")
    print(f"{'='*70}")
    
    # Συλλογή δεδομένων από όλες τις προσομοιώσεις
    all_failure_times = []
    all_repair_durations = []
    all_up_times = []
    all_total_up_time = []
    all_total_down_time = []
    total_failures = 0
    
    # Αποθήκευση μίας προσομοίωσης για γράφημα
    sample_time = None
    sample_history = None
    
    for i in range(n_sims):
        time_axis, status_history, failure_times, repair_durations, up_times = \
            simulate_component_with_repair(component_name, mttf, duty_cycle, mttr, duration, dt)
        
        # Αποθήκευση πρώτης προσομοίωσης
        if i == 0:
            sample_time = time_axis
            sample_history = status_history
        
        # Συλλογή δεδομένων
        all_failure_times.extend(failure_times)
        all_repair_durations.extend(repair_durations)
        all_up_times.extend(up_times)
        total_failures += len(failure_times)
        
        # Υπολογισμός συνολικού χρόνου λειτουργίας και βλάβης
        up_time = np.sum(status_history == 2) * dt
        down_time = np.sum(status_history == 0) * dt
        
        all_total_up_time.append(up_time)
        all_total_down_time.append(down_time)
        
        if (i + 1) % 100 == 0:
            print(f"Ολοκληρώθηκαν {i + 1}/{n_sims} προσομοιώσεις...")
    
    # --- Υπολογισμός Μετρικών ---
    
    # 1. MTBF (Mean Time Between Failures)
    # Συμπεριλαμβάνεται ο χρόνος μέχρι την 1η βλάβη
    if len(all_up_times) > 0:
        mtbf_exp = np.mean(all_up_times)
        mtbf_std = np.std(all_up_times)
    else:
        mtbf_exp = float('inf')
        mtbf_std = 0
    
    # 2. MUT (Mean Up Time) - ίδιο με MTBF σε αυτή την περίπτωση
    mut_exp = mtbf_exp
    mut_std = mtbf_std
    
    # 3. MTTR (Mean Time To Repair) - πειραματική
    if len(all_repair_durations) > 0:
        mttr_exp = np.mean(all_repair_durations)
        mttr_std = np.std(all_repair_durations)
    else:
        mttr_exp = 0
        mttr_std = 0
    
    # 4. Availability (Διαθεσιμότητα)
    # A = Total Up Time / Total Time
    total_up = sum(all_total_up_time)
    total_down = sum(all_total_down_time)
    total_time = n_sims * duration
    
    availability_exp = total_up / total_time if total_time > 0 else 0
    
    # Θεωρητική διαθεσιμότητα: A = MTBF / (MTBF + MTTR)
    # Προσαρμογή για duty cycle
    if duty_cycle > 0:
        effective_mttf = mttf / duty_cycle
    else:
        effective_mttf = float('inf')
    
    if effective_mttf != float('inf'):
        availability_theo = effective_mttf / (effective_mttf + mttr)
    else:
        availability_theo = 1.0
    
    # Μέσος αριθμός βλαβών ανά προσομοίωση
    avg_failures_per_sim = total_failures / n_sims
    
    # --- Εμφάνιση Αποτελεσμάτων ---
    print(f"\n{'-'*70}")
    print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ για {component_name}:")
    print(f"{'-'*70}")
    
    print(f"\nΣτατιστικά Βλαβών:")
    print(f"  Συνολικός αριθμός βλαβών: {total_failures}")
    print(f"  Μέσος αριθμός βλαβών ανά προσομοίωση: {avg_failures_per_sim:.2f}")
    print(f"  Συνολικός αριθμός up periods: {len(all_up_times)}")
    
    print(f"\n1. MTBF (Mean Time Between Failures):")
    if mtbf_exp != float('inf'):
        print(f"   Πειραματική: {mtbf_exp:.4f} ώρες (σ = {mtbf_std:.4f})")
        print(f"   Θεωρητική (MTTF/DC): {effective_mttf:.4f} ώρες")
        print(f"   Σχετικό σφάλμα: {abs(mtbf_exp - effective_mttf) / effective_mttf * 100:.2f}%")
    else:
        print(f"   Δεν παρατηρήθηκαν βλάβες")
    
    print(f"\n2. MUT (Mean Up Time):")
    if mut_exp != float('inf'):
        print(f"   Πειραματική: {mut_exp:.4f} ώρες (σ = {mut_std:.4f})")
    else:
        print(f"   Δεν παρατηρήθηκαν βλάβες")
    
    print(f"\n3. MTTR (Mean Time To Repair):")
    print(f"   Θεωρητική: {mttr:.4f} ώρες")
    if mttr_exp > 0:
        print(f"   Πειραματική: {mttr_exp:.4f} ώρες (σ = {mttr_std:.4f})")
        print(f"   Σχετικό σφάλμα: {abs(mttr_exp - mttr) / mttr * 100:.2f}%")
    else:
        print(f"   Δεν παρατηρήθηκαν επιδιορθώσεις")
    
    print(f"\n4. AVAILABILITY (Διαθεσιμότητα):")
    print(f"   Πειραματική: A = {availability_exp:.6f} ({availability_exp * 100:.2f}%)")
    print(f"   Θεωρητική:   A = {availability_theo:.6f} ({availability_theo * 100:.2f}%)")
    print(f"   Σχετικό σφάλμα: {abs(availability_exp - availability_theo) / availability_theo * 100:.2f}%")
    
    print(f"\n   Μέσος χρόνος λειτουργίας ανά προσομοίωση: {np.mean(all_total_up_time):.4f}h")
    print(f"   Μέσος χρόνος βλάβης ανά προσομοίωση: {np.mean(all_total_down_time):.4f}h")
    
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
        'avg_failures': avg_failures_per_sim,
        'sample_time': sample_time,
        'sample_history': sample_history,
        'all_up_times': all_up_times,
        'all_repair_durations': all_repair_durations
    }


def visualize_component_with_repair(results):
    """
    Δημιουργεί γραφήματα για κάθε εξάρτημα
    """
    comp_name = results['component']
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)
    
    # --- 1. Δείγμα Προσομοίωσης ---
    ax1 = fig.add_subplot(gs[0, :])
    time = results['sample_time']
    history = results['sample_history']
    
    # Χρωματισμός ανάλογα με την κατάσταση
    for state, color, label in [(2, 'green', 'Λειτουργικό'), 
                                  (1, 'yellow', 'Μη Λειτουργικό (DC)'),
                                  (0, 'red', 'Επιδιόρθωση')]:
        mask = (history == state)
        if np.any(mask):
            ax1.fill_between(time, 0, 1, where=mask, color=color, 
                           alpha=0.6, label=label, step='mid')
    
    ax1.set_xlabel('Χρόνος (ώρες)', fontsize=11)
    ax1.set_ylabel('Κατάσταση', fontsize=11)
    ax1.set_title(f'Δείγμα Προσομοίωσης - {comp_name}', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.set_yticks([])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # --- 2. Κατανομή Up Times (MTBF) ---
    ax2 = fig.add_subplot(gs[1, 0])
    if len(results['all_up_times']) > 0:
        ax2.hist(results['all_up_times'], bins=40, alpha=0.7, 
                color='green', edgecolor='black', density=True)
        ax2.axvline(x=results['mtbf_exp'], color='darkgreen', linestyle='--', 
                   linewidth=2, label=f'MTBF={results["mtbf_exp"]:.2f}h')
        
        # Θεωρητική εκθετική κατανομή
        x = np.linspace(0, max(results['all_up_times']), 100)
        if results['mtbf_exp'] != float('inf'):
            theoretical = (1/results['mtbf_exp']) * np.exp(-x/results['mtbf_exp'])
            ax2.plot(x, theoretical, 'r-', linewidth=2, label='Θεωρητική Εκθετική')
        
        ax2.set_xlabel('Χρόνος Λειτουργίας (ώρες)', fontsize=10)
        ax2.set_ylabel('Πυκνότητα Πιθανότητας', fontsize=10)
        ax2.set_title('Κατανομή MTBF', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # --- 3. Κατανομή Repair Times (MTTR) ---
    ax3 = fig.add_subplot(gs[1, 1])
    if len(results['all_repair_durations']) > 0:
        ax3.hist(results['all_repair_durations'], bins=40, alpha=0.7, 
                color='red', edgecolor='black', density=True)
        ax3.axvline(x=results['mttr_exp'], color='darkred', linestyle='--', 
                   linewidth=2, label=f'MTTR={results["mttr_exp"]:.2f}h')
        
        # Θεωρητική εκθετική κατανομή
        x = np.linspace(0, max(results['all_repair_durations']), 100)
        theoretical = (1/results['mttr_theo']) * np.exp(-x/results['mttr_theo'])
        ax3.plot(x, theoretical, 'b-', linewidth=2, label='Θεωρητική Εκθετική')
        
        ax3.set_xlabel('Χρόνος Επιδιόρθωσης (ώρες)', fontsize=10)
        ax3.set_ylabel('Πυκνότητα Πιθανότητας', fontsize=10)
        ax3.set_title('Κατανομή MTTR', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # --- 4. Σύγκριση MTBF ---
    ax4 = fig.add_subplot(gs[2, 0])
    if results['mtbf_exp'] != float('inf'):
        categories = ['Πειραματική', 'Θεωρητική']
        values = [results['mtbf_exp'], results['mtbf_theo']]
        colors = ['steelblue', 'coral']
        bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}h', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_ylabel('MTBF (ώρες)', fontsize=10)
        ax4.set_title('Σύγκριση MTBF', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # --- 5. Σύγκριση Availability ---
    ax5 = fig.add_subplot(gs[2, 1])
    categories = ['Πειραματική', 'Θεωρητική']
    values = [results['availability_exp'], results['availability_theo']]
    colors = ['green', 'lightgreen']
    bars = ax5.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}\n({val*100:.2f}%)', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    ax5.set_ylabel('Availability', fontsize=10)
    ax5.set_title('Σύγκριση Διαθεσιμότητας', fontsize=11, fontweight='bold')
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f'{comp_name}_repair_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Γράφημα αποθηκεύτηκε: {comp_name}_repair_analysis.png")


def create_summary_comparison(all_results):
    """
    Δημιουργεί συγκεντρωτικά γραφήματα για όλα τα εξαρτήματα
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    components = [r['component'] for r in all_results]
    x = np.arange(len(components))
    width = 0.35
    
    # --- 1. MTBF Comparison ---
    ax1 = axes[0, 0]
    mtbf_exp = [r['mtbf_exp'] if r['mtbf_exp'] != float('inf') else 0 for r in all_results]
    mtbf_theo = [r['mtbf_theo'] if r['mtbf_theo'] != float('inf') else 0 for r in all_results]
    
    ax1.bar(x - width/2, mtbf_exp, width, label='Πειραματική', alpha=0.8)
    ax1.bar(x + width/2, mtbf_theo, width, label='Θεωρητική', alpha=0.8)
    ax1.set_xlabel('Εξάρτημα', fontsize=11)
    ax1.set_ylabel('MTBF (ώρες)', fontsize=11)
    ax1.set_title('Σύγκριση MTBF', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # --- 2. MTTR Comparison ---
    ax2 = axes[0, 1]
    mttr_exp = [r['mttr_exp'] for r in all_results]
    mttr_theo = [r['mttr_theo'] for r in all_results]
    
    ax2.bar(x - width/2, mttr_exp, width, label='Πειραματική', alpha=0.8, color='coral')
    ax2.bar(x + width/2, mttr_theo, width, label='Θεωρητική', alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Εξάρτημα', fontsize=11)
    ax2.set_ylabel('MTTR (ώρες)', fontsize=11)
    ax2.set_title('Σύγκριση MTTR', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # --- 3. Availability Comparison ---
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
    
    # --- 4. Average Failures per Simulation ---
    ax4 = axes[1, 1]
    avg_failures = [r['avg_failures'] for r in all_results]
    
    ax4.bar(components, avg_failures, alpha=0.8, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Εξάρτημα', fontsize=11)
    ax4.set_ylabel('Μέσος Αριθμός Βλαβών', fontsize=11)
    ax4.set_title(f'Μέσος Αριθμός Βλαβών ανά Προσομοίωση (Tc={Tc}h)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('repair_summary_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Συγκεντρωτικό γράφημα αποθηκεύτηκε: repair_summary_comparison.png")


def create_summary_table(all_results):
    """
    Δημιουργεί συγκεντρωτικό πίνακα αποτελεσμάτων
    """
    print("\n" + "="*100)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ - ΑΝΑΛΥΣΗ ΜΕ ΕΠΙΔΙΟΡΘΩΣΗ")
    print("="*100)
    
    # Header
    print(f"\n{'Εξάρτημα':<10} {'MTBF_exp':<12} {'MTBF_theo':<12} {'MTTR_exp':<12} "
          f"{'MTTR_theo':<12} {'A_exp':<10} {'A_theo':<10} {'Βλάβες':<10}")
    print("-"*100)
    
    for r in all_results:
        mtbf_exp_str = f"{r['mtbf_exp']:.4f}" if r['mtbf_exp'] != float('inf') else ">Tc"
        mtbf_theo_str = f"{r['mtbf_theo']:.4f}" if r['mtbf_theo'] != float('inf') else ">Tc"
        
        print(f"{r['component']:<10} {mtbf_exp_str:<12} {mtbf_theo_str:<12} "
              f"{r['mttr_exp']:<12.4f} {r['mttr_theo']:<12.4f} "
              f"{r['availability_exp']:<10.6f} {r['availability_theo']:<10.6f} "
              f"{r['avg_failures']:<10.2f}")
    
    print("="*100)
    print("\nΣημειώσεις:")
    print("  - MTBF: Mean Time Between Failures (περιλαμβάνει χρόνο μέχρι 1η βλάβη)")
    print("  - MTTR: Mean Time To Repair")
    print("  - A: Availability (Διαθεσιμότητα)")
    print("  - Βλάβες: Μέσος αριθμός βλαβών ανά προσομοίωση")
    print("="*100)


# --- ΚΥΡΙΑ ΕΚΤΕΛΕΣΗ ---
if __name__ == "__main__":
    print("="*70)
    print("ΠΡΟΣΟΜΟΙΩΣΗ ΜΕ ΕΠΙΔΙΟΡΘΩΣΗ - ΕΡΩΤΗΜΑ 4.2 (Μέρος 3)")
    print("="*70)
    print(f"Παράμετροι: Tc={Tc}h, dt={DT}h, N={N_SIMS} προσομοιώσεις")
    print("Υπολογισμός: MTBF, MUT, MTTR, Availability")
    
    all_results = []
    
    # Εκτέλεση προσομοιώσεων για κάθε εξάρτημα
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
        
        # Δημιουργία γραφημάτων για κάθε εξάρτημα
        visualize_component_with_repair(results)
    
    # Δημιουργία συγκεντρωτικών γραφημάτων
    create_summary_comparison(all_results)
    
    # Εμφάνιση συγκεντρωτικού πίνακα
    create_summary_table(all_results)
    
    print("\n✓ Ανάλυση με επιδιόρθωση ολοκληρώθηκε επιτυχώς!")
    print("✓ Όλα τα γραφήματα αποθηκεύτηκαν στον τρέχοντα φάκελο.\n")

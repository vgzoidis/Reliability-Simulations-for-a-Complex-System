import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import os

# --- Δημιουργία φακέλου για αποτελέσματα ---
OUTPUT_DIR = "part4_system_repair"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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


def simulate_system_with_repair(components_db, duration, dt):
    """
    Προσομοιώνει ολόκληρο το σύστημα με βλάβες και επιδιορθώσεις.
    
    Δομή Συστήματος: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7
    
    Καταστάσεις εξαρτημάτων:
    - 2: Λειτουργικό (Operational)
    - 1: Μη λειτουργικό λόγω duty cycle (Non-operational due to DC)
    - 0: Σε επιδιόρθωση (Under repair)
    
    Returns:
        time_axis: Χρονικός άξονας
        system_history: Ιστορικό κατάστασης συστήματος (1=up, 0=down)
        component_histories: Dict με ιστορικά όλων των εξαρτημάτων
        system_up_times: Λίστα με διάρκειες λειτουργίας συστήματος
        system_down_times: Λίστα με διάρκειες βλάβης συστήματος
    """
    time_axis = np.arange(0, duration, dt)
    
    # Αρχικοποίηση καταστάσεων
    current_status = {name: 2 for name in components_db}  # Όλα λειτουργικά
    repair_timers = {name: 0.0 for name in components_db}
    
    # Ιστορικά
    component_histories = {name: [] for name in components_db}
    system_history = []
    
    # Καταγραφή χρόνων λειτουργίας/βλάβης συστήματος
    system_up_times = []
    system_down_times = []
    
    current_system_state = None
    current_period_start = 0.0
    
    for i, t in enumerate(time_axis):
        # --- Ενημέρωση κατάστασης κάθε εξαρτήματος ---
        for name, specs in components_db.items():
            if current_status[name] == 0:
                # Το εξάρτημα είναι σε επιδιόρθωση
                repair_timers[name] -= dt
                
                if repair_timers[name] <= 0:
                    # Ολοκληρώθηκε η επιδιόρθωση
                    current_status[name] = 2
                
                component_histories[name].append(0)
                
            else:
                # Το εξάρτημα δεν είναι σε επιδιόρθωση
                # Έλεγχος duty cycle
                is_active = np.random.rand() < specs['DC']
                
                if is_active:
                    # Το εξάρτημα είναι ενεργό - έλεγχος για βλάβη
                    lam = 1.0 / specs['MTTF']
                    prob_fail = 1 - np.exp(-lam * dt)
                    
                    if np.random.rand() < prob_fail:
                        # Βλάβη!
                        current_status[name] = 0
                        # Δημιουργία τυχαίου χρόνου επιδιόρθωσης
                        repair_timers[name] = np.random.exponential(specs['MTTR'])
                        component_histories[name].append(0)
                    else:
                        # Λειτουργικό
                        current_status[name] = 2
                        component_histories[name].append(2)
                else:
                    # Μη λειτουργικό λόγω duty cycle
                    current_status[name] = 1
                    component_histories[name].append(1)
        
        # --- Υπολογισμός κατάστασης συστήματος ---
        # Δομή: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7
        # Το σύστημα λειτουργεί αν:
        # - C1 δεν είναι σε βλάβη (status != 0)
        # - Τουλάχιστον ένα από {C2, C3, C4} δεν είναι σε βλάβη
        # - Τουλάχιστον ένα από {C5, C6} δεν είναι σε βλάβη
        # - C7 δεν είναι σε βλάβη
        
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
        
        # --- Καταγραφή χρόνων up/down ---
        if current_system_state is None:
            # Πρώτο βήμα
            current_system_state = system_state
            current_period_start = t
        elif current_system_state != system_state:
            # Αλλαγή κατάστασης
            period_duration = t - current_period_start
            
            if current_system_state == 1:
                # Ολοκληρώθηκε περίοδος λειτουργίας
                system_up_times.append(period_duration)
            else:
                # Ολοκληρώθηκε περίοδος βλάβης
                system_down_times.append(period_duration)
            
            current_system_state = system_state
            current_period_start = t
    
    # Καταγραφή τελευταίας περιόδου
    final_period_duration = duration - current_period_start
    if final_period_duration > 0:
        if current_system_state == 1:
            system_up_times.append(final_period_duration)
        else:
            system_down_times.append(final_period_duration)
    
    return (time_axis, np.array(system_history), component_histories, 
            system_up_times, system_down_times)


def run_system_monte_carlo_with_repair(components_db, duration, dt, n_sims):
    """
    Εκτελεί N προσομοιώσεις για το σύστημα με επιδιόρθωση και υπολογίζει:
    - MTBF_system (Mean Time Between Failures)
    - MUT_system (Mean Up Time)
    - MTTR_system (Mean Time To Repair)
    - A_system (Availability)
    """
    print("="*70)
    print("ΠΡΟΣΟΜΟΙΩΣΗ ΣΥΣΤΗΜΑΤΟΣ ΜΕ ΕΠΙΔΙΟΡΘΩΣΗ")
    print("="*70)
    print(f"Δομή: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7")
    print(f"Παράμετροι: Tc={duration}h, dt={dt}h, N={n_sims} προσομοιώσεις")
    print("="*70)
    
    # Συλλογή δεδομένων από όλες τις προσομοιώσεις
    all_system_up_times = []
    all_system_down_times = []
    total_up_time = 0
    total_down_time = 0
    
    # Αποθήκευση μίας προσομοίωσης για γράφημα
    sample_time = None
    sample_system_history = None
    sample_component_histories = None
    
    for i in range(n_sims):
        time_axis, system_history, comp_histories, up_times, down_times = \
            simulate_system_with_repair(components_db, duration, dt)
        
        # Αποθήκευση πρώτης προσομοίωσης
        if i == 0:
            sample_time = time_axis
            sample_system_history = system_history
            sample_component_histories = comp_histories
        
        # Συλλογή δεδομένων
        all_system_up_times.extend(up_times)
        all_system_down_times.extend(down_times)
        
        # Συνολικός χρόνος λειτουργίας/βλάβης
        up_time = np.sum(system_history == 1) * dt
        down_time = np.sum(system_history == 0) * dt
        
        total_up_time += up_time
        total_down_time += down_time
        
        if (i + 1) % 100 == 0:
            print(f"Ολοκληρώθηκαν {i + 1}/{n_sims} προσομοιώσεις...")
    
    # --- Υπολογισμός Μετρικών ---
    
    print(f"\n{'='*70}")
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΣΥΣΤΗΜΑΤΟΣ")
    print(f"{'='*70}")
    
    # 1. MTBF (Mean Time Between Failures) = MUT σε αυτή την περίπτωση
    if len(all_system_up_times) > 0:
        mtbf_system = np.mean(all_system_up_times)
        mtbf_std = np.std(all_system_up_times)
        mtbf_median = np.median(all_system_up_times)
        
        print(f"\n1. MTBF_system (Mean Time Between Failures):")
        print(f"   Μέσος όρος:      {mtbf_system:.4f} ώρες")
        print(f"   Τυπική απόκλιση: {mtbf_std:.4f} ώρες")
        print(f"   Διάμεσος:        {mtbf_median:.4f} ώρες")
        print(f"   Ελάχιστος:       {min(all_system_up_times):.4f} ώρες")
        print(f"   Μέγιστος:        {max(all_system_up_times):.4f} ώρες")
        print(f"   Συνολικές περίοδοι λειτουργίας: {len(all_system_up_times)}")
    else:
        mtbf_system = float('inf')
        mtbf_std = 0
        print(f"\n1. MTBF_system:")
        print(f"   Δεν παρατηρήθηκαν βλάβες συστήματος!")
    
    # 2. MUT (Mean Up Time) - ίδιο με MTBF
    mut_system = mtbf_system
    print(f"\n2. MUT_system (Mean Up Time):")
    if mut_system != float('inf'):
        print(f"   {mut_system:.4f} ώρες (ίδιο με MTBF)")
    else:
        print(f"   > {duration} ώρες")
    
    # 3. MTTR (Mean Time To Repair)
    if len(all_system_down_times) > 0:
        mttr_system = np.mean(all_system_down_times)
        mttr_std = np.std(all_system_down_times)
        mttr_median = np.median(all_system_down_times)
        
        print(f"\n3. MTTR_system (Mean Time To Repair):")
        print(f"   Μέσος όρος:      {mttr_system:.4f} ώρες")
        print(f"   Τυπική απόκλιση: {mttr_std:.4f} ώρες")
        print(f"   Διάμεσος:        {mttr_median:.4f} ώρες")
        print(f"   Ελάχιστος:       {min(all_system_down_times):.4f} ώρες")
        print(f"   Μέγιστος:        {max(all_system_down_times):.4f} ώρες")
        print(f"   Συνολικές περίοδοι βλάβης: {len(all_system_down_times)}")
    else:
        mttr_system = 0
        mttr_std = 0
        print(f"\n3. MTTR_system:")
        print(f"   Δεν παρατηρήθηκαν βλάβες συστήματος!")
    
    # 4. Availability (Διαθεσιμότητα)
    total_time = n_sims * duration
    availability_system = total_up_time / total_time if total_time > 0 else 0
    
    print(f"\n4. AVAILABILITY_system (Διαθεσιμότητα):")
    print(f"   A = {availability_system:.6f} ({availability_system * 100:.2f}%)")
    print(f"   Μέσος χρόνος λειτουργίας ανά προσομοίωση: {total_up_time/n_sims:.4f}h")
    print(f"   Μέσος χρόνος βλάβης ανά προσομοίωση:      {total_down_time/n_sims:.4f}h")
    
    # Θεωρητική διαθεσιμότητα (αν γνωρίζουμε MTBF και MTTR)
    if mtbf_system != float('inf') and mtbf_system > 0:
        availability_theo = mtbf_system / (mtbf_system + mttr_system)
        print(f"   A_θεωρητική (από MTBF/(MTBF+MTTR)): {availability_theo:.6f} ({availability_theo * 100:.2f}%)")
        print(f"   Σχετικό σφάλμα: {abs(availability_system - availability_theo) / availability_theo * 100:.2f}%")
    
    # Μέσος αριθμός μεταβάσεων (βλαβών)
    avg_failures = len(all_system_up_times) / n_sims
    print(f"\n5. Στατιστικά Βλαβών:")
    print(f"   Συνολικός αριθμός βλαβών συστήματος: {len(all_system_up_times)}")
    print(f"   Μέσος αριθμός βλαβών ανά προσομοίωση: {avg_failures:.2f}")
    
    if mtbf_system != float('inf') and mtbf_system > 0:
        failure_rate = 1 / mtbf_system
        print(f"   Ρυθμός αποτυχίας λ_system: {failure_rate:.6f} βλάβες/ώρα")
    
    print(f"{'='*70}\n")
    
    return {
        'mtbf': mtbf_system,
        'mtbf_std': mtbf_std,
        'mut': mut_system,
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


def visualize_system_timeline(results):
    """
    Ζητούμενο 4: Διάγραμμα με την κατάσταση του συστήματος και κάθε εξαρτήματος
    σε ξεχωριστές γραμμές με οριζόντιο άξονα τον χρόνο.
    """
    time = results['sample_time']
    system_history = results['sample_system_history']
    comp_histories = results['sample_component_histories']
    
    # Δημιουργία μεγάλου γραφήματος
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Χρώματα για καταστάσεις
    color_map = {
        0: '#FF4444',  # Βλάβη/Επιδιόρθωση - Κόκκινο
        1: '#FFD700',  # Μη λειτουργικό (DC) - Χρυσό
        2: '#44FF44',  # Λειτουργικό - Πράσινο
    }
    
    label_map = {
        0: 'Επιδιόρθωση',
        1: 'Μη Λειτουργικό (DC)',
        2: 'Λειτουργικό',
    }
    
    # Ύψος κάθε γραμμής
    row_height = 0.8
    
    # Λίστα εξαρτημάτων + σύστημα
    components = list(comp_histories.keys())
    all_items = components + ['ΣΥΣΤΗΜΑ']
    n_items = len(all_items)
    
    # --- Σχεδίαση καταστάσεων για κάθε εξάρτημα ---
    for idx, comp_name in enumerate(components):
        y_position = n_items - idx - 1
        states = np.array(comp_histories[comp_name])
        
        # Βρίσκουμε όλες τις αλλαγές κατάστασης
        changes = np.where(states[:-1] != states[1:])[0] + 1
        boundaries = np.concatenate(([0], changes, [len(states)]))
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            state = states[start_idx]
            
            start_time = time[start_idx]
            end_time = time[end_idx - 1] if end_idx < len(time) else time[-1]
            width = end_time - start_time
            
            rect = Rectangle((start_time, y_position - row_height/2), 
                           width, row_height,
                           facecolor=color_map[state], 
                           edgecolor='black', 
                           linewidth=0.5)
            ax.add_patch(rect)
    
    # --- Σχεδίαση κατάστασης συστήματος ---
    y_position = 0
    states = system_history
    
    # Βρίσκουμε αλλαγές κατάστασης
    changes = np.where(states[:-1] != states[1:])[0] + 1
    boundaries = np.concatenate(([0], changes, [len(states)]))
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        state = states[start_idx]
        
        start_time = time[start_idx]
        end_time = time[end_idx - 1] if end_idx < len(time) else time[-1]
        width = end_time - start_time
        
        # Για το σύστημα: 1=λειτουργεί (πράσινο), 0=βλάβη (κόκκκινο)
        color = '#44FF44' if state == 1 else '#FF4444'
        
        rect = Rectangle((start_time, y_position - row_height/2), 
                       width, row_height,
                       facecolor=color, 
                       edgecolor='black', 
                       linewidth=1.5)
        ax.add_patch(rect)
    
    # --- Ρυθμίσεις γραφήματος ---
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([-0.5, n_items - 0.5])
    
    ax.set_xlabel('Χρόνος Λειτουργίας (ώρες)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Εξάρτημα / Σύστημα', fontsize=13, fontweight='bold')
    ax.set_title('Κατάσταση Συστήματος και Εξαρτημάτων σε Συνάρτηση με τον Χρόνο', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Y-axis labels
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(all_items, fontsize=11)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Έντονη γραμμή για το σύστημα
    ax.axhline(y=0, color='black', linewidth=2, linestyle='-')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#44FF44', edgecolor='black', label='Λειτουργικό'),
        Patch(facecolor='#FFD700', edgecolor='black', label='Μη Λειτουργικό (DC)'),
        Patch(facecolor='#FF4444', edgecolor='black', label='Βλάβη/Επιδιόρθωση'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
             framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'system_component_timeline.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Γράφημα χρονοσειράς αποθηκεύτηκε: {output_path}")


def create_system_analysis_plots(results):
    """
    Δημιουργεί συμπληρωματικά γραφήματα ανάλυσης συστήματος
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)
    
    # --- 1. Κατανομή Up Times (MTBF) ---
    ax1 = fig.add_subplot(gs[0, 0])
    if len(results['all_up_times']) > 0:
        ax1.hist(results['all_up_times'], bins=50, alpha=0.7, 
                color='green', edgecolor='black', density=True)
        ax1.axvline(x=results['mtbf'], color='darkgreen', linestyle='--', 
                   linewidth=2.5, label=f'MTBF={results["mtbf"]:.3f}h')
        
        # Θεωρητική εκθετική κατανομή
        if results['mtbf'] != float('inf'):
            x = np.linspace(0, max(results['all_up_times']), 200)
            theoretical = (1/results['mtbf']) * np.exp(-x/results['mtbf'])
            ax1.plot(x, theoretical, 'r-', linewidth=2, label='Θεωρητική Εκθετική')
        
        ax1.set_xlabel('Χρόνος Λειτουργίας Συστήματος (ώρες)', fontsize=11)
        ax1.set_ylabel('Πυκνότητα Πιθανότητας', fontsize=11)
        ax1.set_title('Κατανομή MTBF Συστήματος', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # --- 2. Κατανομή Down Times (MTTR) ---
    ax2 = fig.add_subplot(gs[0, 1])
    if len(results['all_down_times']) > 0:
        ax2.hist(results['all_down_times'], bins=50, alpha=0.7, 
                color='red', edgecolor='black', density=True)
        ax2.axvline(x=results['mttr'], color='darkred', linestyle='--', 
                   linewidth=2.5, label=f'MTTR={results["mttr"]:.3f}h')
        
        # Θεωρητική εκθετική κατανομή (αν ισχύει)
        x = np.linspace(0, max(results['all_down_times']), 200)
        # Για το σύστημα, η κατανομή down times μπορεί να μην είναι εκθετική
        
        ax2.set_xlabel('Χρόνος Βλάβης Συστήματος (ώρες)', fontsize=11)
        ax2.set_ylabel('Πυκνότητα Πιθανότητας', fontsize=11)
        ax2.set_title('Κατανομή MTTR Συστήματος', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # --- 3. Pie Chart - Διαθεσιμότητα ---
    ax3 = fig.add_subplot(gs[1, 0])
    availability = results['availability']
    unavailability = 1 - availability
    
    sizes = [availability * 100, unavailability * 100]
    labels = [f'Up Time\n{availability*100:.2f}%', f'Down Time\n{unavailability*100:.2f}%']
    colors = ['#44FF44', '#FF4444']
    explode = (0.05, 0.05)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='', shadow=True, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
    ax3.set_title(f'Διαθεσιμότητα Συστήματος\nA = {availability:.6f}', 
                 fontsize=12, fontweight='bold')
    
    # --- 4. Μετρικές Συστήματος ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    metrics_text = f"""
    ΜΕΤΡΙΚΕΣ ΣΥΣΤΗΜΑΤΟΣ
    {'='*40}
    
    MTBF (Mean Time Between Failures):
        {results['mtbf']:.4f} ώρες
        (σ = {results['mtbf_std']:.4f}h)
    
    MUT (Mean Up Time):
        {results['mut']:.4f} ώρες
    
    MTTR (Mean Time To Repair):
        {results['mttr']:.4f} ώρες
        (σ = {results['mttr_std']:.4f}h)
    
    Availability (Διαθεσιμότητα):
        {results['availability']:.6f}
        ({results['availability']*100:.2f}%)
    
    Μέσος αριθμός βλαβών:
        {results['avg_failures']:.2f} ανά προσομοίωση
    
    Ρυθμός αποτυχίας:
        λ = {1/results['mtbf'] if results['mtbf'] != float('inf') else 0:.6f} βλάβες/ώρα
    """
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- 5. Σύγκριση με Θεωρητικά (αν διαθέσιμα) ---
    ax5 = fig.add_subplot(gs[2, :])
    
    if results['mtbf'] != float('inf'):
        metrics = ['MTBF\n(ώρες)', 'MTTR\n(ώρες)', 'Availability']
        experimental = [results['mtbf'], results['mttr'], results['availability']]
        
        # Θεωρητική availability από MTBF/(MTBF+MTTR)
        theo_availability = results['mtbf'] / (results['mtbf'] + results['mttr'])
        theoretical = [results['mtbf'], results['mttr'], theo_availability]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, experimental, width, 
                       label='Πειραματική', alpha=0.8, color='steelblue')
        bars2 = ax5.bar(x + width/2, theoretical, width, 
                       label='Θεωρητική (από δεδομένα)', alpha=0.8, color='coral')
        
        ax5.set_ylabel('Τιμή', fontsize=11)
        ax5.set_title('Σύγκριση Μετρικών Συστήματος', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Προσθήκη τιμών πάνω από τις μπάρες
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    output_path = os.path.join(OUTPUT_DIR, 'system_repair_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Γράφημα ανάλυσης συστήματος αποθηκεύτηκε: {output_path}")


def create_summary_table(results):
    """
    Συγκεντρωτικός πίνακας αποτελεσμάτων συστήματος
    """
    print("\n" + "="*70)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΥΣΤΗΜΑΤΟΣ")
    print("="*70)
    
    print(f"\n{'Μέγεθος':<35} {'Τιμή':<25}")
    print("-"*70)
    
    if results['mtbf'] != float('inf'):
        print(f"{'MTBF_system (ώρες)':<35} {results['mtbf']:.4f} (σ={results['mtbf_std']:.4f})")
    else:
        print(f"{'MTBF_system (ώρες)':<35} {'> ' + str(Tc):<25}")
    
    if results['mut'] != float('inf'):
        print(f"{'MUT_system (ώρες)':<35} {results['mut']:.4f}")
    else:
        print(f"{'MUT_system (ώρες)':<35} {'> ' + str(Tc):<25}")
    
    print(f"{'MTTR_system (ώρες)':<35} {results['mttr']:.4f} (σ={results['mttr_std']:.4f})")
    print(f"{'Availability A_system':<35} {results['availability']:.6f} ({results['availability']*100:.2f}%)")
    
    if results['mtbf'] != float('inf'):
        failure_rate = 1 / results['mtbf']
        print(f"{'Ρυθμός αποτυχίας λ (βλάβες/ώρα)':<35} {failure_rate:.6f}")
    
    print(f"{'Μέσος αριθμός βλαβών':<35} {results['avg_failures']:.2f} ανά προσομοίωση")
    print(f"{'Συνολικές περίοδοι λειτουργίας':<35} {len(results['all_up_times'])}")
    print(f"{'Συνολικές περίοδοι βλάβης':<35} {len(results['all_down_times'])}")
    
    print("="*70)


# --- ΚΥΡΙΑ ΕΚΤΕΛΕΣΗ ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ΑΝΑΛΥΣΗ ΣΥΣΤΗΜΑΤΟΣ ΜΕ ΕΠΙΔΙΟΡΘΩΣΗ - ΕΡΩΤΗΜΑ 4.2 (Μέρος 4)")
    print("="*70)
    print("\nΜπλοκ Διάγραμμα: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7")
    print("Υπολογισμός: MTBF, MUT, MTTR, Availability για το σύστημα\n")
    
    # Εκτέλεση προσομοιώσεων Monte Carlo
    results = run_system_monte_carlo_with_repair(components_data, Tc, DT, N_SIMS)
    
    # Ζητούμενο 4: Διάγραμμα με όλες τις καταστάσεις
    print("\nΔημιουργία διαγράμματος χρονοσειράς...")
    visualize_system_timeline(results)
    
    # Δημιουργία συμπληρωματικών γραφημάτων
    print("\nΔημιουργία γραφημάτων ανάλυσης...")
    create_system_analysis_plots(results)
    
    # Εμφάνιση συγκεντρωτικού πίνακα
    create_summary_table(results)
    
    print("\n" + "="*70)
    print("ΕΠΙΠΛΕΟΝ ΠΛΗΡΟΦΟΡΙΕΣ")
    print("="*70)
    print("\nΤα αποτελέσματα περιλαμβάνουν:")
    print("  1. Πειραματικές τιμές MTBF, MUT, MTTR και A για το σύστημα")
    print("  2. Διάγραμμα με την κατάσταση κάθε εξαρτήματος και του συστήματος")
    print("  3. Στατιστική ανάλυση των χρόνων λειτουργίας και βλάβης")
    print("  4. Σύγκριση με θεωρητικές προβλέψεις")
    
    print("\n✓ Ανάλυση ολοκληρώθηκε επιτυχώς!")
    print(f"✓ Όλα τα γραφήματα αποθηκεύτηκαν στον φάκελο: {OUTPUT_DIR}/")
    print("="*70 + "\n")

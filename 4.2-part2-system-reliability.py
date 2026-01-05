import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Παράμετροι Εξαρτημάτων (Από Πίνακα 2) ---
components_data = {
    'C1': {'MTTF': 30, 'DC': 0.3, 'MTTR': 12},
    'C2': {'MTTF': 27, 'DC': 1.0, 'MTTR': 12},
    'C3': {'MTTF': 27, 'DC': 1.0, 'MTTR': 12},
    'C4': {'MTTF': 24, 'DC': 1.0, 'MTTR': 10},
    'C5': {'MTTF': 25, 'DC': 1.0, 'MTTR': 10},
    'C6': {'MTTF': 15, 'DC': 1.0, 'MTTR': 8},
    'C7': {'MTTF': 31, 'DC': 0.4, 'MTTR': 12},
}

# --- Παράμετροι Προσομοίωσης ---
Ts = 30.0        # Χρόνος μελέτης συστήματος (ώρες)
DT = 0.01        # Χρονικό βήμα (ώρες)
N_SIMS = 1000    # Αριθμός προσομοιώσεων Monte Carlo


def simulate_system_reliability(components_db, duration, dt):
    """
    Προσομοιώνει το σύστημα σύμφωνα με το μπλοκ διάγραμμα αξιοπιστίας.
    
    Δομή Συστήματος: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7
    
    Returns:
        time_axis: Χρονικός άξονας
        system_history: Ιστορικό κατάστασης συστήματος (1=λειτουργεί, 0=βλάβη)
        component_states: Ιστορικό καταστάσεων όλων των εξαρτημάτων
        system_failure_time: Χρόνος πρώτης βλάβης συστήματος (ή None)
    """
    time_axis = np.arange(0, duration, dt)
    
    # Αρχικοποίηση καταστάσεων εξαρτημάτων
    # 1 = Λειτουργικό, 0 = Μη λειτουργικό (DC), -1 = Βλάβη
    component_states = {name: [] for name in components_db}
    current_status = {name: 1 for name in components_db}
    failed_components = {name: False for name in components_db}
    
    system_history = []
    system_failure_time = None
    system_failed = False
    
    for t in time_axis:
        # --- Ενημέρωση κατάστασης κάθε εξαρτήματος ---
        for name, specs in components_db.items():
            if failed_components[name]:
                # Το εξάρτημα έχει ήδη αποτύχει
                current_status[name] = -1
            else:
                # Έλεγχος duty cycle
                is_active = np.random.rand() < specs['DC']
                
                if is_active:
                    # Το εξάρτημα είναι ενεργό - έλεγχος για βλάβη
                    lam = 1.0 / specs['MTTF']
                    prob_fail = 1 - np.exp(-lam * dt)
                    
                    if np.random.rand() < prob_fail:
                        # Βλάβη εξαρτήματος!
                        current_status[name] = -1
                        failed_components[name] = True
                    else:
                        current_status[name] = 1
                else:
                    # Μη λειτουργικό λόγω duty cycle
                    current_status[name] = 0
            
            component_states[name].append(current_status[name])
        
        # --- Υπολογισμός κατάστασης συστήματος ---
        # Δομή: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7
        # Το σύστημα λειτουργεί αν όλα τα εξαρτήματα είναι είτε λειτουργικά (1) 
        # είτε μη λειτουργικά λόγω DC (0), αλλά ΟΧΙ σε βλάβη (-1)
        
        # Για σειριακά στοιχεία (C1, C7): πρέπει να ΜΗΝ είναι σε βλάβη
        c1_ok = (current_status['C1'] != -1)
        c7_ok = (current_status['C7'] != -1)
        
        # Για παράλληλα μπλοκ: τουλάχιστον ένα πρέπει να ΜΗΝ είναι σε βλάβη
        block2_ok = (current_status['C2'] != -1) or \
                    (current_status['C3'] != -1) or \
                    (current_status['C4'] != -1)
        
        block3_ok = (current_status['C5'] != -1) or \
                    (current_status['C6'] != -1)
        
        # Το σύστημα λειτουργεί αν όλα τα μπλοκ λειτουργούν
        system_operational = c1_ok and block2_ok and block3_ok and c7_ok
        
        system_history.append(1 if system_operational else 0)
        
        # Καταγραφή χρόνου πρώτης βλάβης συστήματος
        if not system_operational and not system_failed:
            system_failure_time = t
            system_failed = True
    
    return time_axis, np.array(system_history), component_states, system_failure_time


def run_system_monte_carlo(components_db, duration, dt, n_sims):
    """
    Εκτελεί N προσομοιώσεις για το σύστημα και υπολογίζει:
    - Πειραματικό ρυθμό αποτυχίας λ_system
    - Πειραματική αξιοπιστία R_system(Ts)
    - Πειραματικό MTTF_system
    """
    print("="*70)
    print("ΠΡΟΣΟΜΟΙΩΣΗ ΑΞΙΟΠΙΣΤΙΑΣ ΣΥΣΤΗΜΑΤΟΣ")
    print("="*70)
    print(f"Δομή: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7")
    print(f"Παράμετροι: Ts={duration}h, dt={dt}h, N={n_sims} προσομοιώσεις")
    print("="*70)
    
    system_failure_times = []
    system_failures_count = 0
    
    # Αποθήκευση ενός δείγματος για γραφική παράσταση
    sample_time = None
    sample_history = None
    sample_components = None
    
    for i in range(n_sims):
        time_axis, system_history, comp_states, failure_time = simulate_system_reliability(
            components_db, duration, dt
        )
        
        # Αποθήκευση πρώτης προσομοίωσης για γράφημα
        if i == 0:
            sample_time = time_axis
            sample_history = system_history
            sample_components = comp_states
        
        if failure_time is not None:
            system_failure_times.append(failure_time)
            system_failures_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"Ολοκληρώθηκαν {i + 1}/{n_sims} προσομοιώσεις...")
    
    print(f"\n{'='*70}")
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΡΟΣΟΜΟΙΩΣΗΣ ΣΥΣΤΗΜΑΤΟΣ")
    print(f"{'='*70}")
    
    # --- 1. ΑΞΙΟΠΙΣΤΙΑ R_system(Ts) ---
    reliability_system_exp = (n_sims - system_failures_count) / n_sims
    
    print(f"\n1. ΑΞΙΟΠΙΣΤΙΑ R_system(Ts={duration}h):")
    print(f"   Πειραματική: R = {reliability_system_exp:.6f}")
    print(f"   Αριθμός βλαβών: {system_failures_count}/{n_sims}")
    print(f"   Ποσοστό επιτυχίας: {reliability_system_exp * 100:.2f}%")
    
    # --- 2. MTTF_system ---
    if len(system_failure_times) > 0:
        mttf_system_exp = np.mean(system_failure_times)
        mttf_std = np.std(system_failure_times)
        mttf_median = np.median(system_failure_times)
        
        print(f"\n2. MTTF_system (Mean Time To Failure):")
        print(f"   Μέσος όρος:    {mttf_system_exp:.4f} ώρες")
        print(f"   Τυπική απόκλιση: {mttf_std:.4f} ώρες")
        print(f"   Διάμεσος:      {mttf_median:.4f} ώρες")
        print(f"   Ελάχιστος:     {min(system_failure_times):.4f} ώρες")
        print(f"   Μέγιστος:      {max(system_failure_times):.4f} ώρες")
    else:
        mttf_system_exp = float('inf')
        print(f"\n2. MTTF_system:")
        print(f"   Δεν παρατηρήθηκε καμία βλάβη στις {n_sims} προσομοιώσεις!")
        print(f"   MTTF > {duration} ώρες")
    
    # --- 3. ΡΥΘΜΟΣ ΑΠΟΤΥΧΙΑΣ λ_system ---
    # Μέθοδος 1: Από το MTTF
    if mttf_system_exp != float('inf'):
        lambda_system_method1 = 1.0 / mttf_system_exp
    else:
        lambda_system_method1 = 0
    
    # Μέθοδος 2: Από την αξιοπιστία R(Ts)
    if reliability_system_exp > 0:
        lambda_system_method2 = -np.log(reliability_system_exp) / duration
    else:
        lambda_system_method2 = float('inf')
    
    print(f"\n3. ΡΥΘΜΟΣ ΑΠΟΤΥΧΙΑΣ λ_system:")
    print(f"   Μέθοδος 1 (από MTTF):  λ = {lambda_system_method1:.6f} βλάβες/ώρα")
    print(f"   Μέθοδος 2 (από R):     λ = {lambda_system_method2:.6f} βλάβες/ώρα")
    
    # --- 4. Θεωρητικά Αποτελέσματα για Σύγκριση ---
    print(f"\n{'='*70}")
    print("ΘΕΩΡΗΤΙΚΟΙ ΥΠΟΛΟΓΙΣΜΟΙ (για σύγκριση)")
    print(f"{'='*70}")
    
    # Για κάθε εξάρτημα: R_i(t) = exp(-λ_i * DC_i * t)
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
    
    # Αξιοπιστία παράλληλων μπλοκ:
    # R_parallel = 1 - (1-R1)(1-R2)...(1-Rn)
    R_block2 = 1 - (1-R_C2)*(1-R_C3)*(1-R_C4)  # C2, C3, C4 σε παράλληλη
    R_block3 = 1 - (1-R_C5)*(1-R_C6)           # C5, C6 σε παράλληλη
    
    # Αξιοπιστία συστήματος (σειριακή σύνδεση των μπλοκ):
    R_system_theo = R_C1 * R_block2 * R_block3 * R_C7
    
    print(f"\nΑξιοπιστίες εξαρτημάτων στο Ts={duration}h:")
    print(f"  R_C1 = {R_C1:.6f}")
    print(f"  R_C2 = {R_C2:.6f}, R_C3 = {R_C3:.6f}, R_C4 = {R_C4:.6f}")
    print(f"  R_C5 = {R_C5:.6f}, R_C6 = {R_C6:.6f}")
    print(f"  R_C7 = {R_C7:.6f}")
    
    print(f"\nΑξιοπιστίες μπλοκ:")
    print(f"  R_block2 (C2‖C3‖C4) = {R_block2:.6f}")
    print(f"  R_block3 (C5‖C6)    = {R_block3:.6f}")
    
    print(f"\nΑξιοπιστία συστήματος:")
    print(f"  Θεωρητική:   R_system = {R_system_theo:.6f}")
    print(f"  Πειραματική: R_system = {reliability_system_exp:.6f}")
    print(f"  Σχετικό σφάλμα: {abs(R_system_theo - reliability_system_exp) / R_system_theo * 100:.2f}%")
    
    if R_system_theo > 0:
        lambda_system_theo = -np.log(R_system_theo) / duration
        print(f"\n  Θεωρητικό λ_system = {lambda_system_theo:.6f} βλάβες/ώρα")
    
    print(f"{'='*70}\n")
    
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


def visualize_system_simulation(results):
    """
    Δημιουργεί γραφήματα για την ανάλυση του συστήματος
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # --- 1. Κατάσταση Συστήματος (Δείγμα) ---
    ax1 = fig.add_subplot(gs[0, :])
    time = results['sample_time']
    history = results['sample_history']
    
    ax1.fill_between(time, 0, history, where=(history==1), 
                     color='green', alpha=0.3, label='Λειτουργικό')
    ax1.fill_between(time, 0, history, where=(history==0), 
                     color='red', alpha=0.3, label='Βλάβη')
    ax1.plot(time, history, 'k-', linewidth=0.5, alpha=0.5)
    
    # Βρες πρώτη βλάβη
    failures = np.where(history == 0)[0]
    if len(failures) > 0:
        first_failure = time[failures[0]]
        ax1.axvline(x=first_failure, color='red', linestyle='--', 
                   linewidth=2, label=f'Πρώτη βλάβη: {first_failure:.2f}h')
    
    ax1.set_xlabel('Χρόνος (ώρες)', fontsize=11)
    ax1.set_ylabel('Κατάσταση Συστήματος', fontsize=11)
    ax1.set_title('Δείγμα Προσομοίωσης - Κατάσταση Συστήματος', fontsize=13, fontweight='bold')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Βλάβη', 'Λειτουργικό'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # --- 2. Καταστάσεις Εξαρτημάτων (Δείγμα) ---
    ax2 = fig.add_subplot(gs[1, :])
    comp_states = results['sample_components']
    
    offset = 0
    colors_map = {1: 'green', 0: 'yellow', -1: 'red'}
    
    for i, (name, states) in enumerate(comp_states.items()):
        states_array = np.array(states)
        for state in [-1, 0, 1]:
            mask = (states_array == state)
            if np.any(mask):
                ax2.fill_between(time, i-0.4, i+0.4, where=mask,
                               color=colors_map[state], alpha=0.6)
    
    ax2.set_xlabel('Χρόνος (ώρες)', fontsize=11)
    ax2.set_ylabel('Εξάρτημα', fontsize=11)
    ax2.set_title('Δείγμα Προσομοίωσης - Καταστάσεις Εξαρτημάτων', fontsize=13, fontweight='bold')
    ax2.set_yticks(range(len(comp_states)))
    ax2.set_yticklabels(list(comp_states.keys()))
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Λειτουργικό'),
        Patch(facecolor='yellow', alpha=0.6, label='Μη Λειτουργικό (DC)'),
        Patch(facecolor='red', alpha=0.6, label='Βλάβη')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # --- 3. Κατανομή Χρόνων Βλάβης ---
    ax3 = fig.add_subplot(gs[2, 0])
    if len(results['failure_times']) > 0:
        ax3.hist(results['failure_times'], bins=40, alpha=0.7, 
                color='steelblue', edgecolor='black')
        ax3.axvline(x=results['mttf_exp'], color='red', linestyle='--', 
                   linewidth=2, label=f'MTTF={results["mttf_exp"]:.2f}h')
        ax3.set_xlabel('Χρόνος Βλάβης Συστήματος (ώρες)', fontsize=11)
        ax3.set_ylabel('Συχνότητα', fontsize=11)
        ax3.set_title('Κατανομή Χρόνων Βλάβης Συστήματος', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Δεν παρατηρήθηκαν βλάβες', 
                ha='center', va='center', fontsize=14)
        ax3.set_title('Κατανομή Χρόνων Βλάβης', fontsize=12, fontweight='bold')
    
    # --- 4. Σύγκριση Αξιοπιστίας ---
    ax4 = fig.add_subplot(gs[2, 1])
    categories = ['Πειραματική', 'Θεωρητική']
    values = [results['reliability_exp'], results['reliability_theo']]
    colors = ['steelblue', 'coral']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Αξιοπιστία R(Ts)', fontsize=11)
    ax4.set_title(f'Αξιοπιστία Συστήματος στο Ts={Ts}h', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.1 * max(values)])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Προσθήκη τιμών πάνω από τις μπάρες
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.savefig('system_reliability_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Γράφημα αποθηκεύτηκε: system_reliability_analysis.png")


def create_summary_table(results):
    """
    Δημιουργεί συγκεντρωτικό πίνακα αποτελεσμάτων
    """
    print("\n" + "="*70)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΥΣΤΗΜΑΤΟΣ")
    print("="*70)
    print(f"\n{'Μέγεθος':<30} {'Πειραματικό':<20} {'Θεωρητικό':<20}")
    print("-"*70)
    print(f"{'Αξιοπιστία R(Ts)':<30} {results['reliability_exp']:<20.6f} {results['reliability_theo']:<20.6f}")
    
    if results['mttf_exp'] != float('inf'):
        print(f"{'MTTF (ώρες)':<30} {results['mttf_exp']:<20.4f} {'N/A':<20}")
    else:
        print(f"{'MTTF (ώρες)':<30} {'>'+str(Ts):<20} {'N/A':<20}")
    
    print(f"{'λ (βλάβες/ώρα) - Μέθοδος 1':<30} {results['lambda_method1']:<20.6f} {'N/A':<20}")
    print(f"{'λ (βλάβες/ώρα) - Μέθοδος 2':<30} {results['lambda_method2']:<20.6f} {'N/A':<20}")
    print(f"{'Αριθμός βλαβών':<30} {results['failures_count']}/{results['total_sims']:<14} {'N/A':<20}")
    print("="*70)


# --- ΚΥΡΙΑ ΕΚΤΕΛΕΣΗ ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ΑΝΑΛΥΣΗ ΑΞΙΟΠΙΣΤΙΑΣ ΣΥΣΤΗΜΑΤΟΣ - ΕΡΩΤΗΜΑ 4.2 (Μέρος 2)")
    print("="*70)
    print("\nΜπλοκ Διάγραμμα: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7\n")
    
    # Εκτέλεση προσομοιώσεων Monte Carlo
    results = run_system_monte_carlo(components_data, Ts, DT, N_SIMS)
    
    # Δημιουργία γραφημάτων
    visualize_system_simulation(results)
    
    # Εμφάνιση συγκεντρωτικού πίνακα
    create_summary_table(results)
    
    print("\n✓ Ανάλυση ολοκληρώθηκε επιτυχώς!")
    print("✓ Το γράφημα αποθηκεύτηκε στον τρέχοντα φάκελο.\n")

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


def simulate_component_failures(component_name, mttf, duty_cycle, duration, dt):
    """
    Προσομοιώνει τις βλάβες ενός εξαρτήματος χωρίς MTTR.
    
    Καταστάσεις:
    - 1: Λειτουργικό (Operational)
    - 0: Μη λειτουργικό λόγω duty cycle (Non-operational)
    - -1: Κατάσταση βλάβης (Failed)
    
    Returns:
        time_axis: Χρονικός άξονας
        status_history: Ιστορικό κατάστασης
        failure_time: Χρόνος πρώτης βλάβης (ή None αν δεν απέτυχε)
    """
    time_axis = np.arange(0, duration, dt)
    status_history = []
    failure_time = None
    
    # Υπολογισμός παραμέτρου λ της διαδικασίας Poisson
    lam = 1.0 / mttf
    
    current_status = 1  # Αρχικά λειτουργικό
    
    for t in time_axis:
        if current_status == -1:
            # Μόλις απέτυχε, παραμένει σε κατάσταση βλάβης
            status_history.append(-1)
        else:
            # Έλεγχος duty cycle: Είναι το εξάρτημα ενεργό σε αυτή τη στιγμή;
            is_active = np.random.rand() < duty_cycle
            
            if is_active:
                # Το εξάρτημα λειτουργεί - έλεγχος για βλάβη
                # Πιθανότητα αποτυχίας σε χρόνο dt: P = 1 - e^(-λ * dt)
                prob_fail = 1 - np.exp(-lam * dt)
                
                if np.random.rand() < prob_fail:
                    # Βλάβη!
                    current_status = -1
                    failure_time = t
                    status_history.append(-1)
                else:
                    # Λειτουργικό
                    current_status = 1
                    status_history.append(1)
            else:
                # Μη λειτουργικό λόγω duty cycle
                current_status = 0
                status_history.append(0)
    
    return time_axis, np.array(status_history), failure_time


def run_monte_carlo_simulations(component_name, mttf, duty_cycle, duration, dt, n_sims):
    """
    Εκτελεί N προσομοιώσεις για ένα εξάρτημα και υπολογίζει:
    - Πειραματικό ρυθμό αποτυχίας λ
    - Πειραματική αξιοπιστία R(Tc)
    """
    failure_times = []
    failures_occurred = 0
    
    print(f"\n{'='*60}")
    print(f"Προσομοίωση εξαρτήματος: {component_name}")
    print(f"MTTF = {mttf} ώρες, Duty Cycle = {duty_cycle}, Tc = {duration} ώρες")
    print(f"{'='*60}")
    
    for i in range(n_sims):
        _, _, failure_time = simulate_component_failures(
            component_name, mttf, duty_cycle, duration, dt
        )
        
        if failure_time is not None:
            failure_times.append(failure_time)
            failures_occurred += 1
        
        if (i + 1) % 100 == 0:
            print(f"Ολοκληρώθηκαν {i + 1}/{n_sims} προσομοιώσεις...")
    
    # --- Υπολογισμός Αποτελεσμάτων ---
    
    # Πειραματική Αξιοπιστία R(Tc): Ποσοστό των προσομοιώσεων που ΔΕΝ απέτυχαν
    reliability_experimental = (n_sims - failures_occurred) / n_sims
    
    # Θεωρητική Αξιοπιστία για σύγκριση
    # R(t) = e^(-λ * t_effective) όπου t_effective = duty_cycle * t
    lam_theoretical = 1.0 / mttf
    t_effective = duty_cycle * duration
    reliability_theoretical = np.exp(-lam_theoretical * t_effective)
    
    # Πειραματικός ρυθμός αποτυχίας λ
    # Μέθοδος 1: Από το μέσο χρόνο μεταξύ βλαβών
    if len(failure_times) > 0:
        mean_failure_time = np.mean(failure_times)
        # Προσαρμογή για duty cycle
        effective_mttf = mean_failure_time / duty_cycle if duty_cycle > 0 else float('inf')
        lambda_experimental_method1 = 1.0 / effective_mttf if effective_mttf > 0 else 0
    else:
        lambda_experimental_method1 = 0
        mean_failure_time = None
    
    # Μέθοδος 2: Από την αξιοπιστία R(Tc)
    # R(Tc) = e^(-λ * Tc_effective) => λ = -ln(R(Tc)) / Tc_effective
    if reliability_experimental > 0 and duty_cycle > 0:
        lambda_experimental_method2 = -np.log(reliability_experimental) / t_effective
    else:
        lambda_experimental_method2 = 0
    
    # --- Εμφάνιση Αποτελεσμάτων ---
    print(f"\n{'-'*60}")
    print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ για {component_name}:")
    print(f"{'-'*60}")
    print(f"Αριθμός βλαβών: {failures_occurred}/{n_sims}")
    print(f"\nΑΞΙΟΠΙΣΤΙΑ R(Tc={duration}):")
    print(f"  Πειραματική:  R = {reliability_experimental:.6f}")
    print(f"  Θεωρητική:    R = {reliability_theoretical:.6f}")
    print(f"  Σχετικό Σφάλμα: {abs(reliability_experimental - reliability_theoretical) / reliability_theoretical * 100:.2f}%")
    
    print(f"\nΡΥΘΜΟΣ ΑΠΟΤΥΧΙΑΣ λ:")
    print(f"  Θεωρητική:    λ = {lam_theoretical:.6f} βλάβες/ώρα")
    print(f"  Πειραματική (Μέθοδος 1 - από MTTF): λ = {lambda_experimental_method1:.6f} βλάβες/ώρα")
    print(f"  Πειραματική (Μέθοδος 2 - από R):    λ = {lambda_experimental_method2:.6f} βλάβες/ώρα")
    
    if mean_failure_time is not None:
        print(f"\nΜέσος χρόνος πρώτης βλάβης: {mean_failure_time:.4f} ώρες")
        print(f"Effective MTTF (προσαρμοσμένο για DC): {effective_mttf:.4f} ώρες")
    
    return {
        'component': component_name,
        'failures': failures_occurred,
        'total_sims': n_sims,
        'reliability_exp': reliability_experimental,
        'reliability_theo': reliability_theoretical,
        'lambda_theo': lam_theoretical,
        'lambda_exp_method1': lambda_experimental_method1,
        'lambda_exp_method2': lambda_experimental_method2,
        'failure_times': failure_times,
        'mean_failure_time': mean_failure_time
    }


def visualize_single_simulation(component_name, mttf, duty_cycle, duration, dt):
    """
    Δημιουργεί γράφημα για μία προσομοίωση
    """
    time_axis, status_history, failure_time = simulate_component_failures(
        component_name, mttf, duty_cycle, duration, dt
    )
    
    plt.figure(figsize=(12, 4))
    
    # Χρωματισμός ανάλογα με την κατάσταση
    colors = []
    for status in status_history:
        if status == 1:
            colors.append('green')  # Λειτουργικό
        elif status == 0:
            colors.append('yellow')  # Μη λειτουργικό
        else:
            colors.append('red')  # Βλάβη
    
    plt.scatter(time_axis, status_history, c=colors, s=1, alpha=0.5)
    
    if failure_time is not None:
        plt.axvline(x=failure_time, color='red', linestyle='--', 
                   label=f'Βλάβη στο t={failure_time:.2f}h')
    
    plt.xlabel('Χρόνος (ώρες)', fontsize=12)
    plt.ylabel('Κατάσταση', fontsize=12)
    plt.title(f'Προσομοίωση Βλαβών - {component_name} (MTTF={mttf}h, DC={duty_cycle})', fontsize=14)
    plt.yticks([-1, 0, 1], ['Βλάβη', 'Μη Λειτουργικό', 'Λειτουργικό'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{component_name}_simulation_example.png', dpi=150)
    print(f"\nΓράφημα αποθηκεύτηκε: {component_name}_simulation_example.png")


def create_summary_plots(results_list):
    """
    Δημιουργεί συγκεντρωτικά γραφήματα για όλα τα εξαρτήματα
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = [r['component'] for r in results_list]
    
    # 1. Σύγκριση Αξιοπιστίας
    ax1 = axes[0, 0]
    rel_exp = [r['reliability_exp'] for r in results_list]
    rel_theo = [r['reliability_theo'] for r in results_list]
    x = np.arange(len(components))
    width = 0.35
    ax1.bar(x - width/2, rel_exp, width, label='Πειραματική', alpha=0.8)
    ax1.bar(x + width/2, rel_theo, width, label='Θεωρητική', alpha=0.8)
    ax1.set_xlabel('Εξάρτημα')
    ax1.set_ylabel('Αξιοπιστία R(Tc)')
    ax1.set_title(f'Αξιοπιστία σε Tc={Tc} ώρες')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Σύγκριση Ρυθμού Αποτυχίας
    ax2 = axes[0, 1]
    lam_exp = [r['lambda_exp_method2'] for r in results_list]
    lam_theo = [r['lambda_theo'] for r in results_list]
    ax2.bar(x - width/2, lam_exp, width, label='Πειραματικό', alpha=0.8)
    ax2.bar(x + width/2, lam_theo, width, label='Θεωρητικό', alpha=0.8)
    ax2.set_xlabel('Εξάρτημα')
    ax2.set_ylabel('Ρυθμός Αποτυχίας λ (βλάβες/ώρα)')
    ax2.set_title('Ρυθμός Αποτυχίας')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ποσοστό Βλαβών
    ax3 = axes[1, 0]
    failure_rates = [r['failures'] / r['total_sims'] * 100 for r in results_list]
    ax3.bar(components, failure_rates, alpha=0.8, color='coral')
    ax3.set_xlabel('Εξάρτημα')
    ax3.set_ylabel('Ποσοστό Προσομοιώσεων με Βλάβη (%)')
    ax3.set_title('Ποσοστό Βλαβών')
    ax3.grid(True, alpha=0.3)
    
    # 4. Κατανομή Χρόνων Βλάβης (παράδειγμα για πρώτο εξάρτημα)
    ax4 = axes[1, 1]
    if len(results_list[0]['failure_times']) > 0:
        ax4.hist(results_list[0]['failure_times'], bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Χρόνος Βλάβης (ώρες)')
        ax4.set_ylabel('Συχνότητα')
        ax4.set_title(f'Κατανομή Χρόνων Βλάβης - {results_list[0]["component"]}')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('failure_simulation_summary.png', dpi=150)
    print(f"\nΣυγκεντρωτικό γράφημα αποθηκεύτηκε: failure_simulation_summary.png")


# --- ΚΥΡΙΑ ΕΚΤΕΛΕΣΗ ---
if __name__ == "__main__":
    print("="*60)
    print("ΠΡΟΣΟΜΟΙΩΣΗ ΒΛΑΒΩΝ ΕΞΑΡΤΗΜΑΤΩΝ - ΕΡΩΤΗΜΑ 4.2 (Μέρος 1)")
    print("="*60)
    print(f"Παράμετροι: Tc={Tc}h, dt={DT}h, N={N_SIMS} προσομοιώσεις")
    
    all_results = []
    
    # Εκτέλεση προσομοιώσεων για κάθε εξάρτημα
    for comp_name, specs in components_data.items():
        results = run_monte_carlo_simulations(
            comp_name, 
            specs['MTTF'], 
            specs['DC'], 
            Tc, 
            DT, 
            N_SIMS
        )
        all_results.append(results)
        
        # Δημιουργία γραφήματος για ένα παράδειγμα προσομοίωσης
        if comp_name in ['C1', 'C7']:  # Μόνο για τα εξαρτήματα με DC < 1
            visualize_single_simulation(comp_name, specs['MTTF'], specs['DC'], Tc, DT)
    
    # Δημιουργία συγκεντρωτικών γραφημάτων
    create_summary_plots(all_results)
    
    # Συγκεντρωτικός πίνακας
    print("\n" + "="*80)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    print("="*80)
    print(f"{'Εξάρτημα':<12} {'R_exp':<10} {'R_theo':<10} {'λ_exp':<12} {'λ_theo':<12} {'Βλάβες':<10}")
    print("-"*80)
    for r in all_results:
        print(f"{r['component']:<12} {r['reliability_exp']:<10.6f} {r['reliability_theo']:<10.6f} "
              f"{r['lambda_exp_method2']:<12.6f} {r['lambda_theo']:<12.6f} {r['failures']:<10}")
    print("="*80)
    
    print("\n✓ Προσομοίωση ολοκληρώθηκε επιτυχώς!")
    print("✓ Τα γραφήματα αποθηκεύτηκαν στον τρέχοντα φάκελο.")

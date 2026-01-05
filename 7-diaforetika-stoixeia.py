import numpy as np
import matplotlib.pyplot as plt

# --- Παράμετροι Εξαρτημάτων (Από Πίνακα 2) ---
# Format: 'Name': {'MTTF': value, 'DC': value, 'MTTR': value}
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
TS = 30.0        # Χρόνος μελέτης συστήματος (ώρες)
DT = 0.01        # Χρονικό βήμα (λεπτομέρεια προσομοίωσης)
NUM_SIMS = 1000  # Αριθμός επαναλήψεων Monte Carlo

def run_simulation(components_db, duration, dt, repair_enabled=True):
    # Αρχικοποίηση κατάστασης
    # 1 = Λειτουργεί, 0 = Βλάβη
    current_status = {name: 1 for name in components_db} 
    
    # Χρόνος που απομένει για επιδιόρθωση (αν είναι σε βλάβη)
    repair_timers = {name: 0 for name in components_db}
    
    system_history = []
    time_axis = np.arange(0, duration, dt)

    for t in time_axis:
        # --- Έλεγχος κάθε εξαρτήματος ---
        for name, specs in components_db.items():
            if current_status[name] == 0: 
                # Το εξάρτημα είναι σε βλάβη
                if repair_enabled:
                    repair_timers[name] -= dt
                    if repair_timers[name] <= 0:
                        current_status[name] = 1 # Επιδιορθώθηκε
            else:
                # Το εξάρτημα λειτουργεί
                # Έλεγχος Duty Cycle (αν είναι ενεργό σε αυτό το βήμα)
                # Αν DC=1 είναι πάντα ενεργό. Αν DC=0.3 έχει 30% πιθανότητα να είναι ενεργό
                is_active = np.random.rand() < specs['DC']
                
                if is_active:
                    # Πιθανότητα αποτυχίας σε χρόνο dt: P = 1 - e^(-lambda * dt)
                    # lambda = 1 / MTTF
                    lam = 1.0 / specs['MTTF']
                    prob_fail = 1 - np.exp(-lam * dt)
                    
                    if np.random.rand() < prob_fail:
                        current_status[name] = 0 # Βλάβη!
                        # Ορισμός χρόνου επιδιόρθωσης (εκθετική κατανομή)
                        if repair_enabled:
                            repair_timers[name] = np.random.exponential(specs['MTTR'])

        # --- Υπολογισμός Κατάστασης Συστήματος (Λογική Εικόνας 2) ---
        # Δομή: C1 AND (C2 OR C3 OR C4) AND (C5 OR C6) AND C7
        
        block1 = current_status['C1']
        block2 = current_status['C2'] or current_status['C3'] or current_status['C4']
        block3 = current_status['C5'] or current_status['C6']
        block4 = current_status['C7']
        
        system_status = block1 and block2 and block3 and block4
        system_history.append(system_status)

    return time_axis, system_history

# --- Εκτέλεση Monte Carlo ---
total_up_time = 0
failures_count = 0

print(f"Έναρξη προσομοίωσης για {NUM_SIMS} επαναλήψεις...")

for i in range(NUM_SIMS):
    time, history = run_simulation(components_data, TS, DT, repair_enabled=True)
    
    # Υπολογισμός χρόνου λειτουργίας σε αυτή την επανάληψη
    up_time = sum(history) * DT
    total_up_time += up_time
    print(f"προσομοίωση {i+1} από {NUM_SIMS} ολοκληρώθηκε. Χρόνος λειτουργίας: {up_time:.2f} ώρες.")
    
    # Αν θέλουμε να δούμε αν το σύστημα απέτυχε ποτέ
    if 0 in history:
        failures_count += 1

# --- Αποτελέσματα ---
mean_availability = total_up_time / (NUM_SIMS * TS)
print(f"\nΑποτελέσματα για το Σύστημα 4.2:")
print(f"Μέση Διαθεσιμότητα (Availability): {mean_availability:.4f}")
print(f"Θεωρητικός Μέγιστος Χρόνος Λειτουργίας: {TS} ώρες")

# Παράδειγμα γραφήματος για μία τυχαία εκτέλεση (Ζητούμενο 4)
t, h = run_simulation(components_data, TS, DT, repair_enabled=True)
plt.figure(figsize=(10, 4))
plt.plot(t, h, label='Κατάσταση Συστήματος')
plt.xlabel('Χρόνος (ώρες)')
plt.ylabel('Κατάσταση (1=OK, 0=Failed)')
plt.title('Προσομοίωση Λειτουργίας Συστήματος 4.2 (Μία εκτέλεση)')
plt.grid(True)
plt.yticks([0, 1], ['Βλάβη', 'Λειτουργία'])
plt.show()
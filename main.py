import sys
import time
import argparse
import subprocess
from config import print_header, print_simulation_info

def run_part1():
    print_header("ΜΕΡΟΣ 1: ΠΡΟΣΟΜΟΙΩΣΗ ΒΛΑΒΩΝ ΕΞΑΡΤΗΜΑΤΩΝ", "=", 70)
    result = subprocess.run([sys.executable, "src/part1_failure_simulation.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 1 failed with return code {result.returncode}")

def run_part2():
    print_header("ΜΕΡΟΣ 2: ΑΞΙΟΠΙΣΤΙΑ ΣΥΣΤΗΜΑΤΟΣ", "=", 70)
    result = subprocess.run([sys.executable, "src/part2_system_reliability.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 2 failed with return code {result.returncode}")

def run_part3():
    print_header("ΜΕΡΟΣ 3: ΠΡΟΣΟΜΟΙΩΣΗ ΕΞΑΡΤΗΜΑΤΩΝ ΜΕ ΕΠΙΔΙΟΡΘΩΣΗ", "=", 70)
    result = subprocess.run([sys.executable, "src/part3_component_repair.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 3 failed with return code {result.returncode}")

def run_part4():
    print_header("ΜΕΡΟΣ 4: ΑΝΑΛΥΣΗ ΣΥΣΤΗΜΑΤΟΣ ΜΕ ΕΠΙΔΙΟΡΘΩΣΗ", "=", 70)
    result = subprocess.run([sys.executable, "src/part4_system_repair.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 4 failed with return code {result.returncode}")

def run_all():    
    parts = [
        ("1", "Προσομοίωση Βλαβών Εξαρτημάτων (χωρίς MTTR)", run_part1),
        ("2", "Αξιοπιστία Συστήματος (χωρίς MTTR)", run_part2),
        ("3", "Προσομοίωση Εξαρτημάτων με Επιδιόρθωση", run_part3),
        ("4", "Ανάλυση Συστήματος με Επιδιόρθωση", run_part4),
    ]
    
    total_start = time.time()
    results = []
    
    for part_num, part_name, part_func in parts:
        print(f"\n{'#' * 80}")
        print(f"# ΕΚΚΙΝΗΣΗ ΜΕΡΟΥΣ {part_num}: {part_name}")
        print(f"{'#' * 80}")
        
        start_time = time.time()
        try:
            part_func()
            elapsed = time.time() - start_time
            results.append((part_num, part_name, "ΕΠΙΤΥΧΙΑ", elapsed))
            print(f"\n✓ Μέρος {part_num} ολοκληρώθηκε σε {elapsed:.2f} δευτερόλεπτα")
        except Exception as e:
            elapsed = time.time() - start_time
            results.append((part_num, part_name, f"ΣΦΑΛΜΑ: {e}", elapsed))
            print(f"\n✗ Μέρος {part_num} απέτυχε: {e}")
    
    # Εμφάνιση συνοπτικών αποτελεσμάτων
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("ΣΥΝΟΨΗ ΕΚΤΕΛΕΣΗΣ")
    print("=" * 80)
    print(f"\n{'Μέρος':<8} {'Περιγραφή':<45} {'Κατάσταση':<12} {'Χρόνος':<10}")
    print("-" * 80)
    
    for part_num, part_name, status, elapsed in results:
        status_short = "✓" if status == "ΕΠΙΤΥΧΙΑ" else "✗"
        print(f"{part_num:<8} {part_name:<45} {status_short:<12} {elapsed:.2f}s")
    
    print("-" * 80)
    print(f"{'Σύνολο:':<55} {' ':<12} {total_elapsed:.2f}s")
    print("=" * 80)
    
    # Έλεγχος για σφάλματα
    errors = [r for r in results if r[2] != "ΕΠΙΤΥΧΙΑ"]
    if errors:
        print(f"\n⚠ Προσοχή: {len(errors)} μέρος/η απέτυχε(αν)")
        return False
    else:
        print("\n✓ Όλες οι προσομοιώσεις ολοκληρώθηκαν επιτυχώς!")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--part',
        nargs='+',
        type=int,
        choices=[1, 2, 3, 4],
        help='execute specific part'
    )
    parser.add_argument(
        '-i', '--info',
        action='store_true',
        help='show config parameters'
    )

    print("\n" + "=" * 70)
    print("ΠΡΟΣΟΜΟΙΩΣΗ ΑΞΙΟΠΙΣΤΙΑΣ ΣΥΝΘΕΤΟΥ ΣΥΣΤΗΜΑΤΟΣ".center(70))
    print("=" * 70)
    
    # Parse arguments
    args = parser.parse_args()
    if args.info:
        print_simulation_info()
        return
    elif args.part:
        part_functions = {
            1: run_part1,
            2: run_part2,
            3: run_part3,
            4: run_part4,
        }
        
        for part_num in sorted(set(args.part)):           
            start_time = time.time()
            part_functions[part_num]()
            elapsed = time.time() - start_time
            print(f"\nPart {part_num} completed in {elapsed:.2f} seconds")
    else:
        run_all()

if __name__ == "__main__":
    main()

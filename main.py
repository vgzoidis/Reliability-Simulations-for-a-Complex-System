import sys
import time
import argparse
import subprocess
from config import print_header, print_simulation_info

def run_no_repair():
    print_header("SIMULATION WITHOUT REPAIR", "-", 70)
    result = subprocess.run([sys.executable, "src/simulation_no_repair.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Simulation failed with return code {result.returncode}")

def run_with_repair():
    print_header("SIMULATION WITH REPAIR", "-", 70)
    result = subprocess.run([sys.executable, "src/simulation_with_repair.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Simulation failed with return code {result.returncode}")

def run_all():    
    run_no_repair()
    run_with_repair()

def main():
    parser = argparse.ArgumentParser(description="Reliability Simulation for a Complex System")
    parser.add_argument(
        '-m', '--mode',
        choices=['no_repair', 'with_repair', 'all'],
        default='all',
        help='Simulation mode: no_repair (λ,R,MTTF), with_repair (MTBF,MUT,MTTR,A), or all'
    )
    parser.add_argument(
        '-i', '--info',
        action='store_true',
        help='Show config parameters'
    )

    print("\n" + "=" * 70)
    print("RELIABILITY SIMULATION FOR A COMPLEX SYSTEM".center(70))
    print("=" * 70)
    
    args = parser.parse_args()
    
    if args.info:
        print_simulation_info()
        return
    
    start_time = time.time()
    
    if args.mode == 'no_repair':
        run_no_repair()
    elif args.mode == 'with_repair':
        run_with_repair()
    else:
        run_all()
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

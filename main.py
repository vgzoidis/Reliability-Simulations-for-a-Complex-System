import sys
import time
import argparse
import subprocess
from config import print_header, print_simulation_info

def run_part1():
    print_header("PART 1: COMPONENT FAILURE SIMULATION", "=", 70)
    result = subprocess.run([sys.executable, "src/part1_failure_simulation.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 1 failed with return code {result.returncode}")

def run_part2():
    print_header("PART 2: SYSTEM RELIABILITY", "=", 70)
    result = subprocess.run([sys.executable, "src/part2_system_reliability.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 2 failed with return code {result.returncode}")

def run_part3():
    print_header("PART 3: COMPONENT SIMULATION WITH REPAIR", "=", 70)
    result = subprocess.run([sys.executable, "src/part3_component_repair.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 3 failed with return code {result.returncode}")

def run_part4():
    print_header("PART 4: SYSTEM ANALYSIS WITH REPAIR", "=", 70)
    result = subprocess.run([sys.executable, "src/part4_system_repair.py"], capture_output=False, text=True)
    if result.returncode != 0:
        raise Exception(f"Part 4 failed with return code {result.returncode}")

def run_all():    
    parts = [
        ("1", "Component Failure Simulation (without MTTR)", run_part1),
        ("2", "System Reliability (without MTTR)", run_part2),
        ("3", "Component Simulation with Repair", run_part3),
        ("4", "System Analysis with Repair", run_part4),
    ]
    
    for _,_, part_func in parts:
        part_func()

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
    print("RELIABILITY SIMULATION FOR A COMPLEX SYSTEM".center(70))
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

import os

# Component Parameters
COMPONENTS_DATA = {
    'C1': {'MTTF': 30, 'DC': 0.3, 'MTTR': 12},
    'C2': {'MTTF': 24, 'DC': 1.0, 'MTTR': 12},
    'C3': {'MTTF': 23, 'DC': 1.0, 'MTTR': 12},
    'C4': {'MTTF': 24, 'DC': 1.0, 'MTTR': 10},
    'C5': {'MTTF': 27, 'DC': 1.0, 'MTTR': 10},
    'C6': {'MTTF': 28, 'DC': 1.0, 'MTTR': 8},
    'C7': {'MTTF': 33, 'DC': 0.4, 'MTTR': 12},
}

# Simulation Parameters
Tc = 100.0       # Component study time
Ts = 30.0        # System study time
DT = 0.1       # Time step (hours)
N_SIMS = 1000    # Number of simulations

# Results directories
OUTPUT_DIRS = {
    'no_repair': 'results/no_repair',
    'with_repair': 'results/with_repair',
}

# System structure
SYSTEM_STRUCTURE = {
    'series': ['C1', 'block2', 'block3', 'C7'],
    'parallel_blocks': {
        'block2': ['C2', 'C3', 'C4'],
        'block3': ['C5', 'C6'],
    }
}

def ensure_output_dir(dir_name: str) -> str:
    if dir_name in OUTPUT_DIRS:
        output_dir = OUTPUT_DIRS[dir_name]
    else:
        output_dir = dir_name
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def print_header(title: str, char: str = "=", width: int = 70):
    print(f"{char * width}")
    print(title.center(width))
    print(f"{char * width}")

def print_simulation_info():
    print_header("SIMULATION PARAMETERS")
    print(f"Tc: {Tc} hours")
    print(f"Ts: {Ts} hours")
    print(f"dt: {DT} hours")
    print(f"N:  {N_SIMS}")

if __name__ == "__main__":
    print_simulation_info()

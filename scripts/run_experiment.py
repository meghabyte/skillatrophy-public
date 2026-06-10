import os
import json
import random
import subprocess
import argparse
import time
import sys

# Resolve paths relative to project root (PythonAPI/examples/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

SCRIPT_NAME = os.path.join(_PROJECT_ROOT, "src", "steeringwheel_custom.py")
JSON_FILE = os.path.join(_PROJECT_ROOT, "data", "results.json")
TOTAL_STATES = 9  # States 0 through 8
TRIALS_PER_CONFIG = 2
SHARED_AUTONOMY_OPTIONS = [True, False] # True = -s, False = no flag

def load_completed_trials(user_name):
    """
    Reads results.json and returns a set of completed trials for the specific user.
    Returns set format: { (state_int, is_sa_bool, trial_str) }
    """
    if not os.path.exists(JSON_FILE):
        return set()

    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"Warning: {JSON_FILE} exists but is empty or corrupt. Assuming 0 completed.")
        return set()

    completed = set()
    for entry in data:
        # Filter by user
        if entry.get('user') == user_name:
            # Extract key identifiers
            try:
                s = int(entry.get('state'))
                sa = bool(entry.get('shared_autonomy'))
                t = str(entry.get('trial'))
                completed.add((s, sa, t))
            except (ValueError, TypeError):
                continue
            
    return completed

def get_todo_list(completed_set):
    """
    Generates a list of all required trials that haven't been completed yet.
    
    LOGIC FIX:
    This now enforces sequential ordering for trials.
    For any specific State+Mode combination, we only add the LOWEST 
    missing trial number to the pool. 
    
    You cannot be assigned Trial 2 until Trial 1 is in the JSON.
    """
    todo = []
    
    for state in range(TOTAL_STATES):
        for use_sa in SHARED_AUTONOMY_OPTIONS:
            
            # Find the first missing trial for this specific combo
            next_trial_to_do = None
            
            for trial_num in range(1, TRIALS_PER_CONFIG + 1):
                trial_str = str(trial_num)
                
                # If this trial is NOT done, this is the one we must do next.
                if (state, use_sa, trial_str) not in completed_set:
                    next_trial_to_do = trial_str
                    break # STOP checking higher trials. Logic enforces 1 before 2.
            
            # If we found a missing step for this combo, add it to the random pool
            if next_trial_to_do is not None:
                todo.append({
                    'state': state,
                    'sa': use_sa,
                    'trial': next_trial_to_do
                })
                
    return todo

def run_trial(user, config):
    """
    Constructs the command and runs the simulation.
    Returns True if the user completed the task (found in json), False otherwise.
    """
    state = config['state']
    trial = config['trial']
    use_sa = config['sa']

    # 1. Construct Command
    cmd = [
        sys.executable,  # Uses the current python interpreter
        SCRIPT_NAME,
        "--user", user,
        "--state", str(state),
        "--trial", trial
    ]
    
    if use_sa:
        cmd.append("-s")

    mode_str = "SHARED AUTONOMY" if use_sa else "MANUAL CONTROL"
    print("\n" + "="*60)
    print(f"STARTING NEXT RANDOMIZED TRIAL")
    print(f"User:  {user}")
    print(f"State: {state} (of 0-8)")
    print(f"Mode:  {mode_str}")
    print(f"Trial: {trial}")
    print("="*60)
    
    # 2. Run the script
    try:
        # We wait for the process to finish
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExperiment script interrupted by user.")
        sys.exit(0)

    # 3. Verification
    # Reload JSON to see if the new trial exists
    post_run_completed = load_completed_trials(user)
    
    key = (state, use_sa, trial)
    
    if key in post_run_completed:
        print(f"\n[SUCCESS] Trial recorded in {JSON_FILE}.")
        return True
    else:
        print(f"\n[FAILURE] Trial ended but was not found in {JSON_FILE}.")
        print("It seems the trial was aborted or failed.")
        print("This specific configuration will remain in the pool.")
        time.sleep(2) # Give user a moment to breathe
        return False

def main():
    parser = argparse.ArgumentParser(description="Randomized Data Collection Runner")
    parser.add_argument("--user", required=True, type=str, help="Name of the user for logging")
    args = parser.parse_args()

    user = args.user
    
    # Validation of existing file
    if not os.path.exists(SCRIPT_NAME):
        print(f"ERROR: Could not find '{SCRIPT_NAME}'.")
        sys.exit(1)

    print(f"Initializing Experiment Runner for user: {user}...")

    while True:
        # 1. Load what is done
        completed = load_completed_trials(user)
        
        # 2. Calculate what is left (Sequentially enforced)
        todo_list = get_todo_list(completed)
        
        total_required = TOTAL_STATES * len(SHARED_AUTONOMY_OPTIONS) * TRIALS_PER_CONFIG
        total_done = len(completed)
        
        if not todo_list:
            print("\n" + "*"*60)
            print("ALL TRIALS COMPLETED!")
            print(f"Total successful trials found: {total_done}/{total_required}")
            print("*"*60)
            break

        print(f"Progress: {total_done}/{total_required} trials completed.")
        print(f"Valid options in random pool: {len(todo_list)}")
        
        # 3. Pick a random configuration
        current_config = random.choice(todo_list)
        
        # 4. Run it
        success = run_trial(user, current_config)
        
        if success:
            print("Moving to next random trial...")
            time.sleep(1)
        else:
            print("Retrying (randomly re-shuffling)...")

if __name__ == "__main__":
    main()
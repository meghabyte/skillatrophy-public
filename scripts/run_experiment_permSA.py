import os
import json
import random
import subprocess
import argparse
import time
import sys

# ==============================================================================
# -- CONFIGURATION -------------------------------------------------------------
# ==============================================================================

# Resolve paths relative to project root (PythonAPI/examples/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

SCRIPT_NAME = os.path.join(_PROJECT_ROOT, "src", "steeringwheel_custom.py")
JSON_FILE = os.path.join(_PROJECT_ROOT, "data", "results.json")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "experiment_permSA")

# Run Trials 1-10 for State 9 only
TARGET_STATE = 9       
TRIALS_PER_STATE = 10

# Randomization Limits (for Trials 9-10)
RANDOM_Y_RANGE = 0.0        # Meters (+/-)
RANDOM_HEADING_RANGE = 12.0  # Degrees (+/-)

# Shared Autonomy Settings
ALPHA_VALUE = 0.5  # Fixed blending weight for SA phases

# ==============================================================================
# -- HELPER FUNCTIONS ----------------------------------------------------------
# ==============================================================================

def load_completed_trials(user_name):
    """
    Reads results.json and returns a set of completed tasks.
    Returns set format: { (state_int, trial_int) }
    """
    if not os.path.exists(JSON_FILE):
        return set()

    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return set()

    completed = set()
    for entry in data:
        if entry.get('user') == user_name:
            try:
                s = int(entry.get('state'))
                t = int(entry.get('trial'))
                completed.add((s, t))
            except (ValueError, TypeError):
                continue
            
    return completed

def get_next_task(user_name):
    """
    Scans for the next incomplete trial for TARGET_STATE (9).
    """
    completed = load_completed_trials(user_name)
    
    # Check trials 1 through 10 for this state
    for trial in range(1, TRIALS_PER_STATE + 1):
        if (TARGET_STATE, trial) not in completed:
            # Enforce order: Return the first missing trial we find
            return (TARGET_STATE, trial)

    # State is fully complete
    return None

def run_trial(user, state, trial):
    # -------------------------------------------------
    # 1. DETERMINE EXPERIMENTAL CONDITIONS
    # -------------------------------------------------
    
    y_offset = 0.0
    heading_offset = 0.0
    use_shared_autonomy = False
    use_phase2_sa = False 
    mode_description = "UNKNOWN"

    # TRIALS 1-2: Baseline (Manual, Straight)
    if 1 <= trial <= 2:
        use_shared_autonomy = False
        mode_description = "BASELINE (Manual - Straight)"

    # TRIALS 3-6: Practice (High AI Assistance, Straight)
    elif 3 <= trial <= 6:
        use_shared_autonomy = True
        use_phase2_sa = True  # Keep SA active during Phase 2 for practice trials
        mode_description = f"PRACTICE (Shared Autonomy - Straight | α={ALPHA_VALUE})"

    # TRIALS 7-8: Evaluation 1 (Manual, Straight)
    elif 7 <= trial <= 8:
        use_shared_autonomy = False
        mode_description = "EVAL 1 (Manual - Straight)"

    # TRIALS 9-10: Evaluation 2 (Manual, Randomized)
    elif 9 <= trial <= 10:
        use_shared_autonomy = False
        mode_description = "EVAL 2 (Manual - Randomized)"
        y_offset = random.uniform(-RANDOM_Y_RANGE, RANDOM_Y_RANGE)
        heading_offset = random.uniform(-RANDOM_HEADING_RANGE, RANDOM_HEADING_RANGE)

    # -------------------------------------------------
    # 2. CALCULATE PATHS
    # -------------------------------------------------
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    file_base = f"experiment_{user}_state{state}_trial{trial}_sa{use_shared_autonomy}"
    csv_path = os.path.join(DATA_DIR, f"{file_base}.csv")
    log_name = f"{file_base}.log"

    # -------------------------------------------------
    # 3. CONSTRUCT COMMAND
    # -------------------------------------------------
    cmd = [
        sys.executable, SCRIPT_NAME,
        "--user", user,
        "--state", str(state), 
        "--trial", str(trial),
        "--y-offset", str(y_offset),
        "--heading-offset", str(heading_offset),
        "--csv_overwrite", csv_path,
        "--filepath", log_name
    ]
    
    if use_shared_autonomy:
        cmd.append("-s")
        cmd.extend(["--alpha", str(ALPHA_VALUE)])

    if use_phase2_sa:
        cmd.append("--phase2-sa")

    # -------------------------------------------------
    # 4. PRINT STATUS & EXECUTE
    # -------------------------------------------------
    print("\n" + "="*65)
    print(f"STARTING TRIAL (FIXED IN-LINE PROTOCOL)")
    print(f"User:     {user}")
    print(f"State:    {state}")
    print(f"Trial:    {trial} / {TRIALS_PER_STATE}")
    print(f"Config:   {mode_description}")
    print(f"Offsets:  Y={y_offset:.1f}m | Heading={heading_offset:.1f}°")
    print("="*65)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(0)

    # -------------------------------------------------
    # 5. VERIFY RESULT
    # -------------------------------------------------
    completed_now = load_completed_trials(user)
    
    if (state, trial) in completed_now:
        print(f"\n[SUCCESS] Data saved for State {state}, Trial {trial}.")
        return True
    else:
        print(f"\n[FAILURE] Trial finished but data was not found in {JSON_FILE}.")
        print("Likely crashed or exited early. Retrying...")
        time.sleep(2)
        return False

# ==============================================================================
# -- MAIN LOOP -----------------------------------------------------------------
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment Manager - Fixed SA Protocol")
    parser.add_argument("--user", required=True, type=str, help="Participant ID")
    args = parser.parse_args()

    user = args.user
    
    if not os.path.exists(SCRIPT_NAME):
        print(f"ERROR: Cannot find '{SCRIPT_NAME}'.")
        sys.exit(1)

    print(f"Initializing Experiment Runner for User: {user}")
    
    while True:
        task = get_next_task(user)
        
        if not task:
            print("\n" + "*"*65)
            print("ALL TRIALS COMPLETED SUCCESSFULLY!")
            print("Thank you for participating.")
            print("*"*65)
            break
            
        state, trial = task
        success = run_trial(user, state, trial)
        
        if success:
            print("Loading next trial...")
            time.sleep(1)

if __name__ == "__main__":
    main()
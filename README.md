# Proximal State Nudging: Reducing Skill Atrophy from AI Assistance
The gradual loss of human skills, or **skill atrophy**, is a rising concern as users increasingly rely on AI assistance. This problem is particularly salient in  cooperative AI systems, such as aircraft piloting or driving, where humans and AI agents jointly share control and decision-making. In these settings, human operators often struggle to disentangle which outcomes arise from AI intervention versus their own actions, undermining opportunities for their own learning and long-term skill retention. 

We propose **Proximal State Nudging (PSN)**, a shared autonomy algorithm that jointly optimizes for skill development and task performance by nudging users toward states estimated to be most learnable. 

We first show that PSN outperforms existing shared autonomy baselines in balancing student improvement in unassisted reward with overall shared performance, using simulated students in the classic LunarLander environment. We then present, to the best of our knowledge, the first human subject studies of a planner incorporating learning-compatible shared autonomy: across two driving tasks in the CARLA simulator (High Performance Racing and Parallel Parking, n = 60), PSN produces up to 7x larger gains in unassisted skill than standard blended shared autonomy, while incurring 50% fewer collisions than unassisted self-practice.

Authors: Megha Srivastava*, Jonathan Ouyang^, Eric Zhou^, Andrew Silva~, Emily Sumner~, Dorsa Sadigh*, Yuchen Cui^, Deepak Gopinath~, Guy Rosman~

Institutions: Stanford Computer Science (*), UCLA Robot Intelligence Lab (^), Toyota Research Institute (~)

Paper: https://arxiv.org/abs/2605.20355

Contact: megha@cs.stanford.edu


## Code Release

This repository implements a shared autonomy parallel parking task using the [CARLA simulator](https://carla.org/) (v0.9.15). A human driver uses a Logitech G29 steering wheel to park, optionally assisted by an NMPC-based autonomous agent that blends its controls with the user's inputs. The codebase supports running user studies with treatment (shared autonomy) and control (manual only) groups, logging per-frame telemetry, and visualizing results.

### Installation

1. Install CARLA 0.9.15 from [carla.org](https://carla.org/).

2. Clone this repository:
   ```bash
   git clone https://github.com/JonOuyang/ILIAD-CARLA
   ```

3. Install Python dependencies:
   ```bash
   conda activate carla  # or your preferred environment
   pip install -r requirements.txt
   ```

4. Copy or symlink the contents of this repo into your CARLA installation's `PythonAPI/examples/` directory, or run scripts from this repo with CARLA's Python API on your `PYTHONPATH`.

### Quick Start

#### One-command launch
```bash
chmod +x scripts/run_custom.sh
./scripts/run_custom.sh
```

#### Manual launch
```bash
# Terminal 1: Start CARLA
cd /home/driving_sim/CARLA_0.9.15/
./CarlaUE4.sh &

# Terminal 2: Load Town15 and run
cd /home/driving_sim/CARLA_0.9.15/PythonAPI/util/
./config.py --map Town15

cd /path/to/ILIAD-CARLA/
python src/steeringwheel_custom.py --user Alice --state 9 --trial 1 -s
```

**Town15 is required.** All coordinates are hardcoded for this map.

### Running Experiments
> Note that for all experiments, we have implemented automatic trial iteration. After starting up any of the run_experiment scripts, simply click `esc` to iterate to the next state. If the user has not successfully parked or time failed when `esc` is pressed, we restart that trial. If the user time fails, we iterate to the next trial. Same for successes. "Success" is measured by multiple features, including angle difference from target, euclidean position difference from target, and speed difference from target

#### Treatment group (shared autonomy practice)
```bash
python scripts/run_experiment_user.py --user <ParticipantName>
```
Runs 10 sequential trials for State 9:
- Trials 1-2: Baseline (manual)
- Trials 3-6: Practice (shared autonomy)
- Trials 7-8: Evaluation 1 (manual)
- Trials 9-10: Evaluation 2 (manual, randomized heading)

#### Control group (manual only)
```bash
python scripts/run_experiment_control.py --user <ParticipantName>
```
Same structure as above but all 10 trials are manual (no AI assistance during practice).

### Main Driving Script

`src/steeringwheel_custom.py` is the primary entry point. Key arguments:

| Flag | Description |
|------|-------------|
| `-s` / `--shared-autonomy` | Enable NMPC blending (agent assists user) |
| `--user` / `--state` / `--trial` | Label the run (used in filenames + JSON) |
| `--filepath` | CARLA `.log` recording name (saved to CARLA's `data/` dir) |
| `--csv_overwrite` | Custom path for the per-frame CSV |
| `--ignore` | Skip writing to `results.json` |
| `--y-offset` / `--heading-offset` | Perturb starting position (for randomized trials) |
| `--phase` / `--distance` | Starting pose variant |
| `--window-size` / `--res` | UI window vs. camera render resolution |

Each run produces:
- **CARLA recording:** `/home/driving_sim/CARLA_0.9.15/data/<name>.log`
- **Per-frame CSV:** `data/<user>_state<state>_trial<trial>_sa<True|False>.csv`
- **Trial summary:** appended to `data/results.json` (unless `--ignore`)

### Autonomous Agent

The NMPC parking agent in `agent/parking_agent.py` implements a two-phase autonomous parking policy:

1. **Approach phase** (`ApproachController`): Drives forward/backward from spawn to a staging position near the parking spot, maintaining orientation.
2. **Parking phase** (`NMPCController`): Uses CasADi-based nonlinear model predictive control to execute the parallel parking maneuver with obstacle avoidance.

In shared autonomy mode, the agent's controls are blended with the user's steering wheel inputs. In fully autonomous mode, the agent acts alone.

#### Running the agent standalone
```bash
python agent/run_autonomous.py
```
This runs the agent with no human input, demonstrating the autonomous parking capability.

### Recording and Playback

#### Replay a CARLA recording
```bash
python utils/recording_playback.py --recorder-filename Alice_state9_trial1_saTrue.log --time-factor 0.75
```
Relative filenames resolve under `/home/driving_sim/CARLA_0.9.15/data/`.
Options: `--start`/`--duration` (trim), `--camera` (follow actor), `--move-spectator`.

#### Export frame data from a recording (offline)
```bash
python utils/recording_data_dump.py --recorder-filename /path/to/recording.log
```
Extracts per-frame hero vehicle pose and velocity to `data_csv/<name>_hero_data.csv`.
Options: `--hero-id` (override auto-detection), `--output-file` (custom output path).


### Data Format

#### Per-frame CSV columns
`frame`, `timestamp`, `agent_active`, `x`, `y`, `velocity_x`, `velocity_y`, `velocity_z`, `heading_deg`, `yaw_deg`, `pitch_deg`, `roll_deg`, `user_throttle`, `user_brake`, `user_steer`, `agent_throttle`, `agent_brake`, `agent_steer`, `in_collision`, `current_phase`, `applied_throttle`, `applied_brake`, `applied_steer`

### Solver Note

The NMPC controller uses CasADi with IPOPT. If performance is slow, reducing the number of predicted steps from the solver can significantly reduce overhead.

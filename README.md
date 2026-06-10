# Proximal State Nudging: Reducing Skill Atrophy from AI Assistance
The gradual loss of human skills, or **skill atrophy**, is a rising concern as users increasingly rely on AI assistance. This problem is particularly salient in  cooperative AI systems, such as aircraft piloting or driving, where humans and AI agents jointly share control and decision-making. In these settings, human operators often struggle to disentangle which outcomes arise from AI intervention versus their own actions, undermining opportunities for their own learning and long-term skill retention. 

We propose **Proximal State Nudging (PSN)**, a shared autonomy algorithm that jointly optimizes for skill development and task performance by nudging users toward states estimated to be most learnable. 

We first show that PSN outperforms existing shared autonomy baselines in balancing student improvement in unassisted reward with overall shared performance, using simulated students in the classic LunarLander environment. We then present, to the best of our knowledge, the first human subject studies of a planner incorporating learning-compatible shared autonomy: across two driving tasks in the CARLA simulator (High Performance Racing and Parallel Parking, n = 60), PSN produces up to 7x larger gains in unassisted skill than standard blended shared autonomy, while incurring 50% fewer collisions than unassisted self-practice.

Authors: Megha Srivastava*, Jonathan Ouyang^, Eric Zhou^, Andrew Silva~, Emily Sumner~, Dorsa Sadigh*, Yuchen Cui^, Deepak Gopinath~, Guy Rosman~

Institutions: Stanford Computer Science (*), UCLA Robot Intelligence Lab (^), Toyota Research Institute (~)

Paper: https://arxiv.org/abs/2605.20355

Contact: megha@cs.stanford.edu


## Code Release


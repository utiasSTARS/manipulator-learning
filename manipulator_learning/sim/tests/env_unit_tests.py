# Some simple tests to make sure environments at least still run, if the package is modified.
# Note that this file doesn't really stress test the environments at all, and also doesn't actually confirm whether
# things like cameras, etc. are good. It's mostly helpful to see if any env-stopping bugs have been introduced.

from manipulator_learning.sim.envs import *

# options
acts_per_env = 10
stop_on_fail = True

globals_static = globals().copy()
env_prefixes = ['Thing', 'Panda']
ignore_suffixes = ['Generic']
envs = []
for key in globals_static:
    for pref in env_prefixes:
        if key.startswith(pref):
            keep = True
            for ignore_suf in ignore_suffixes:
                if key.endswith(ignore_suf):
                    keep = False
            if keep:
                envs.append(key)

print(f"Found the following envs: {envs}")
num_envs = len(envs)
passed_tests = 0
tried_tests = 0

for e_str in envs:
    print("------------------------------------------------------------------------------------------")
    print(f"Testing env {e_str}")
    print("------------------------------------------------------------------------------------------")
    try:
        env = globals()[e_str]()
        obs = env.reset()
        for acts in range(acts_per_env):
            next_obs, rew, done, info = env.step(env.action_space.sample())

        passed_tests += 1

    except Exception as e:
        print(f"Error in {e_str}: Exception {e}. Moving on...")
        if stop_on_fail:
            raise Exception(e)

    tried_tests += 1

    print("------------------------------------------------------------------------------------------")
    print(f"Test complete, {passed_tests}/{tried_tests} envs successfully tested.")
    print("------------------------------------------------------------------------------------------")
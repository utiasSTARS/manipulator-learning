# Manipulator Learning

This repository contains a set of manipulation environments that are compatible with [OpenAI Gym](https://gym.openai.com/) and simulated in [pybullet](pybullet.org), as well as a set of semi-generic imitation learning tools. In particular, we have a set of environments with a simulated version of our lab's mobile manipulator, the Thing, containing a [UR10](https://www.universal-robots.com/products/ur10-robot/) mounted on a [Ridgeback](https://clearpathrobotics.com/ridgeback-indoor-robot-platform/) base, as well as a set of environments using a table-mounted [Franka Emika Panda](https://www.franka.de/robot-system/).

The package environments (at `manipulator_learning/sim`) contain variations of the following tasks:
- Reach
- Lift
- Stack
- Pick and Place
- Sort
- Insert
- Pick and Insert
- Door Open
- Play (multitask)

For examples of what these tasks look like, see our work on [Multiview Manipulation](https://papers.starslab.ca/multiview-manipulation) and [Learning from Guided Play](https://papers.starslab.ca/lfgp).

The package learning tools (at `manipulator_learning/learning`) contain [TensorFlow 2.x](https://www.tensorflow.org/install) code for collecting human expert data (with a keyboard, gamepad, or HTC Vive VR hand controller), behaviour cloning and intervention-based learning for use with any gym-style environments, but with many extra added tools for environments contained in this package.

## Requirements
The environments in this package are agnostic to what package you use for learning, but the learning code works only with tensorflow. However, you can still use the environments without installing tensorflow or any of the other learning requirements.

### Sim Requirements
- python (3.7+)
- pybullet
- numpy
- gym
- transforms3d
- Pillow (for rendering)
- [liegroups](https://github.com/utiasSTARS/liegroups)

### Learning Requirements
- python (3.7+)
- numpy
- gym
- tensorflow-gpu (2.x)
- tensorflow-probability (matchiong tensorflow version)

### Additional Collection Device Requirements (optional)
If you want to collect human demonstrations, you'll also need some or all of these requirements:
- [inputs](https://github.com/trevorablett/inputs) (gamepad, keyboard)
- openvr (HTC Vive hand controller)

## Installation
```
git clone https://github.com/utiasSTARS/manipulator-learning
cd manipulator-learning && pip install .
```

### Learning Code installation
If you want to use the learning code in addition to the environments, hopefully you already have Tensorflow with GPU support. If not, this has worked for us:
1. [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#).
2. Create a new conda env to use for this work and activate it.
3. Run the following to install a version of TensorFlow that *may* work
```
conda install cudatoolkit cudnn
pip install tensorflow==2.6.* tensorflow-probability==0.14
```

### Human Demonstrations Installation
If you want to collect human demonstrations:
```
pip install -r device_requirements.txt
```
## Usage
### Environments
The easiest way to use environments in this repository is to import the whole `envs` module and then initialize using `getattr`. For example, to load our Panda Play environment with the insertion tray:

```
import manipulator_learning.sim.envs as manlearn_envs
env = getattr(manlearn_envs, 'PandaPlayInsertTrayXYZState')()

obs = env.reset()
next_obs, rew, done, info = env.step(env.action_space.sample())
```

You can also easily initialize the environment with a wide variety of different keyword arguments, e.g:
```
env = getattr(manlearn_envs, 'PandaPlayInsertTrayXYZState')(main_task='stack_01')
```

#### Image environments
All environments that are suffixed with `Image` or `Multiview` produce observations that contain RGB and depth images as well as numerical proprioceptive data. Here is an example of how you can access each type of data in these environments:
```
obs = env.reset()
img = obs['img']
depth = obs['depth']
proprioceptive = obs['obs']
```

By default, all image based environments render headlessly using EGL, but if you want to render the full pybullet GUI, you can using the `render_opengl_gui` and `egl` flags like this:
```
env = getattr(manlearn_envs, 'PandaPlayInsertTrayXYZState')(render_opengl_gui=True, egl=False)
```

### Imitation Learning
You can collect a set of demonstrations using `manipulator_learning/learning/imitation/collect_demos.py`. See the file for possible arguments.

A simple example:
```
python manipulator_learning/learning/imitation/collect_demos.py \
    --device keyboard \
    --directory demonstrations \
    --demo_name ThingDoorMultiview01 \
    --environment ThingDoorMultiview
```
To steer the robot, use `wasdqe` for 3-dof translation, and `rftgyh` for 3-dof rotation, and `Space` for opening/closing the gripper. Use `Enter` to start a demonstration, `Right Shift` to reset & save, and press and hold `Backspace` for >2s to delete the previous trajectory.

For examples of how the learning code can be used, see our [work on Multiview Manipulation](https://github.com/utiasSTARS/multiview-manipulation).

## Environment Details

### Thing (mobile manipulator) environments
Our mobile manipulation environments were primarily designed to allow base position changes between task episodes, but don't actually allow movement during an episode. For this reason, many included environments include both an `Image` version and a `Multiview` version, where all observation and control parameters are identical, except that the base is fixed in the `Image` version, and the base moves (between episodes) in the `Multiview` version. See, for example, `manipulator_learning/sim/envs/thing_door.py`.

### Panda Environments
Our panda environments contain several of the same tasks as our Thing environments. Additionally, we have a set of "play" environments that are multitask.

### Current environment list
```
['PandaPlayXYZState', 
'PandaPlayInsertTrayXYZState', 
'PandaPlayInsertTrayDPGripXYZState', 
'PandaPlayInsertTrayPlusPickPlaceXYZState', 
'PandaLiftXYZState', 
'PandaBringXYZState', 
'PandaPickAndPlaceAirGoal6DofState', 
'PandaReachXYZState', 
'PandaStackXYZState',
'ThingInsertImage', 
'ThingInsertMultiview', 
'ThingPickAndInsertSucDoneImage', 
'ThingPickAndInsertSucDoneMultiview',
'ThingPickAndPlaceXYState', 
'ThingPickAndPlacePrevPosXYState', 
'ThingPickAndPlaceGripPosXYState', 
'ThingPickAndPlaceXYZState', 
'ThingPickAndPlaceGripPosXYZState', 
'ThingPickAndPlaceAirGoalXYZState', 
'ThingPickAndPlace6DofState', 
'ThingPickAndPlace6DofLongState', 
'ThingPickAndPlace6DofSmallState', 
'ThingPickAndPlaceAirGoal6DofState', 
'ThingBringXYZState',
'ThingLiftXYZStateMultiview',
'ThingLiftXYZState', 
'ThingLiftXYZMultiview', 
'ThingLiftXYZImage', 
'ThingPickAndPlace6DofSmallImage', 
'ThingPickAndPlace6DofSmall160120Image', 
'ThingPickAndPlace6DofSmallMultiview', 
'ThingSort2Multiview', 
'ThingSort3Multiview', 
'ThingPushingXYState', 
'ThingPushingXYImage', 
'ThingPushing6DofMultiview', 
'ThingReachingXYState', 
'ThingReachingXYImage', 
'ThingStackImage', 
'ThingStackMultiview', 
'ThingStackSmallMultiview', 
'ThingStackSameMultiview', 
'ThingStackSameMultiviewV2', 
'ThingStackSameImageV2', 
'ThingStack3Multiview', 
'ThingStackTallMultiview', 
'ThingDoorImage', 
'ThingDoorMultiview']
```

## Roadmap
- [ ] Make environment generation compatible with `gym.make`
- [ ] Documentation for environments and options for customization
- [x] Add imitation learning/data collection code
- [ ] Fix bug that timesteps remaining on rendered window takes an extra step to update
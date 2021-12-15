# Manipulator Learning

This repository contains a set of manipulation environments that are compatible with [OpenAI Gym](https://gym.openai.com/) and simulated in [pybullet](pybullet.org). In particular, we have a set of environments with a simulated version of our lab's mobile manipulator, the Thing, containing a [UR10](https://www.universal-robots.com/products/ur10-robot/) mounted on a [Ridgeback](https://clearpathrobotics.com/ridgeback-indoor-robot-platform/) base, as well as a set of environments using a table-mounted [Franka Emika Panda](https://www.franka.de/robot-system/).

The package currently contains variations of the following tasks:
- Reach
- Lift
- Stack
- Pick and Place
- Sort
- Insert
- Pick and Insert
- Door Open
- Play (multitask)

## Requirements

- python (3.7+)
- pybullet
- numpy
- gym
- transforms3d
- Pillow (for rendering)
- [liegroups](https://github.com/utiasSTARS/liegroups)

## Installation
```
git clone https://github.com/utiasSTARS/manipulator-learning
cd manipulator-learning && pip install .
```

## Usage
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

### Image environments
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

## Environment Details

### Thing (mobile manipulator) environments
Our mobile manipulation environments were primarily designed to allow base position changes between task episodes, but don't actually allow movement during an episode. For this reason, many included environments include both an `Image` version and a `Multiview` version, where all observation and control parameters are identical, except that the base is fixed in the `Image` version, and the base moves (between episodes) in the `Multiview` version. See, for example, `manipulator_learning/sim/envs/thing_door.py`.

### Panda Environments
Our panda environments contain several of the same tasks as our Thing environments. Additionally, we have a set of "play" environments that are multi-task.

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
- [ ] Add imitation learning/data collection code
- [ ] Fix bug that timesteps remaining on rendered window takes an extra step to update
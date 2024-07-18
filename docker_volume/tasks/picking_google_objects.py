import os

import numpy as np
from tasks.task import Task
from primitives.pick_and_place import Pick
from agents.oracle_agent import OracleAgent
from scipy.signal import correlate
from tasks import utils
import matplotlib.pyplot as plt

import pybullet as p


class PickingSeenGoogleObjectsSeq(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.primitive = Pick()
        self.lang_template = "pick up {obj}"
        self.task_completed_desc = "done picking objects."
        self.obj_names = self.get_object_names()
        self.loaded_obj_names = {}

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                # 'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                # 'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                # 'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                # 'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                # 'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                # 'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                # 'scissors',
                'screwdriver',
                'silver tape',
                # 'spatula with purple head',
                'spiderman figure',
                # 'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'valid': [
                'alarm clock',
                'android toy',
                # 'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                # 'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                # 'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                # 'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                # 'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                # 'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                # 'scissors',
                'screwdriver',
                'silver tape',
                # 'spatula with purple head',
                'spiderman figure',
                # 'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'test': [
                'alarm clock',
                'android toy',
                # 'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                # 'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                # 'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                # 'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                # 'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                # 'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                # 'scissors',
                'screwdriver',
                'silver tape',
                # 'spatula with purple head',
                'spiderman figure',
                # 'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
        }

    def reset(self, env):
        super().reset(env)

        # object names
        obj_names = self.obj_names[self.mode]

        # Add Google Scanned Objects to scene.
        obj_uids = []
        obj_descs = []

        n_objs = np.random.randint(1, 6)
        size = (0.1, 0.1, 0.1)
        obj_scale = 0.5
        obj_template = 'google/object-template.urdf'
        chosen_objs = self.choose_objects(obj_names, n_objs)
        self.loaded_obj_names = {}

        for i in range(n_objs):

            pose = self.get_random_pose(env, size)

            # Add object only if valid pose found.
            if pose[0] is not None:
                # Initialize with a slightly tilted pose so that the objects aren't always erect.
                slight_tilt = utils.q_mult(
                    pose[1], (-0.1736482, 0, 0, 0.9848078))
                ps = ((pose[0][0], pose[0][1], pose[0][2]+0.05), slight_tilt)

                obj_name = chosen_objs[i]
                obj_name_with_underscore = obj_name.replace(" ", "_")
                mesh_file = os.path.join(self.assets_root,
                                         'google',
                                         'meshes_fixed',
                                         f'{obj_name_with_underscore}.obj')
                texture_file = os.path.join(self.assets_root,
                                            'google',
                                            'textures',
                                            f'{obj_name_with_underscore}.png')

                try:
                    replace = {'FNAME': (mesh_file,),
                               'SCALE': [obj_scale, obj_scale, obj_scale],
                               'COLOR': (0.2, 0.2, 0.2)}
                    urdf = self.fill_template(obj_template, replace)
                    obj_uid = env.add_object(urdf, ps)
                    if os.path.exists(urdf):
                        os.remove(urdf)
                    obj_uids.append(obj_uid)

                    texture_id = p.loadTexture(texture_file)
                    p.changeVisualShape(
                        obj_uid, -1, textureUniqueId=texture_id)
                    p.changeVisualShape(obj_uid, -1, rgbaColor=[1, 1, 1, 1])

                    obj_descs.append(obj_name)
                    self.loaded_obj_names[obj_uid] = obj_name_with_underscore

                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(obj_name_with_underscore, mesh_file, texture_file)
                    print(f"Exception: {e}")

        self.set_goals(obj_uids, obj_descs)

        for i in range(480):
            p.stepSimulation()

    def choose_objects(self, object_names, k):
        return np.random.choice(object_names, k, replace=False)

    def set_goals(self, obj_uids, obj_descs):
        n_pick_objs = 1
        obj_uids = obj_uids[:n_pick_objs]

        for obj_idx, obj_uid in enumerate(obj_uids):
            self.goals.append(
                {"obj_uid": obj_uid, "max_reward": (1/len(obj_uids))})
            self.lang_goals.append(
                self.lang_template.format(obj=obj_descs[obj_idx]))

    def reward(self, env):
        """Get delta rewards for current timestep.

        Returns:
          A tuple consisting of the scalar (delta) reward, plus `extras`
            dict which has extra task-dependent info from the process of
            computing rewards that gives us finer-grained details. Use
            `extras` for further data analysis.
        """
        # Unpack next goal step.
        obj_uid = self.goals[0]["obj_uid"]
        max_reward = self.goals[0]["max_reward"]

        def obj_airborne(obj_uid):
            obj_z = p.getBasePositionAndOrientation(obj_uid)[0][2]
            if obj_z > 0.2:
                return True
            return False

        # Move to next goal step if current goal step is complete.
        if obj_airborne(obj_uid) and env.robot.ee.obj_grasped(obj_uid):
            self.goals.pop(0)
            self.lang_goals.pop(0)
            self._rewards += max_reward

        return self._rewards

    # -------------------------------------------------------------------------
    # Oracle Agent
    # -------------------------------------------------------------------------

    def create_oracle_agent(self, env) -> OracleAgent:
        self.n_random_poses = 50
        self.idx_random_pose = 0

        def act(obs, info):  # pylint: disable=unused-argument
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            _, hmap, obj_mask = self.get_true_image(env)

            # Unpack next goal step.
            obj_uid = self.goals[0]["obj_uid"]
            pick_mask = np.uint8(obj_mask == obj_uid)

            # Trigger task reset if no object is visible.
            if np.sum(pick_mask) == 0:
                self.goals = []
                self.lang_goals = []
                print('Object for pick is not visible. Skipping demonstration.')
                return

            # Get picking pix
            pick_prob = np.float32(pick_mask)
            pick_pix = utils.sample_distribution(
                pick_prob, self.n_random_poses)
            # Get random orientation
            min_azimuth = -np.pi / 2
            max_azimuth = np.pi / 2
            min_polar = 0
            max_polar = np.pi / 4
            azimuth = np.random.uniform(
                min_azimuth, max_azimuth, self.n_random_poses)
            cos_polar = np.random.uniform(
                np.cos(max_polar), np.cos(min_polar), self.n_random_poses)
            polar = np.arccos(cos_polar)

            # Generate poses from pix and random orientation
            # NOTE: radius is almost 0 because radius of 0 would lead to xyz = (0, 0, 0) which is bad if you want to get a direction of z_axis
            pick_pos = np.array(utils.pix_to_xyz(pick_pix[self.idx_random_pose], hmap,
                                                 self.bounds, self.pix_size))
            azi = azimuth[self.idx_random_pose]
            pol = polar[self.idx_random_pose]
            pick_pose = utils.get_pose_on_sphere(
                azi, pol, radius=1e-6, sph_pos=pick_pos)

            # Chose most feasible z
            obj_z = p.getBasePositionAndOrientation(obj_uid)[0][2]
            pick_pose_z = pick_pose[0][2]
            if obj_z < pick_pose_z:
                pick_pos = ((pick_pose[0][0], pick_pose[0][1], obj_z),
                            (pick_pose[1][0], pick_pose[1][1], pick_pose[1][2], pick_pose[1][3]))

            self.idx_random_pose += 1
            return [pick_pose]

        return OracleAgent(act)

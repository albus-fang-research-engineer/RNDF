import sys
sys.path.append("../")
import time
import copy
import trimesh
import numpy as np
from utils import robot_kinematic

def wrap_to_pi(q):
    return (q + np.pi) % (2*np.pi) - np.pi
class DataSampler(robot_kinematic):
    def __init__(self, urdf_path=None, dataset_path=None):
        super().__init__(urdf_path)
        self.dataset_path = dataset_path

    # def get_link_mesh(self, link_name, joint_positions=None):
    #     assert link_name in self.link_names
    #     if joint_positions is None:
    #         fk = self.robot.link_fk(cfg=self.robot_joints)
    #     else:
    #         robot_config = copy.deepcopy(self.robot_joints)
    #         for i, name in self.link_names:
    #             robot_config[name] = joint_positions[i]
    #         fk = self.robot.link_fk(cfg=self.robot_joints)

    #     robot_meshes = self.robot_links_mesh
    #     mesh = robot_meshes[link_name].copy()
    #     link_index = self.link_names.index(link_name)
    #     mesh = mesh.apply_transform(fk[self.robot.links[link_index]])
    #     return mesh
    def get_link_mesh(self, link_name, joint_positions=None):

        assert link_name in self.link_names

        if joint_positions is None:
            fk = self.robot.link_fk(cfg=self.robot_joints)
        else:
            robot_config = copy.deepcopy(self.robot_joints)
            for i, joint_name in enumerate(self.joint_names):
                robot_config[joint_name] = joint_positions[i]
            fk = self.robot.link_fk(cfg=robot_config)

        mesh = self.robot_links_mesh[link_name].copy()

        link_obj = self.robot.link_map[link_name]
        mesh.apply_transform(fk[link_obj])

        return mesh


    def set_robot_joints(self, positions):
        positions = wrap_to_pi(positions)
        assert len(positions) == self.num_joints
        for i in range(self.num_joints):
            assert (self.joint_lower_bound[i] < positions[i]) & (positions[i] < self.joint_upper_bound[i])

        for i in range(self.num_joints):
            self._robot_joints[self.joint_names[i]] = positions[i]

    def whole_arm_normal_sampling(self,
                                  base_num=10,
                                  offset_range=None,
                                  joint_positions=None):

        if offset_range is None:
            offset_range = [0, 0.03]

        if joint_positions is not None:
            self.set_robot_joints(joint_positions)

        sampled_points = []
        for i, name in enumerate(self.link_names):
            link_mesh = self.get_link_mesh(name)
            points = self.random_range_normal_sampling(link_mesh,
                                                       offset=offset_range,
                                                       num_points=base_num)
            sampled_points.append(points)
        return np.vstack(sampled_points)

    def whole_arm_inside_sampling(self,
                                  base_num=5,
                                  joint_positions=None,
                                  link_weights=None):
        # trimesh uses a rejection-based sampling method
        # mesh with intricate geometry needs to sample more
        if link_weights is None:
            link_weights = [1, 1, 1, 1, 1, 1, 1]
        assert len(link_weights) == self.num_links

        if joint_positions is not None:
            self.set_robot_joints(joint_positions)
        sampled_points = []
        for i, name in enumerate(self.link_names):
            link_mesh = self.get_link_mesh(name)

            points = self.random_sample_inside_mesh(link_mesh, num_points=int(base_num * link_weights[i]))
            sampled_points.append(points)

        return np.vstack(sampled_points)


    def batch_calculate_signed_distance(self, sampled_points):
        signed_distance_links = []
        for name in self.link_names:
            link_mesh = self.get_link_mesh(name)
            # print(name, "watertight:", link_mesh.is_watertight)

            signed_distance = -trimesh.proximity.signed_distance(link_mesh, sampled_points)
            signed_distance_links.append(signed_distance)

        return np.asarray(signed_distance_links).transpose()

    def batch_sample_inside_mesh(self, batch_size, base_num, link_weights=None):
        iter_time = 0
        batch_data = []
        while iter_time < batch_size:
            rand_q = self.sample_random_robot_config()
            self.set_robot_joints(rand_q)
            if self.self_collision_detected():
                continue
            sample_points = self.whole_arm_inside_sampling(base_num=base_num, link_weights=link_weights)
            signed_dist = self.batch_calculate_signed_distance(sample_points)
            rand_q = np.asarray([rand_q for _ in range(len(sample_points))])

            iter_data = np.concatenate((rand_q, sample_points, signed_dist), axis=1)
            batch_data.append(iter_data)
            iter_time += 1
            if iter_time % 20 == 0:
                print("Data Generation {} with base_num: {} progress {}/{}".format(iter_data.shape,
                                                                                   base_num,
                                                                                   iter_time,
                                                                                   batch_size))
        # combined data and define data type
        batch_data = np.vstack(batch_data).astype(np.float32)
        print("Saving generated data with shape of {}".format(batch_data.shape))
        np.save(self.dataset_path + '/inside/Inside_{}_{}'.format(str(time.time())[-4:],
                                                                      batch_data.shape[0]),
                batch_data)

    def batch_sample_outside_mesh(self, batch_size, base_num, offset_range=None):

        if offset_range is None:
            offset_range = [0, 0.03]

        iter_time = 0
        batch_data = []
        while iter_time < batch_size:
            rand_q = self.sample_random_robot_config()
            self.set_robot_joints(rand_q)
            if self.self_collision_detected():
                print("self-collision detected, resampling robot configuration", rand_q)
                continue
            sample_points = self.whole_arm_normal_sampling(base_num=base_num, offset_range=offset_range)
            signed_dist = self.batch_calculate_signed_distance(sample_points)
            rand_q = np.asarray([rand_q for _ in range(len(sample_points))])
            iter_data = np.concatenate((rand_q, sample_points, signed_dist), axis=1)
            batch_data.append(iter_data)
            iter_time += 1
            if iter_time % 50 == 0:
                print("Data Generation {} with base_num: {} progress {}/{}".format(iter_data.shape,
                                                                                   base_num,
                                                                                   iter_time,
                                                                                   batch_size))
        # combined data and define data type
        batch_data = np.vstack(batch_data).astype(np.float32)
        print("Saving generated data with shape of {}".format(batch_data.shape))
        np.save(self.dataset_path + '/outside/Outside_{}_{}_{}_{}'.format(str(time.time())[-4:],
                                                                              offset_range[0],
                                                                              offset_range[1],
                                                                              batch_data.shape[0]),
                batch_data)

    def random_sample_inside_mesh(self, sampled_mesh, num_points=200):
        """
        sample inside points w.r.t. a given mesh
        rejection-based sampling
        """
        sampled_points = trimesh.sample.volume_mesh(sampled_mesh, count=num_points)
        return np.asarray(sampled_points)

    def random_range_normal_sampling(self, sampled_mesh, offset=None, num_points=3000):
        """
        sample outside points w.r.t. a given mesh
        random distance * normal vector
        """
        if offset is None:
            offset = [0.2, 1]
        vertices, faces = sampled_mesh.sample(num_points, return_index=True)
        normals = sampled_mesh.face_normals[faces]
        rand_dist = np.random.uniform(low=offset[0], high=offset[1], size=(len(vertices), 1))
        rand_dist = np.repeat(rand_dist, 3, axis=1)
        sampled_points = vertices - normals * rand_dist
        return sampled_points

    def create_point_cloud_scene(self, sampled_points, mesh, point_radius=0.002):
        scene = trimesh.Scene()
        for point in sampled_points:
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=point_radius, face_colors=np.random.uniform(size=4))
            sphere.apply_translation(point)
            scene.add_geometry(sphere)

        scene.add_geometry(mesh)
        return scene
    
    def sample_uniform_workspace(self, num_points, margin=0.3):
        mesh = self.get_combined_mesh(convex=False, bounding_box=False)
        # bounds = mesh.bounds
        bounds = np.array([
            [-0.9, 0.9],
            [-0.9, 0.9],
            [ 0.3, 0.9]
        ])
        bounds_min = bounds[:,0] - margin
        bounds_max = bounds[:,1] + margin

        pts = np.random.uniform(
            low=bounds_min,
            high=bounds_max,
            size=(num_points, 3)
        )
        return pts
    
    def sample_mixed_points(self, uniform_base_num = 500):
        """
        Generate PR, PN, PI for ONE robot configuration
        """

        # --- PR (uniform) ---
        PR = self.sample_uniform_workspace(uniform_base_num)

        # --- PN (near surface) ---
        PN = self.whole_arm_normal_sampling(
            base_num=10,
            offset_range=[0, 0.1]
        )

        # --- PI (inside) ---
        PI = self.whole_arm_inside_sampling(
            base_num=20
        )

        return np.vstack([PR, PN, PI])
    
    def batch_sample_mixed(self, batch_size, base_num):

        batch_data = []
        iter_time = 0

        while iter_time < batch_size:

            rand_q = self.sample_random_robot_config()
            rand_q = wrap_to_pi(rand_q)
            self.set_robot_joints(rand_q)

            if self.self_collision_detected():
                continue

            pts = self.sample_mixed_points(base_num)

            sd = self.batch_calculate_signed_distance(pts)

            q_repeat = np.tile(rand_q, (len(pts), 1))

            data = np.concatenate((q_repeat, pts, sd), axis=1)

            batch_data.append(data)
            iter_time += 1
            if iter_time % 10 == 0 or iter_time == batch_size:
                print_progress(iter_time, batch_size)
        return np.vstack(batch_data)
def print_progress(iter_time, batch_size):
    progress = iter_time / batch_size

    print(f"[{iter_time}/{batch_size}] "
          f"{progress*100:.1f}%") 
if __name__ == "__main__":
    np.random.seed(26)

    robo = DataSampler(dataset_path='../dataset_new_scheme/')

    # --- sample ONE valid robot configuration ---
    while True:
        q = robo.sample_random_robot_config()
        # q = np.array([0, -1.57, 0, -1.57, 0, 0])
        robo.set_robot_joints(q)
        if not robo.self_collision_detected():
            break
    
    print("Using joint configuration:", q)

    # --- get robot mesh ---
    robot_mesh = robo.get_combined_mesh(convex=False, bounding_box=False)

    # --- sample mixed points ---
    pts = robo.sample_mixed_points(uniform_base_num=500)

    print(f"Total sampled points: {pts.shape[0]}")

    # --- create visualization ---
    def create_axes(length=0.3):
        axis = trimesh.creation.axis(origin_size=0.08)
        axis.apply_scale(length)
        return axis
    scene = trimesh.Scene(create_axes())

    # add robot mesh
    scene.add_geometry(robot_mesh)

    # add sampled points
    for p in pts:
        sphere = trimesh.creation.icosphere(
            subdivisions=1,
            radius=0.01,
            face_colors=[0, 255, 0, 150]  # green points
        )
        sphere.apply_translation(p)
        scene.add_geometry(sphere)

    # show
    scene.show()

    batch_size = 3000          # number of configurations
    uniform_base_num = 500     # PR samples

    print("Starting dataset generation...")

    data = robo.batch_sample_mixed(
        batch_size=batch_size,
        base_num=uniform_base_num
    )

    print("Dataset shape:", data.shape)

    save_path = robo.dataset_path + "/mixed_dataset.npy"
    np.save(save_path, data)

    print(f"Saved dataset to: {save_path}")
'''
if __name__ == "__main__":
    np.random.seed(16)
    robo = DataSampler(dataset_path='../dataset/')
    # base_mesh = robo.get_link_mesh("upper_arm_link")

    # scene = trimesh.Scene()
    # scene.add_geometry(base_mesh)
    # scene.show()
    # random joint configuration
    sampled_q = robo.sample_random_robot_config()
    print("Sampled q:", sampled_q)
    sampled_q = np.array([0, -1.57, 0, -1.57, 0, 0])
    # print("Current q:", robo.robot_q)
    robo.set_robot_joints(sampled_q)

    # visualize sampled joint configuration
    # robo.show_robot_meshes(convex=False, bounding_box=False)
    # robo.show_robot_meshes(convex=False, bounding_box=True)

    print("link_names",robo.link_names)
    print("joint_names",robo.joint_names)
    print("self-collision detected: {}".format(robo.self_collision_detected()))

    combined_mesh = robo.get_combined_mesh(convex=False, bounding_box=False)
    # ----- create ONE test point in workspace -----
    test_point = np.array([[0.0, 0.0, 1.1]])   # shape must be (1, 3)

    # ----- compute signed distance to every link -----
    sd = robo.batch_calculate_signed_distance(test_point)
    # sd shape = (1, num_links)

    print("\nSigned distance per link for point:", test_point[0])
    for name, dist in zip(robo.link_names, sd[0]):
        print(f"{name:20s}: {dist: .6f}")

    # ----- visualize the point with the robot -----
    scene = robo.create_point_cloud_scene(
        sampled_points=test_point,
        mesh=combined_mesh,
        point_radius=0.02
    )
    scene.show()

    # sampled points outside
    outside_points = robo.whole_arm_normal_sampling(offset_range=[0.4, 0.5], base_num=50)
    scene_outside = robo.create_point_cloud_scene(outside_points, combined_mesh, point_radius=0.02)
    scene_outside.show()
    print("signed distance:", robo.batch_calculate_signed_distance(outside_points))

    # sampled points inside (well you may not see it without zooming in)
    inside_points = robo.whole_arm_inside_sampling(base_num=10)
    scene_inside = robo.create_point_cloud_scene(inside_points, combined_mesh, point_radius=0.02)
    scene_inside.show(viewer="gl")
    print("signed distance:", robo.batch_calculate_signed_distance(inside_points))

    # batch sample outside
    robo.batch_sample_outside_mesh(batch_size=500, base_num=10, offset_range=[0., 0.1])

    # batch sample inside
    robo.batch_sample_inside_mesh(batch_size=500, base_num=10, link_weights=[1, 1, 1, 2, 3, 3, 3])
'''

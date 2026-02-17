from urdfpy import URDF
import pyrender

robot = URDF.load("ur5e.urdf")
robot.show()



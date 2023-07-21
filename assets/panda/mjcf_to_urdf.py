import pybullet_utils as pu
import pybullet_utils.bullet_client as bulllet_client
import pybullet_utils.urdfEditor as urdfEditor

def main():
    client = bulllet_client.BulletClient()
    objs = client.loadMJCF(
        'robot.mjcf')# , flags=client.URDF_USE_IMPLICIT_CYLINDER)

if __name__ == "__main__":
    main()

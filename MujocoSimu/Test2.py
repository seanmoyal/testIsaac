import time
import numpy as np

import mujoco
import mujoco.viewer

from MujocoSimu.ObjetsEnvironnement.AlbertCube import EulerFromQuaternion
from XmlConversionDirectory.xmlMerger import mergeMjcfFiles

#roomManagerxml = "C:/Users/moyal/PycharmProjects/testEnviSim/xmlDirectory/Room1bis.xml"
#Albertxml = "C:/Users/moyal/PycharmProjects/testEnviSim/xmlDirectory/Actor.xml"
# faire une fo qui construit l'objet room à partir du xml
#mergeMjcfFiles(roomManagerxml, Albertxml, "AlbertEnvironment")


def euler_to_rotation_matrix(euler):
  # Convert Euler angles to rotation matrix
  # R = Rz(yaw) * Ry(pitch) * Rx(roll)
  roll = euler[0]
  pitch = euler[1]
  yaw = euler[2]
  cos_r, sin_r = np.cos(roll), np.sin(roll)
  cos_p, sin_p = np.cos(pitch), np.sin(pitch)
  cos_y, sin_y = np.cos(yaw), np.sin(yaw)

  rotation_matrix = np.array([
    [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
    [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
    [-sin_p, cos_p * sin_r, cos_p * cos_r]
  ])

  return rotation_matrix


AlbertEnvironment = "C:/Users/moyal/PycharmProjects/testEnviSim/xmlDirectory/AlbertEnvironment.xml"

m = mujoco.MjModel.from_xml_path(AlbertEnvironment)


"""
mujoco.mj_objectVelocity()
mujoco.mj_rayMesh()
mujoco.mj_ray()
mujoco.mj_collision()
"""


id=m.body("Albert").id


#m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)


def transform_point_with_quaternion_and_translation(point, quaternion, translation):
  # Normalize the quaternion
  quaternion /= np.linalg.norm(quaternion)

  # Convert the point to a quaternion
  point_quat = np.concatenate(([0], point))

  # Perform quaternion multiplication to rotate the point to the new frame
  rotated_point_quat = np.quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]) * np.quaternion(
    point_quat[0], point_quat[1], point_quat[2], point_quat[3]) * np.conj(
    np.quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]))

  # Extract the rotated point from the quaternion
  rotated_point = np.array([rotated_point_quat.x, rotated_point_quat.y, rotated_point_quat.z])

  # Translate the point to the new frame
  translated_point = rotated_point + translation

  return translated_point
def raycasting(model,data,viewer,id):
  cubePos = data.xpos[id]
  ori = data.xquat[id]

  cubeOri = EulerFromQuaternion(ori)

  matriceOri = euler_to_rotation_matrix(cubeOri)
  rayLength = 5

  # départ des angles :
  depAnglesYaw = -35 * np.pi / 180
  depAnglesPitch = -10 * np.pi / 180
  # Pas yaw pour 70°
  stepYaw = 70 / 6
  stepYawRad = stepYaw * np.pi / 180

  # pas pitch pour 70°
  stepPitch = 20 / 2
  stepPitchRad = stepPitch * np.pi / 180

  # rayVec1 : premier rayon droit devant le cube
  rayVects = []
  for i in range(3):
    for n in range(7):
      baseRay = [np.cos((n * stepYawRad + depAnglesYaw)) * np.cos((i * stepPitchRad + depAnglesPitch)),
                 np.sin((n * stepYawRad + depAnglesYaw)), np.sin((i * stepPitchRad + depAnglesPitch))]
      normRay = np.linalg.norm(baseRay)

      a=np.dot(matriceOri, np.array(
        [(baseRay[0] / normRay * rayLength),
         (rayLength * baseRay[1] / normRay),
         (rayLength * baseRay[2] / normRay)
         ]
      ))
      #print("avant : "+str(a[0]))
      a[0]+=cubePos[0]
      #print("apres : "+str(a[0]))
      a[1] += cubePos[1]
      a[2] += cubePos[2]

      rayVects.append(a)
  # tracé du rayon / faudra changer avec rayTestBatch() quand on aura plus de rayons
  contactResults = []
  geomIds=np.empty([21])
  geomID = np.array([130], dtype='int32')# ID du geom de l'acteur mais en vrai ca doit pas etre ca qu'il faut
  for n in range(21):
    contactResults.append(
      mujoco.mj_ray(model,data,pnt=cubePos,vec=rayVects[n],geomgroup=None, flg_static=1, bodyexclude=131, geomid=geomID))  ################################################################################
    geomIds[n]=geomID
  # dans les resultats [0] : hitObjectId // [3] hit position
  # faudra changer les coordonnées de globales à locales
  for n in range(21):
    #if contactResults[n] != -1:
      mujoco.mjv_initGeom(viewer.scn.geoms[n],
                        mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba=np.array([1., 0., 0., 1.], dtype=np.float32))
      mujoco.mjv_makeConnector(viewer.scn.geoms[n],mujoco.mjtGeom.mjGEOM_LINE,width=5,a0=cubePos[0],a1=cubePos[1],a2=cubePos[2],b0=rayVects[n][0],b1=rayVects[n][1],b2=rayVects[n][2])

  return contactResults,geomIds


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.


  start = time.time()
  while viewer.is_running() and time.time() - start < 1000:
    step_start = time.time()
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.


    print(d.qpos)
    res,geomIds=raycasting(m, d, viewer, id)
    mujoco.mj_step(m, d)


    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
import gym
import gymtorch
sim=1
num_envs=50
actors_per_env=200

# lignes pour regarder toutes les pos,ori,vit,vit_ang, à faire un fois, avant que la sim commence
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor) # wrap it to acces the data
# mettre une vue : vecteur d'environements
root_states_vec = root_tensor.view(num_envs,actors_per_env,13)
root_positions = root_states_vec[..., 0:3]
root_orientations = root_states_vec[..., 3:7]
root_linvels = root_states_vec[..., 7:10]
root_angvels = root_states_vec[..., 10:13]

# pour update ( dans la boucle de simulation ) ca : ( à mettre après gym.simulate(sim) )
gym.refresh_actor_root_state(sim)

# pour modifier mettre ca aprees avoir changé le tenseur _root_tensor du début:
gym.set_actor_root_state_tensor(sim,_root_tensor)

# APPLYING CONTROLS


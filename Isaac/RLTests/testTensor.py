from Isaac.ClassFrameWorkTensor import AlbertEnvironment
import torch
env=AlbertEnvironment()
env.reset(reset_tensor=torch.full((env.num_envs,),True))

episodes = 10


while not env.gym.query_viewer_has_closed(env.viewer):
    env.render()
    env.pre_physics_step(env.action_space.sample())
    env.post_physics_step()
    # step the physics
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    # update the viewer
    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    env.gym.sync_frame_time(env.sim)

env.gym.destroy_viewer(env.viewer)
env.gym.destroy_sim(env.sim)

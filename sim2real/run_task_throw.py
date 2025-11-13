import onnxruntime as ort
import numpy as np

# Load ONNX model
ort_sess = ort.InferenceSession("/home/nur/RL_catch/policy.onnx")

# Create dummy input (batch size = 1, 38 features)
dummy_input = np.random.randn(1, 38).astype(np.float32)

# Run inference
outputs = ort_sess.run(None, {"obs": dummy_input})

print("Dummy input:", dummy_input)
print("Model output:", outputs[0])
## policy_obs
    #  1  joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    #     can be got from the robot
    #  2  joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    #     can be got from the robot  
    #  3  object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame) 
    #     
    #  4  obj_ee_tracking=ObsTerm(func=mdp.object_ee_distance_obs, params={"std": 0.1})
    #
    #  5  Obj_bsk_tracking=ObsTerm(func=mdp.obj_bsk_distance_obs)
    #
    #  6  target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "Target_pose"})
    #   
    #  7  actions = ObsTerm(func=mdp.last_action)
    #
    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True
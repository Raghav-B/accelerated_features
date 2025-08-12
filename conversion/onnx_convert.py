import qai_hub as hub
import os

# Step 1: Using onnx model from https://github.com/meyiao/xfeatc/blob/main/model/xfeat_640x640.onnx
onnx_model_path = "./xfeat_640x640.onnx"
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX file not found")

# Step 2: Compile model
compile_job = hub.submit_compile_job(
    model=onnx_model_path,
    device=hub.Device("QCS6490 (Proxy)"),
    input_specs=dict(input=(1, 1, 640, 640)),
    options="--compute_unit cpu",
)

compile_job.wait()
print("Compile job status:", compile_job.get_status, compile_job.get_status().message)

# Step 3: Profile on cloud-hosted device
target_model = compile_job.get_target_model()
profile_job = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("QCS6490 (Proxy)"),
)

assert isinstance(compile_job, hub.CompileJob)

from d455_full_pipeline import run_vlm_and_validator, run_spatial_pipeline_from_validation

run_vlm_and_validator("aku mau minum soda")

result = run_spatial_pipeline_from_validation()

print(result)
print("3D Camera Point:", result["object_position"]["point_camera_m"])
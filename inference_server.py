from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="eytDtQ1Q75OZyEFEgHNF"
)
with client.use_model(model_id="smokey-flaring-detection/2"):
    predictions = client.infer("/workspaces/energy_industry_computer_vision/videos/flare_video_1.mp4")




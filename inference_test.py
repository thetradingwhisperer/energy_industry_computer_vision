from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import VideoFileSink

var = 1
video_sink = VideoFileSink.init(video_file_name=f"output{var}.avi")
pipeline = InferencePipeline.init(
        model_id="smokey-flaring-detection/2",
        video_reference=f"./videos/flare_video_{var}.mp4",
        on_prediction=video_sink.on_prediction,
        api_key="eytDtQ1Q75OZyEFEgHNF"
    )

pipeline.start()
pipeline.join()
video_sink.release()



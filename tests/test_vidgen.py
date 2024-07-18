from models.video_retalking import video_inference # lip-sync video gen model
import logging

logger = logging.getLogger(__name__)

videoretalk = video_inference.VideoRetalking(
    'cpu', 'weights/checkpoints/'
)

vid_clip_path = 'tests/clips/biden_clip.mp4'
speaker_audio = 'tests/clips/biden_clip.wav'

gen_vid_path = videoretalk.main(logger, vid_clip_path, speaker_audio)

print('Test Passed!')
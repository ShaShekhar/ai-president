from models.coqui.tts import TTS # load tts model

tts = TTS(model_path="weights/coqui/model.pth", config_path="weights/coqui/config.json", progress_bar=False, gpu=True)

text = "I am joe biden and i'm the president of United states. I'm running for president in 2024 and i approve this message."

tts.tts_to_file(text=text, speaker_wav='tests/clips/biden_clip.wav', language="en",
                file_path="/tests/clips/clone_biden.wav", speed=1.30, split_sentences=True)
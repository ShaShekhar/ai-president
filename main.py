import logging
import os
import importlib
from omegaconf import OmegaConf
from dotenv import load_dotenv
from models.faster_whisper import WhisperModel # audio transcription model
from models.coqui.tts import TTS # load tts model
from models.video_retalking import video_inference # lip-sync video gen model

load_dotenv()
config = OmegaConf.load("configs/inference.yaml")

logging.basicConfig(filename=config.LOG_FILE_PATH, format='%(asctime)s: %(levelname)s: %(message)s',
   level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)

whisper_model = WhisperModel(**config.models.faster_whisper)
logging.info("Successfully loaded Whisper Model!")

llm_type = config.models.language_model.active_model

if llm_type == "local":
    local_attrib = config.models.language_model.local
    download_dir = config.models.language_model.local.download_dir
    local_model = local_attrib.model_name # phi3
    logging.info(f"Using local language model: {local_model}")

    local_module = importlib.import_module(f"models.language_models.{local_model}")
    model_class = getattr(local_module, local_attrib.model_class)
    language_model = model_class(local_attrib.device, download_dir)
else:
    provider = config.models.language_model.api.provider
    logging.info(f"Using {provider} language model API")
    if provider == 'gemini':
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
    # Dynamically import the provider module
    provider_module = importlib.import_module(f"models.language_models.{provider}_api")

    # Get the model class from the provider module
    model_class = getattr(provider_module, config.models.language_model.api.provider.capitalize())

    # Initialize and use the language model
    model_name = config.models.language_model.api[f"{provider}"].model_name  
    language_model = model_class(api_key, model_name=model_name)

tts = TTS(model_path=config.models.coqui.model_path, 
          config_path=config.models.coqui.config_path, 
          progress_bar=False,
          gpu=config.models.coqui.gpu
    )
logging.info('Loaded TTS model...')

lipsyncvideo = video_inference.LipSyncVideo(
    config.models.video_retalking.device,
    config.models.video_retalking.base_dir
)
logging.info('Loaded Video model...')

def transcribe(audio_data):
    # process audio data
    try:
        segments, info = whisper_model.transcribe(audio_data)
        seg = []
        for segment in segments:
            seg.append(segment.text)
        text_data = ''.join(seg) # transcribed_text
        flag = True
        if not len(text_data):
            text_data = "No speech detected, Try Speaking again."
            flag = False
        logging.info("Whisper transcription done.")

        return text_data, flag
    except Exception as e:
        error = "Error during Transcription, Please try again."
        logging.warning(error)
        return error, False


def audio_to_video_pipeline(text_data, audio_clip, video_clip, lang, speed, socketio=None):
    # language model response.
    model_resp, flag = language_model.getResponse(text_data)

    if flag:
        logging.info(f"LLM Response: {model_resp}")

        gen_audio_path = os.path.join('temp', 'generated_audio.wav')
        try:
            tts.tts_to_file(text=model_resp, speaker_wav=audio_clip, language=lang,
                            file_path=gen_audio_path, speed=speed, split_sentences=True)
            socketio.emit('processing', 'TTS Processing done!')
        except Exception as e:
            logging.warning(f"[ERROR] TTS: {str(e)}")
            socketio.emit('error', str(e))
            return

        try:
            lipsyncvideo.run(logger, video_clip, gen_audio_path, socketio)
            socketio.emit('done') # indicate processing has finished
        except Exception as e:
            logging.warning(f"[ERROR] VIDGEN: {str(e)}")
            socketio.emit('error', str(e))
        
    else:
        error = 'LLM failed to parse, Please try again.'
        logging.warning(error)
        socketio.emit('error', error)
        
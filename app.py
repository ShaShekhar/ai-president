from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit # type: ignore
import subprocess
import os
import main

app = Flask(__name__, template_folder='web_frontend/templates', static_folder='web_frontend/static')
cors = CORS(app)
socketio = SocketIO(app, websocket_transport_options={'tcp_nodelay': True}, 
                    cors_allowed_origins="*", async_model='eventlet',
                    ping_timeout=60, ping_interval=10
            )
is_processing = False # Boolean to indicate ongoing processing
os.makedirs('temp', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', recording_time=main.config.web_frontend.recording_time)

@socketio.on('client_connected')
def handle_connect():
    global is_processing
    if is_processing:
        emit('processing', 'Processing...')  # Notify the client if processing is already ongoing
    else:
        emit('ready')  # Signal the client that it can initiate processing

@socketio.on('process_video')
def handle_process_video(data):
    global is_processing
    if not is_processing: # Check if processing is already ongoing
        is_processing = True
        dataType = data.get("dataType")
        dataFile = data.get("data")

        if dataType not in ["text", "audio"]:
            emit('error', 'Invalid data type')
            return

        if dataType == "audio":
            # Convert webm to raw audio bytes using FFmpeg
            cmd = [
            "ffmpeg",
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1"
            ]
            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                audio_data, stderr = process.communicate(input=dataFile)
                if process.returncode != 0:
                    raise Exception(f"FFmpeg error: {stderr.decode()}")
            except subprocess.CalledProcessError as e:
                emit('error', 'FFmpeg error, Please try again.')
                return
            except Exception as e:
                emit('error', 'Error converting audio, Please try again.')
                return

            text_data, flag = main.transcribe(audio_data)
            if not flag:
                emit('error', text_data)
                return

        elif dataType == "text":
            text_data = dataFile.decode("utf-8") # Decode text data

        socketio.start_background_task(generate_video_segments, text_data)
        emit('processing_started', 'Processing started...')
    else:
        emit('processing', 'Processing previous request, Please wait..')


def generate_video_segments(text_data):
    global is_processing
    try:
        audio_configs = main.config.models.coqui
        audio_clip = audio_configs.audio_clip
        lang = audio_configs.language
        speed = audio_configs.speed
        video_clip = main.config.models.video_retalking.video_clip

        main.audio_to_video_pipeline(text_data, audio_clip,
            video_clip, lang, speed, socketio=socketio)
        
        is_processing = False
    except Exception as e:
        socketio.emit('error', str(e))


if __name__ == "__main__":
    host = os.environ.get("IP", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    # app.run(host=host, port=port, debug=False)
    socketio.run(app, host=host, port=port, debug=False)

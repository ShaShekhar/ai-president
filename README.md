## AI-President:

Talk to President Biden in real time using the web interface's audio feature. Responses will stream live. To install, clone the repository and follow the instructions.

https://github.com/user-attachments/assets/a80052e9-6d0a-4100-ad2b-0f3fd7261cf3

## Project Structure

- **faster_whisper**: Contains the faster_whisper which is implementation of OpenAI Whisper for audio transcription/translation.
- **language_model**: Using small Language model like Phi3 or any of the LLM API: Gemini, Claude, GPT4 etc.
- **coqui XTTS2**: Text-to-Speech model.
- **video-retalking**: LipSync Video Generation Model.

## Installation:

**Prerequisites:**

- **Python >= 3.9**
- **PyTorch >= 1.8 with CUDA 11.x or 12.x support:**
  Follow the instructions on [official PyTorch website!](https://pytorch.org/)

**Steps:**

```bash
$ git clone https://github.com/shashekhar/ai_president.git
$ cd ai_president
$ pip install -r requirements.txt
$ # pip install --force-reinstall ctranslate2==3.24.0 # for cuda 11.8
```

Download XTTS2 and Video-retalking model weights.

```bash
./download_models.sh
```

**Conda Installation:**

```bash
$ conda create -n ai_president python=3.9
$ conda activate ai_president
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # or 11.8
$ git clone https://github.com/shashekhar/ai_president.git
$ cd aaiela
$ pip install -r requirements.txt
$ cd models
$ python -m tests.test_whisper # test_gemini, test_phi, test_tts, test_vidgen
$ gunicorn -k eventlet -w 1 --timeout 300 --bind unix:president.sock wsgi:application
$ tail -f /tmp/app.log
```

**docker Installation:**

- The official [Dockerfile](Dockerfile) installs with a few simple commands.

```bash
$ git clone https://github.com/shashekhar/ai-president.git
$ cd ai-president
$ docker build -t aipresident_conda --network=host --build-arg CUDA_VERSION=12.1 . # or 11.8
$ docker run --gpus all -it --rm -v weights:/app/weights -p 5000:5000 aipresident_conda
$ conda activate aipresident_conda
$ gunicorn -k eventlet -w 1 --timeout 300 --bind unix:president.sock wsgi:application # start a tmux session
```

API Keys: Create a `.env` file in the root directory of the project. Fill in API keys if intend to use API-based
language models. Use the provided `.env.example` file as a template.

Or to use a small language model like Phi-3, set the `active_model:local` in config file.

To run individual test files:

```bash
$ python -m tests.<test_file_name>
```

Configuration: adjust some settings in the `inference.yaml` config file e.g., device, active_model.
Toggle between using an API-based model or a local LLM by modifying the `active_model` parameter.

- Run the project's main script to load the model and start the web interface.

  `gunicorn -k eventlet -w 1 --timeout 300 --bind unix:president.sock wsgi:application`

## Todo

- [ ] Optimize the video-retalking model.

- [ ] streamline the clip selection process.

- [ ] options to select source and target languages.

- [ ] debug socketio response delay.

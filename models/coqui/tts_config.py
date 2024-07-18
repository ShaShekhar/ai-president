from dataclasses import dataclass, field
from typing import List, Dict
from trainer import TrainerConfig

@dataclass
class BaseTrainingConfig(TrainerConfig):
    """Base config to define the basic 🐸TTS training parameters that are shared
    among all the models. It is based on ```Trainer.TrainingConfig```.

    Args:
        model (str):
            Name of the model that is used in the training.

        num_loader_workers (int):
            Number of workers for training time dataloader.

        num_eval_loader_workers (int):
            Number of workers for evaluation time dataloader.
    """

    model: str = None
    # dataloading
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False

from coqpit import Coqpit, check_argument

@dataclass
class CharactersConfig(Coqpit):
    """Defines arguments for the `BaseCharacters` or `BaseVocabulary` and their subclasses.

    Args:
        characters_class (str):
            Defines the class of the characters used. If None, we pick ```Phonemes``` or ```Graphemes``` based on
            the configuration. Defaults to None.

        vocab_dict (dict):
            Defines the vocabulary dictionary used to encode the characters. Defaults to None.

        pad (str):
            characters in place of empty padding. Defaults to None.

        eos (str):
            characters showing the end of a sentence. Defaults to None.

        bos (str):
            characters showing the beginning of a sentence. Defaults to None.

        blank (str):
            Optional character used between characters by some models for better prosody. Defaults to `_blank`.

        characters (str):
            character set used by the model. Characters not in this list are ignored when converting input text to
            a list of sequence IDs. Defaults to None.

        punctuations (str):
            characters considered as punctuation as parsing the input sentence. Defaults to None.

        phonemes (str):
            characters considered as parsing phonemes. This is only for backwards compat. Use `characters` for new
            models. Defaults to None.

        is_unique (bool):
            remove any duplicate characters in the character lists. It is a bandaid for compatibility with the old
            models trained with character lists with duplicates. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    characters_class: str = None

    # using BaseVocabulary
    vocab_dict: Dict = None

    # using on BaseCharacters
    pad: str = None
    eos: str = None
    bos: str = None
    blank: str = None
    characters: str = None
    punctuations: str = None
    phonemes: str = None
    is_unique: bool = True  # for backwards compatibility of models trained with char sets with duplicates
    is_sorted: bool = True

from dataclasses import asdict, dataclass
from typing import List

from coqpit import Coqpit, check_argument
from trainer import TrainerConfig

@dataclass
class BaseAudioConfig(Coqpit):
    """Base config to definge audio processing parameters. It is used to initialize
    ```TTS.utils.audio.AudioProcessor.```

    Args:
        fft_size (int):
            Number of STFT frequency levels aka.size of the linear spectogram frame. Defaults to 1024.

        win_length (int):
            Each frame of audio is windowed by window of length ```win_length``` and then padded with zeros to match
            ```fft_size```. Defaults to 1024.

        hop_length (int):
            Number of audio samples between adjacent STFT columns. Defaults to 1024.

        frame_shift_ms (int):
            Set ```hop_length``` based on milliseconds and sampling rate.

        frame_length_ms (int):
            Set ```win_length``` based on milliseconds and sampling rate.

        stft_pad_mode (str):
            Padding method used in STFT. 'reflect' or 'center'. Defaults to 'reflect'.

        sample_rate (int):
            Audio sampling rate. Defaults to 22050.

        resample (bool):
            Enable / Disable resampling audio to ```sample_rate```. Defaults to ```False```.

        preemphasis (float):
            Preemphasis coefficient. Defaults to 0.0.

        ref_level_db (int): 20
            Reference Db level to rebase the audio signal and ignore the level below. 20Db is assumed the sound of air.
            Defaults to 20.

        do_sound_norm (bool):
            Enable / Disable sound normalization to reconcile the volume differences among samples. Defaults to False.

        log_func (str):
            Numpy log function used for amplitude to DB conversion. Defaults to 'np.log10'.

        do_trim_silence (bool):
            Enable / Disable trimming silences at the beginning and the end of the audio clip. Defaults to ```True```.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to True.

        do_amp_to_db_mel (bool, optional):
            enable/disable amplitude to dB conversion of mel spectrograms. Defaults to True.

        pitch_fmax (float, optional):
            Maximum frequency of the F0 frames. Defaults to ```640```.

        pitch_fmin (float, optional):
            Minimum frequency of the F0 frames. Defaults to ```1```.

        trim_db (int):
            Silence threshold used for silence trimming. Defaults to 45.

        do_rms_norm (bool, optional):
            enable/disable RMS volume normalization when loading an audio file. Defaults to False.

        db_level (int, optional):
            dB level used for rms normalization. The range is -99 to 0. Defaults to None.

        power (float):
            Exponent used for expanding spectrogra levels before running Griffin Lim. It helps to reduce the
            artifacts in the synthesized voice. Defaults to 1.5.

        griffin_lim_iters (int):
            Number of Griffing Lim iterations. Defaults to 60.

        num_mels (int):
            Number of mel-basis frames that defines the frame lengths of each mel-spectrogram frame. Defaults to 80.

        mel_fmin (float): Min frequency level used for the mel-basis filters. ~50 for male and ~95 for female voices.
            It needs to be adjusted for a dataset. Defaults to 0.

        mel_fmax (float):
            Max frequency level used for the mel-basis filters. It needs to be adjusted for a dataset.

        spec_gain (int):
            Gain applied when converting amplitude to DB. Defaults to 20.

        signal_norm (bool):
            enable/disable signal normalization. Defaults to True.

        min_level_db (int):
            minimum db threshold for the computed melspectrograms. Defaults to -100.

        symmetric_norm (bool):
            enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else
            [0, k], Defaults to True.

        max_norm (float):
            ```k``` defining the normalization range. Defaults to 4.0.

        clip_norm (bool):
            enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.

        stats_path (str):
            Path to the computed stats file. Defaults to None.
    """

    # stft parameters
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    # audio processing parameters
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0
    ref_level_db: int = 20
    do_sound_norm: bool = False
    log_func: str = "np.log10"
    # silence trimming
    do_trim_silence: bool = True
    trim_db: int = 45
    # rms volume normalization
    do_rms_norm: bool = False
    db_level: float = None
    # griffin-lim params
    power: float = 1.5
    griffin_lim_iters: int = 60
    # mel-spec params
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    # f0 params
    pitch_fmax: float = 640.0
    pitch_fmin: float = 1.0
    # normalization params
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument(
            "frame_length_ms",
            c,
            restricted=True,
            min_val=10,
            max_val=1000,
            alternative="win_length",
        )
        check_argument("frame_shift_ms", c, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)

        # normalization parameters
        check_argument("signal_norm", c, restricted=True)
        check_argument("symmetric_norm", c, restricted=True)
        check_argument("max_norm", c, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", c, restricted=True)
        check_argument("mel_fmin", c, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", c, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", c, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", c, restricted=True)
        check_argument("trim_db", c, restricted=True)

@dataclass
class BaseDatasetConfig(Coqpit):
    """Base config for TTS datasets.

    Args:
        formatter (str):
            Formatter name that defines used formatter in ```TTS.tts.datasets.formatter```. Defaults to `""`.

        dataset_name (str):
            Unique name for the dataset. Defaults to `""`.

        path (str):
            Root path to the dataset files. Defaults to `""`.

        meta_file_train (str):
            Name of the dataset meta file. Or a list of speakers to be ignored at training for multi-speaker datasets.
            Defaults to `""`.

        ignored_speakers (List):
            List of speakers IDs that are not used at the training. Default None.

        language (str):
            Language code of the dataset. If defined, it overrides `phoneme_language`. Defaults to `""`.

        phonemizer (str):
            Phonemizer used for that dataset's language. By default it uses `DEF_LANG_TO_PHONEMIZER`. Defaults to `""`.

        meta_file_val (str):
            Name of the dataset meta file that defines the instances used at validation.

        meta_file_attn_mask (str):
            Path to the file that lists the attention mask files used with models that require attention masks to
            train the duration predictor.
    """

    formatter: str = ""
    dataset_name: str = ""
    path: str = ""
    meta_file_train: str = ""
    ignored_speakers: List[str] = None
    language: str = ""
    phonemizer: str = ""
    meta_file_val: str = ""
    meta_file_attn_mask: str = ""

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("formatter", c, restricted=True)
        check_argument("path", c, restricted=True)
        check_argument("meta_file_train", c, restricted=True)
        check_argument("meta_file_val", c, restricted=False)
        check_argument("meta_file_attn_mask", c, restricted=False)

@dataclass
class BaseTTSConfig(BaseTrainingConfig):
    """Shared parameters among all the tts models.

    Args:

        audio (BaseAudioConfig):
            Audio processor config object instance.

        use_phonemes (bool):
            enable / disable phoneme use.

        phonemizer (str):
            Name of the phonemizer to use. If set None, the phonemizer will be selected by `phoneme_language`.
            Defaults to None.

        phoneme_language (str):
            Language code for the phonemizer. You can check the list of supported languages by running
            `python TTS/tts/utils/text/phonemizers/__init__.py`. Defaults to None.

        compute_input_seq_cache (bool):
            enable / disable precomputation of the phoneme sequences. At the expense of some delay at the beginning of
            the training, It allows faster data loader time and precise limitation with `max_seq_len` and
            `min_seq_len`.

        text_cleaner (str):
            Name of the text cleaner used for cleaning and formatting transcripts.

        enable_eos_bos_chars (bool):
            enable / disable the use of eos and bos characters.

        test_senteces_file (str):
            Path to a txt file that has sentences used at test time. The file must have a sentence per line.

        phoneme_cache_path (str):
            Path to the output folder caching the computed phonemes for each sample.

        characters (CharactersConfig):
            Instance of a CharactersConfig class.

        batch_group_size (int):
            Size of the batch groups used for bucketing. By default, the dataloader orders samples by the sequence
            length for a more efficient and stable training. If `batch_group_size > 1` then it performs bucketing to
            prevent using the same batches for each epoch.

        loss_masking (bool):
            enable / disable masking loss values against padded segments of samples in a batch.

        min_text_len (int):
            Minimum length of input text to be used. All shorter samples will be ignored. Defaults to 0.

        max_text_len (int):
            Maximum length of input text to be used. All longer samples will be ignored. Defaults to float("inf").

        min_audio_len (int):
            Minimum length of input audio to be used. All shorter samples will be ignored. Defaults to 0.

        max_audio_len (int):
            Maximum length of input audio to be used. All longer samples will be ignored. The maximum length in the
            dataset defines the VRAM used in the training. Hence, pay attention to this value if you encounter an
            OOM error in training. Defaults to float("inf").

        compute_f0 (int):
            (Not in use yet).

        compute_energy (int):
            (Not in use yet).

        compute_linear_spec (bool):
            If True data loader computes and returns linear spectrograms alongside the other data.

        precompute_num_workers (int):
            Number of workers to precompute features. Defaults to 0.

        use_noise_augment (bool):
            Augment the input audio with random noise.

        start_by_longest (bool):
            If True, the data loader will start loading the longest batch first. It is useful for checking OOM issues.
            Defaults to False.

        shuffle (bool):
            If True, the data loader will shuffle the dataset when there is not sampler defined. Defaults to True.

        drop_last (bool):
            If True, the data loader will drop the last batch if it is not complete. It helps to prevent
            issues that emerge from the partial batch statistics. Defaults to True.

        add_blank (bool):
            Add blank characters between each other two characters. It improves performance for some models at expense
            of slower run-time due to the longer input sequence.

        datasets (List[BaseDatasetConfig]):
            List of datasets used for training. If multiple datasets are provided, they are merged and used together
            for training.

        optimizer (str):
            Optimizer used for the training. Set one from `torch.optim.Optimizer` or `TTS.utils.training`.
            Defaults to ``.

        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`

        lr_scheduler (str):
            Learning rate scheduler for the training. Use one from `torch.optim.Scheduler` schedulers or
            `TTS.utils.training`. Defaults to ``.

        lr_scheduler_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"warmup": 4000}`.

        test_sentences (List[str]):
            List of sentences to be used at testing. Defaults to '[]'

        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).

        use_speaker_weighted_sampler (bool):
            Enable / Disable the batch balancer by speaker. Defaults to ```False```.

        speaker_weighted_sampler_alpha (float):
            Number that control the influence of the speaker sampler weights. Defaults to ```1.0```.

        use_language_weighted_sampler (bool):
            Enable / Disable the batch balancer by language. Defaults to ```False```.

        language_weighted_sampler_alpha (float):
            Number that control the influence of the language sampler weights. Defaults to ```1.0```.

        use_length_weighted_sampler (bool):
            Enable / Disable the batch balancer by audio length. If enabled the dataset will be divided
            into 10 buckets considering the min and max audio of the dataset. The sampler weights will be
            computed forcing to have the same quantity of data for each bucket in each training batch. Defaults to ```False```.

        length_weighted_sampler_alpha (float):
            Number that control the influence of the length sampler weights. Defaults to ```1.0```.
    """

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # phoneme settings
    use_phonemes: bool = False
    phonemizer: str = None
    phoneme_language: str = None
    compute_input_seq_cache: bool = False
    text_cleaner: str = None
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    phoneme_cache_path: str = None
    # vocabulary parameters
    characters: CharactersConfig = None
    add_blank: bool = False
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 1
    max_text_len: int = float("inf")
    compute_f0: bool = False
    compute_energy: bool = False
    compute_linear_spec: bool = False
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    shuffle: bool = False
    drop_last: bool = False
    # dataset
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    # optimizer
    optimizer: str = "radam"
    optimizer_params: dict = None
    # scheduler
    lr_scheduler: str = None
    lr_scheduler_params: dict = field(default_factory=lambda: {})
    # testing
    test_sentences: List[str] = field(default_factory=lambda: [])
    # evaluation
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    # weighted samplers
    use_speaker_weighted_sampler: bool = False
    speaker_weighted_sampler_alpha: float = 1.0
    use_language_weighted_sampler: bool = False
    language_weighted_sampler_alpha: float = 1.0
    use_length_weighted_sampler: bool = False
    length_weighted_sampler_alpha: float = 1.0

@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000

@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400


@dataclass
class XttsConfig(BaseTTSConfig):
    """Defines parameters for XTTS TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (XttsArgs):
            Model architecture arguments. Defaults to `XttsArgs()`.

        audio (XttsAudioConfig):
            Audio processing configuration. Defaults to `XttsAudioConfig()`.

        model_dir (str):
            Path to the folder that has all the XTTS models. Defaults to None.

        temperature (float):
            Temperature for the autoregressive model inference. Larger values makes predictions more creative sacrificing stability. Defaults to `0.2`.

        length_penalty (float):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length,
            which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
            length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. Defaults to `2.0`.

        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            Defaults to `0.8`.

        num_gpt_outputs (int):
            Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
            As XTTS is a probabilistic model, more samples means a higher probability of creating something "great".
            Defaults to `16`.

        gpt_cond_len (int):
            Secs audio to be used as conditioning for the autoregressive model. Defaults to `12`.

        gpt_cond_chunk_len (int):
            Audio chunk size in secs. Audio is split into chunks and latents are extracted for each chunk. Then the
            latents are averaged. Chunking improves the stability. It must be <= gpt_cond_len.
            If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to `4`.

        max_ref_len (int):
            Maximum number of seconds of audio to be used as conditioning for the decoder. Defaults to `10`.

        sound_norm_refs (bool):
            Whether to normalize the conditioning audio. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> config = XttsConfig()
    """

    model: str = "xtts"
    # model specific params
    model_args: XttsArgs = field(default_factory=XttsArgs)
    audio: XttsAudioConfig = field(default_factory=XttsAudioConfig)
    model_dir: str = None
    languages: List[str] = field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
            "hi",
        ]
    )

    # inference params
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False
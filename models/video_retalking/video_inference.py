import numpy as np
import cv2, os, sys, subprocess, platform, base64
from subprocess import check_output
import torch
import time
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, 'models/video_retalking/third_part')
sys.path.insert(0, 'models/video_retalking/third_part/GPEN')
sys.path.insert(0, 'models/video_retalking/third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings
warnings.filterwarnings("ignore")

class Arguments: 
    def __init__(self, base_dir): 
        self.DNet_path = base_dir + 'DNet.pt'
        self.LNet_path = base_dir + 'LNet.pth'
        self.ENet_path = base_dir + 'ENet.pth'
        self.face3d_net_path = base_dir + 'face3d_pretrain_epoch_20.pth'
        # self.face = 'test.mp4/jpg' # Filepath of video/image that contains faces to use
        # self.audio = 'audio.wav' # Filepath of video/audio file to use as raw audio source
        self.exp_img = 'neutral' # 'Expression template. neutral, smile or image path
        self.outfile = 'results/output_video.mp4' # Video path to save result
        self.fps = 25.0 # Can be specified only if input is a static image (default: 25)
        self.pads = [0, 20, 0, 0] # Padding (top, bottom, left, right). Please adjust to include chin at least
        self.face_det_batch_size = 4 # Batch size for face detection
        self.LNet_batch_size = 32 # Batch size for LNet
        self.img_size = 384
        # Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg.
        # Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width
        self.crop = [0, -1, 0, -1]
        # Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.
        # Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
        self.box = [-1, -1, -1, -1]
        # Prevent smoothing face detections over a short temporal window
        self.nosmooth = False
        self.static = False
        self.up_face = 'original'
        self.one_shot = False
        self.without_rl1 = False # Do not use the relative l1
        self.tmp_dir = 'temp' # Folder to save tmp results
        self.re_preprocess = False

class LipSyncVideo:
    def __init__(self, device, base_dir):
        # base_dir = 'weights/checkpoints/'
        self.device = device
        self.args = Arguments(base_dir)

        self.args.cd = 5 # chunk_duration in sec
        os.makedirs(self.args.tmp_dir, exist_ok=True)

        self.enhancer = FaceEnhancement(base_dir=base_dir, size=512, model='GPEN-BFR-512', use_sr=False, \
                        sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
        self.restorer = GFPGANer(model_path=base_dir+'GFPGANv1.3.pth', upscale=1, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)

        self.kp_extractor = KeypointExtractor()

        # face detection & cropping, cropping the first frame as the style of FFHQ
        self.croper = Croper(base_dir+'shape_predictor_68_face_landmarks.dat')
        self.net_recon = load_face3d_net(self.args.face3d_net_path, device)
        self.lm3d_std = load_lm3d(base_dir+'BFM')

        self.expression_mouth = torch.tensor(loadmat(base_dir+'expression.mat')['expression_mouth'])[0]
        self.expression_center = torch.tensor(loadmat(base_dir+'expression.mat')['expression_center'])[0]
        # load DNet, model(LNet and ENet)
        self.D_Net, self.model = load_model(self.args, device)

    def run(self, logger, face_path, audio_path, socketio):
        self.args.face = face_path
        self.args.audio = audio_path

        base_name = self.args.face.split('/')[-1]
        if os.path.isfile(self.args.face) and self.args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.args.static = True
        if not os.path.isfile(self.args.face):
            raise ValueError('--face argument must be a valid path to video/image file')
        elif self.args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(self.args.face)]
            fps = self.args.fps
        else:
            video_stream = cv2.VideoCapture(self.args.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                y1, y2, x1, x2 = self.args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)

        logger.info("[Step 0] Number of frames available for inference: "+str(len(full_frames)))
        
        full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
        full_frames_RGB, crop, quad = self.croper.crop(full_frames_RGB, xsize=512)

        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, full_frames[0].shape[0]), clx+lx, min(clx+rx, full_frames[0].shape[1])
        # original_size = (ox2 - ox1, oy2 - oy1)
        frames_pil = [Image.fromarray(cv2.resize(frame,(256,256))) for frame in full_frames_RGB]

        # get the landmark according to the detected face.
        if not os.path.isfile('temp/'+base_name+'_landmarks.txt') or self.args.re_preprocess:
            logger.info('[Step 1] Landmarks Extraction in Video.')
            
            lm = self.kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
        else:
            logger.info('[Step 1] Using saved landmarks.')
            lm = np.loadtxt('temp/'+base_name+'_landmarks.txt').astype(np.float32)
            lm = lm.reshape([len(full_frames), -1, 2])
        
        if not os.path.isfile('temp/'+base_name+'_coeffs.npy') or self.args.exp_img is not None or self.args.re_preprocess:

            video_coeffs = []
            for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
                frame = frames_pil[idx]
                W, H = frame.size
                lm_idx = lm[idx].reshape([-1, 2])
                if np.mean(lm_idx) == -1:
                    lm_idx = (self.lm3d_std[:, :2]+1) / 2.
                    lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
                else:
                    lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

                trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, self.lm3d_std)
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0) 
                with torch.no_grad():
                    coeffs = split_coeff(self.net_recon(im_idx_tensor))

                pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
                pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                            pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
                video_coeffs.append(pred_coeff)
            semantic_npy = np.array(video_coeffs)[:,0]
            np.save('temp/'+base_name+'_coeffs.npy', semantic_npy)
        else:
            logger.info('[Step 2] Using saved coeffs.')
            semantic_npy = np.load('temp/'+base_name+'_coeffs.npy').astype(np.float32)

        # generate the 3dmm coeff from a single image
        if self.args.exp_img is not None and ('.png' in self.args.exp_img or '.jpg' in self.args.exp_img):
            logger.info('extract the exp from',self.args.exp_img)
            exp_pil = Image.open(self.args.exp_img).convert('RGB')
            # lm3d_std = load_lm3d('third_part/face3d/BFM')
            
            W, H = exp_pil.size
            # kp_extractor = KeypointExtractor()
            lm_exp = self.kp_extractor.extract_keypoint([exp_pil], 'temp/'+base_name+'_temp.txt')[0]
            if np.mean(lm_exp) == -1:
                lm_exp = (self.lm3d_std[:, :2] + 1) / 2.
                lm_exp = np.concatenate(
                    [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
            else:
                lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

            trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, self.lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_exp_tensor = torch.tensor(np.array(im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
            with torch.no_grad():
                expression = split_coeff(self.net_recon(im_exp_tensor))['exp'][0]
            # del net_recon
        elif self.args.exp_img == 'smile':
            expression = self.expression_mouth
        else:
            logger.info('using expression center')
            expression = self.expression_center

        if not os.path.isfile('temp/'+base_name+'_stablized.npy') or self.args.re_preprocess:
            imgs = []
            for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):
                if self.args.one_shot:
                    source_img = trans_image(frames_pil[0]).unsqueeze(0).to(self.device)
                    semantic_source_numpy = semantic_npy[0:1]
                else:
                    source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(self.device)
                    semantic_source_numpy = semantic_npy[idx:idx+1]
                ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
                coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(self.device)
            
                # hacking the new expression
                coeff[:, :64, :] = expression[None, :64, None].to(self.device) 
                with torch.no_grad():
                    output = self.D_Net(source_img, coeff)
                img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
                imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR)) 
            np.save('temp/'+base_name+'_stablized.npy',imgs)
            # del D_Net
        else:
            logger.info('[Step 3] Using saved stabilized video.')
            imgs = np.load('temp/'+base_name+'_stablized.npy')
        torch.cuda.empty_cache()

        if not self.args.audio.endswith('.wav'):
            command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(self.args.audio, 'temp/{}/temp.wav'.format(self.args.tmp_dir))
            subprocess.call(command, shell=True)
            self.args.audio = f'{self.args.tmp_dir}/temp_audio.wav'
        
        wav = audio.load_wav(self.args.audio, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        logger.info("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
        imgs = imgs[:len(mel_chunks)]
        full_frames = full_frames[:len(mel_chunks)]
        lm = lm[:len(mel_chunks)]
        
        imgs_enhanced = []
        for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
            img = imgs[idx]
            pred, _, _ = self.enhancer.process(img, img, face_enhance=True, possion_blending=False)
            imgs_enhanced.append(pred)
        gen = self.datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1,oy2,ox1,ox2))

        frame_h, frame_w = full_frames[0].shape[:-1]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        chunk_count = 0
        video_chunk_path = f'{self.args.tmp_dir}/video_chunk_{chunk_count}.mp4'
        out_chunk = cv2.VideoWriter(video_chunk_path, fourcc, fps, (frame_w, frame_h))
        audio_duration = check_output(['ffprobe', '-i', self.args.audio, '-show_entries', 
                            'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")])
        audio_duration = float(audio_duration)
        logger.info(f'generated audio duration: {audio_duration}')
        chunk_duration = 5 # 5sec clip
        frame_counter = 0
        
        if self.args.up_face != 'original':
            instance = GANimationModel()
            instance.initialize()
            instance.setup()

        total = int(np.ceil(float(len(mel_chunks)) / self.args.LNet_batch_size))
        for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=total)):
            socketio.emit('progress', int(i*100/total))
            time.sleep(1)

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(self.device)/255. # BGR -> RGB
            
            with torch.no_grad():
                incomplete, reference = torch.split(img_batch, 3, dim=1) 
                pred, low_res = self.model(mel_batch, img_batch, reference)
                pred = torch.clamp(pred, 0, 1)

                if self.args.up_face in ['sad', 'angry', 'surprise']:
                    tar_aus = exp_aus_dict[self.args.up_face]
                else:
                    pass
                
                if self.args.up_face == 'original':
                    cur_gen_faces = img_original
                else:
                    test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'), 
                                'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                    instance.feed_batch(test_batch)
                    instance.forward()
                    cur_gen_faces = torch.nn.functional.interpolate(instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')
                    
                if self.args.without_rl1 is not False:
                    incomplete, reference = torch.split(img_batch, 3, dim=1)
                    mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete)) 
                    pred = pred * mask + cur_gen_faces * (1 - mask)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            torch.cuda.empty_cache()
            
            for p, f, xf, c in zip(pred, frames, f_frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                
                ff = xf.copy()
                ff[y1:y2, x1:x2] = p
                
                # month region enhancement by GFPGAN
                cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                    ff, has_aligned=False, only_center_face=True, paste_back=True)
                    # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
                mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                mouse_mask = np.zeros_like(restored_img)
                tmp_mask = self.enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
                mouse_mask[y1:y2, x1:x2]= cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

                height, width = ff.shape[:2]
                restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, ff, np.float32(mouse_mask))]
                img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
                pp = np.uint8(cv2.resize(np.clip(img, 0 ,255), (width, height)))

                pp, orig_faces, enhanced_faces = self.enhancer.process(pp, xf, bbox=c, face_enhance=False, possion_blending=True)
                out_chunk.write(pp)
                frame_counter += 1

                if frame_counter >= int(fps*chunk_duration):
                    frame_counter = 0
                    out_chunk.release()
                    self.generate_and_emit_chunk(chunk_count, video_chunk_path, self.args.cd, socketio)
                    chunk_count += 1
                    video_chunk_path = f'{self.args.tmp_dir}/video_chunk_{chunk_count}.mp4'
                    # initialize the out_chunk
                    out_chunk = cv2.VideoWriter(video_chunk_path, fourcc, fps, (frame_w, frame_h))
        
        # Send the chunk after the loop (ensures all frames are written)
        if out_chunk is not None:  # Check if there's a chunk to send
            out_chunk.release()
            cd = audio_duration - chunk_count*chunk_duration # e.g 28 - 5*5 = 3
            self.generate_and_emit_chunk(chunk_count, video_chunk_path, cd, socketio)

    # frames:256x256, full_frames: original size
    def datagen(self, frames, mels, full_frames, frames_pil, cox):
        img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
        base_name = self.args.face.split('/')[-1]
        refs = []
        image_size = 256

        # original frames
        kp_extractor = KeypointExtractor()
        fr_pil = [Image.fromarray(frame) for frame in frames]
        lms = kp_extractor.extract_keypoint(fr_pil, 'temp/'+base_name+'x12_landmarks.txt')
        frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
        crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
        inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
        del kp_extractor.detector

        oy1,oy2,ox1,ox2 = cox
        face_det_results = face_detect(full_frames, self.args, jaw_correction=True)

        for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
            imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
                cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

            ff = full_frame.copy()
            ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
            oface, coords = face_det
            y1, y2, x1, x2 = coords
            refs.append(ff[y1: y2, x1:x2])

        for i, m in enumerate(mels):
            idx = 0 if self.args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face = refs[idx]
            oface, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.args.img_size, self.args.img_size))
            oface = cv2.resize(oface, (self.args.img_size, self.args.img_size))

            img_batch.append(oface)
            ref_batch.append(face) 
            mel_batch.append(m)
            coords_batch.append(coords)
            frame_batch.append(frame_to_save)
            full_frame_batch.append(full_frames[idx].copy())

            if len(img_batch) >= self.args.LNet_batch_size:
                img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
                img_masked = img_batch.copy()
                img_original = img_batch.copy()
                img_masked[:, self.args.img_size//2:] = 0
                img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
                img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, self.args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
    
    def generate_and_emit_chunk(self, i, video_path, chunk_duration, socketio, logger):
        start_time = i * self.args.cd
        logger.info(f'chunk_no: {i}')
        logger.info(f'audio_path: {self.args.audio}')
        logger.info(f'video_path: {video_path}')
        logger.info(f'start_time: {start_time}')
        logger.info(f'chunk duration: {chunk_duration}')
        # Extract audio chunk using FFmpeg
        audio_chunk_path = f'{self.args.tmp_dir}/audio_chunk_{i}.wav'
        command = f'ffmpeg -loglevel error -y -ss {start_time} -t {chunk_duration} -i {self.args.audio} {audio_chunk_path}'
        subprocess.call(command, shell=True)

        final_clip_path = f'{self.args.tmp_dir}/result_{i}.mp4'
        command = f'/usr/bin/ffmpeg -loglevel error -y -i {video_path} -i {audio_chunk_path} -c:v libx264 -c:a aac -movflags frag_keyframe+empty_moov+default_base_moof {final_clip_path}'
        subprocess.call(command, shell=True)

        with open(final_clip_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        socketio.emit('video_segment', {'data': video_base64, 'pts': start_time})
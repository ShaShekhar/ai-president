cd weights
mkdir checkpoints
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth -O ./checkpoints/30_net_gen.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip -O ./checkpoints/BFM.zip
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt -O ./checkpoints/DNet.pt
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth -O ./checkpoints/ENet.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat -O ./checkpoints/expression.mat
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth -O ./checkpoints/face3d_pretrain_epoch_20.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth -O ./checkpoints/GFPGANv1.3.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth -O ./checkpoints/GPEN-BFR-512.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth -O ./checkpoints/LNet.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth -O ./checkpoints/ParseNet-latest.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth -O ./checkpoints/RetinaFace-R50.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat -O ./checkpoints/shape_predictor_68_face_landmarks.dat
unzip -d ./checkpoints/BFM ./checkpoints/BFM.zip

mkdir coqui
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json -O ./coqui/config.json
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth -O ./coqui/model.pth
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json -O ./coqui/vocab.json
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/speakers_xtts.pth -O ./coqui/speakers_xtts.pth
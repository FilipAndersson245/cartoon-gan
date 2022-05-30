# Cartoon-GAN

![Original_0](https://thumbs.gfycat.com/GlaringQuestionableKoala-size_restricted.gif)

**Paper:** https://arxiv.org/abs/2005.07702

## Description
This project takes on the problem of transferring
the style of cartoon images to real-life photographic images by
implementing previous work done by CartoonGAN. We trained
a Generative Adversial Network(GAN) on over 60 000 images
from works by Hayao Miyazaki at Studio Ghibli.  

To the people asking for the dataset, im sorry but as the material is copyright protected i cannot share the dataset.

## Dependencies

1.  Install Anaconda from https://www.anaconda.com/

2.  Install pytorch at: https://pytorch.org/get-started/locally/

3.  Install dependencies:

    ```
    python -m pip install tqdm pillow numpy matplotlib opencv-python
    ```

4. For predicting videos you will also need ffmpeg

## Weights
Weights for the presented models can be found [here](https://drive.google.com/drive/folders/1d_GsZncTGmMdYht0oUWG9pqvV4UqF_kM?usp=sharing)


## Training

All training code can be found in `experiment.ipynb`

## Predict

Predict by running `predict.py`.

Example:

```
python predict.py -i C:/folder/input_image.png -o ./output_folder/output_image.png
```

Predictions can be made on images, videos or a folder of images/videos.

## Demonstration

| Image # | Original | CartoonGAN | GANILLA | Our implementation |
|:-------:|----------|------------|---------|--------------------|
|1| ![Original_1](https://i.imgur.com/YDvlmgB.png) | ![CartoonGAN_1](https://i.imgur.com/LRvgFzi.jpg) | ![GANILLA_1](https://i.imgur.com/wgVvuU2.png) | ![Ours_1](https://i.imgur.com/JVZi2kF.png) |
|2| ![Original_2](https://i.imgur.com/eDys3vZ.png) | ![CartoonGAN_2](https://i.imgur.com/0S5Od97.jpg) | ![GANILLA_2](https://i.imgur.com/ko24O16.png) | ![Ours_2](https://i.imgur.com/PeJDEym.png) |
|3| ![Original_3](https://i.imgur.com/501pwz9.png) | ![CartoonGAN_3](https://i.imgur.com/7T4Vqlq.jpg) | ![GANILLA_3](https://i.imgur.com/3RomURJ.png) | ![Ours_3](https://i.imgur.com/ZPbxRQZ.png) |
|4| ![Original_4](https://i.imgur.com/R1h7MuC.png) | ![CartoonGAN_4](https://i.imgur.com/bcBz5sm.jpg) | ![GANILLA_4](https://i.imgur.com/5cTjhlc.png) | ![Ours_4](https://i.imgur.com/3Lc7koP.png) |
|5| ![Original_5](https://i.imgur.com/7tGZeQy.png) | ![CartoonGAN_5](https://i.imgur.com/IDZ4kh5.jpg) | ![GANILLA_5](https://i.imgur.com/jsxPDVp.png) | ![Ours_5](https://i.imgur.com/jpZAmc3.png) |
|6| ![Original_6](https://i.imgur.com/ZgBHxWm.png) | ![CartoonGAN_6](https://i.imgur.com/sb9VFf8.jpg) | ![GANILLA_6](https://i.imgur.com/Ud7xNmf.png) | ![Ours_6](https://i.imgur.com/DbFPNyq.png) |
|7| ![Original_7](https://i.imgur.com/7j3ysv0.png) | ![CartoonGAN_7](https://i.imgur.com/4g9VgjJ.jpg) | ![GANILLA_7](https://i.imgur.com/dAuJtfd.png) | ![Ours_7](https://i.imgur.com/wSFvpqm.png) |
|8| ![Original_8](https://i.imgur.com/A3nIuQd.png) | ![CartoonGAN_8](https://i.imgur.com/pzLGkR0.jpg) | ![GANILLA_8](https://i.imgur.com/SF0o9Ta.png) | ![Ours_8](https://i.imgur.com/Eaqmu7g.png) |
|9| ![Original_9](https://i.imgur.com/kad7Q9k.png) | ![CartoonGAN_9](https://i.imgur.com/twlJb0R.jpg) | ![GANILLA_9](https://i.imgur.com/MSLtpZv.png) | ![Ours_9](https://i.imgur.com/5haiEKj.png) |
|10| ![Original_10](https://i.imgur.com/3D5YFPY.png) | ![CartoonGAN_10](https://i.imgur.com/7lgypbC.jpg) | ![GANILLA_10](https://i.imgur.com/aX2clAl.png) | ![Ours_10](https://i.imgur.com/iOvedtX.png) |
|11| ![Original_11](https://i.imgur.com/PjqWZJo.png) | ![CartoonGAN_11](https://i.imgur.com/OtS3DbO.jpg) | ![GANILLA_11](https://i.imgur.com/GOSRyY8.png) | ![Ours_11](https://i.imgur.com/QVkCbph.png) |
|12| ![Original_12](https://i.imgur.com/VomTHCt.png) | ![CartoonGAN_12](https://i.imgur.com/0M8z4tY.png) | ![GANILLA_12](https://i.imgur.com/uiy3JEV.png) | ![Ours_12](https://i.imgur.com/Bcv2SO0.png) |
|13| ![Original_13](https://i.imgur.com/xi0B6MT.png) | ![CartoonGAN_13](https://i.imgur.com/WEVH5eE.jpg) | ![GANILLA_13](https://i.imgur.com/MWRE5Rk.png) | ![Ours_13](https://i.imgur.com/7XCvo1f.png) |
|14| ![Original_14](https://i.imgur.com/WxzUekh.png) | ![CartoonGAN_14](https://i.imgur.com/XndBgqY.jpg) | ![GANILLA_14](https://i.imgur.com/X2mOAme.png) | ![Ours_14](https://i.imgur.com/wMmxyaK.png) |
|15| ![Original_15](https://i.imgur.com/xXrNOMo.png) | ![CartoonGAN_15](https://i.imgur.com/wvh1p6A.jpg) | ![GANILLA_15](https://i.imgur.com/DDx2JIU.png) | ![Ours_15](https://i.imgur.com/yUh0l4y.png) |
|16| ![Original_16](https://i.imgur.com/oFzXQHS.png) | ![CartoonGAN_16](https://i.imgur.com/1RUeXdU.jpg) | ![GANILLA_16](https://i.imgur.com/7nzTOhr.png) | ![Ours_16](https://i.imgur.com/OOcEIw8.png) |
|17| ![Original_17](https://i.imgur.com/W8TR51a.png) | ![CartoonGAN_17](https://i.imgur.com/GNBiZeg.jpg) | ![GANILLA_17](https://i.imgur.com/BCs0v1f.png) | ![Ours_17](https://i.imgur.com/QbdnymV.png) |
|18| ![Original_18](https://i.imgur.com/3Bmytxv.png) | ![CartoonGAN_18](https://i.imgur.com/xDDbMvN.jpg) | ![GANILLA_18](https://i.imgur.com/uvqW1qu.png) | ![Ours_18](https://i.imgur.com/T2MZ7sw.png) |
|19| ![Original_19](https://i.imgur.com/mamrZXA.png) | ![CartoonGAN_19](https://i.imgur.com/klK2o5w.jpg) | ![GANILLA_19](https://i.imgur.com/y3cNt1p.png) | ![Ours_19](https://i.imgur.com/0KAU6Cn.png) |
|20| ![Original_20](https://i.imgur.com/KQNKysf.png) | ![CartoonGAN_20](https://i.imgur.com/9d2jhzd.jpg) | ![GANILLA_20](https://i.imgur.com/eER4szz.png) | ![Ours_20](https://i.imgur.com/8jkKL88.png) |

## Citation

```
@misc{andersson2020generative,
      title={Generative Adversarial Networks for photo to Hayao Miyazaki style cartoons}, 
      author={Filip Andersson and Simon Arvidsson},
      year={2020},
      eprint={2005.07702},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
```

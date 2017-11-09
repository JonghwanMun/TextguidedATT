## Text-guided Attention Model for Image Captioning

Created by [Jonghwan Mun](http://cvlab.postech.ac.kr/~jonghwan/), [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at [POSTECH cvlab](http://cvlab.postech.ac.kr/lab/). <br />
If you want to know details of our paper, please refer to [arXiv preprint](https://arxiv.org/abs/1612.03557).
Also, if you use this code in a publication, please cite our paper using following bibtex.

   @inproceedings{mun2017textguided,
      title={Text-guided Attention Model for Image Captioning},
      author={Mun, Jonghwan and Cho, Minsu and Han, Bohyung},
      booktitle={AAAI},
      year={2017}
   }

### Dependencies (This project is tested on linux 14.04 64bit with gpu Titan)
#### Dependencies for torch
  0. torch ['https://github.com/torch/distro']
  0. cutorch (luarocks install cutorch)
  0. cunn (luarocks install cunn)
  0. cudnn ['https://github.com/soumith/cudnn.torch']
  0. display ['https://github.com/szym/display']
  0. cv ['https://github.com/VisionLabs/torch-opencv']
  0. hdf5 (luarocks install hdf5)
  0. image (luarocks install image)
  0. loadcaffe ['https://github.com/szagoruyko/loadcaffe']

#### Dependencies for python (we test on python 2.7.11 with anaconda 4.0)
  0. json
  0. h5py
  0. cPickle
  0. numpy
  <br /> Maybe all dependencies for python are installed if you use anaconda.

### Download pre-trained model

  ```
  bash get_pretrained_model.sh
  ```

### Running (data construction, training, testing)

  ```
  bash running_script.sh
  ```

### Licence

This software is being made available for research purpose only.
Check LICENSE file for details.

### Acknowledgements

This work is funded by the Samsung Electronics Co., (DMC R&D center). <br />
Also, thanks to Andrej Karpathy since this work is implemented based on his code (https://github.com/karpathy/neuraltalk2)


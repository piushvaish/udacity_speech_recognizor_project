[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/select_kernel.png "select speech_recognition kernel"

## Overview

This repository contains instrutions to set up a local environment, details to get dataset and code to build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline. It accepts raw audio and return the predicted transcription of the spoken language. 

//![ASR Pipeline][image1]

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate your models. The algorithm first convert any raw audio to feature representations that are commonly used for ASR. Then we move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, we engage in our own investigations by creating and testing our own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations. 

### Files Details

- The `vui_notebook.ipynb` file with fully functional code, all code cells executed and displaying output.
- The `sample_models.py` file with all model architectures that were trained in the project Jupyter notebook.
* __These are the 15 model architectures__ : 
	* __simple_rnn_model__
	* __rnn_model__
	* __cnn_rnn_model__
	* __deep_rnn_model__
	* __bidirectional_rnn_model__
	* __dense_cnn_model__
	* __LSTM_model__
	* __final_model__: My own implementation
	* __eyben__ : Implementation of Eyben, Florian, et al. "From speech to letters-using a novel neural
        network architecture for grapheme based asr." Automatic Speech
        Recognition & Understanding, 2009. ASRU 2009. IEEE Workshop on. IEEE,
        2009.[link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.330.9782&rep=rep1&type=pdf)
	* __maas__ : Implementation of Maas, Andrew L., et al. "Lexicon-Free Conversational Speech
        Recognition with Neural Networks." HLT-NAACL. 2015.[link](http://www.aclweb.org/anthology/N15-1038)
	* __graves__ : Implementation of Graves, Alex, et al. "Connectionist temporal classification:
        labelling unsegmented sequence data with recurrent neural networks."
        Proceedings of the 23rd international conference on Machine learning.
        ACM, 2006.[link](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
	* __ds1_dropout__ : Implementation of Hannun, A., et al. Deep Speech: Scaling up end-to-end speech recognition.2014.[link](https://arxiv.org/abs/1412.5567)
	* __ds1__ : Implementation of Hannun, A., et al. Deep Speech: Scaling up end-to-end speech recognition.2014. without dropout [link](https://arxiv.org/abs/1412.5567)
	* __ds2_gru_model__ : Implementation of Deep Speech 2: End-to-End Speech Recognition in English and Mandarin
        Amodei D Anubhai R Battenberg E Case C Casper J et. al.
        2015.[link](https://arxiv.org/abs/1512.02595)
	* __baidu_ds__ : Implentation of Baidu's Deep Speech [link] (https://github.com/baidu-research/ba-dls-deepspeech)

- The `results/` folder containing all HDF5 and pickle files corresponding to trained models.
- The `train_utils.py` file defines the functions for training a neural network
- The `languageModel.py` file is for developing a language model
- The `create_desc_json.py` file create JSON-Line description files that can be used to
train deep-speech models through this library.
- The `data_generator.py` file defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
- The `utils.py` file defines various functions for processing the data

## Data

The instructions below describe getting a small and clean dataset from [LibriSpeech dataset](http://www.openslr.org/12/). The dataset is 337M size from LibriSpeech. It is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.

The site contains bigger datasets if you require.



## Reference

This was part of Udacity NLP Nanodegree. Udacity borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms. 


### Environment Setup

You should run this project with GPU acceleration for best performance.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.optum.com/sdsml/Speech_Recognition
cd Speech_Recognition
```

2. Create (and activate) a new environment with Python 3.6 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name speech_recognition python=3.5 numpy
	source activate speech_recognition
	```
	- __Windows__: 
	```
	conda create --name speech_recognition python=3.5 numpy scipy
	activate speech_recognition
	```

3. Install TensorFlow.
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step and only need to install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu==1.1.0
	```
	- Option 2: __To install TensorFlow with CPU support only__,
	```
	pip install tensorflow==1.1.0
	```

4. Install a few pip packages.
```
pip install -r requirements.txt
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```
	- __NOTE:__ a Keras/Windows bug may give this error after the first epoch of training model 0: `‘rawunicodeescape’ codec can’t decode bytes in position 54-55: truncated \uXXXX `. 
To fix it: 
		- Find the file `keras/utils/generic_utils.py` that you are using for the capstone project. It should be in your environment under `Lib/site-packages` . This may vary, but if using miniconda, for example, it might be located at `C:/Users/username/Miniconda3/envs speech_recognition/Lib/site-packages/keras/utils`.
		- Copy `generic_utils.py` to `OLDgeneric_utils.py` just in case you need to restore it.
		- Open the `generic_utils.py` file and change this code line:</br>`marshal.dumps(func.code).decode(‘raw_unicode_escape’)`</br>to this code line:</br>`marshal.dumps(func.code).replace(b’\’,b’/’).decode(‘raw_unicode_escape’)`

6. Obtain the `libav` package.
	- __Linux__: `sudo apt-get install libav-tools`
	- __Mac__: `brew install libav`
	- __Windows__: Browse to the [Libav website](https://libav.org/download/)
		- Scroll down to "Windows Nightly and Release Builds" and click on the appropriate link for your system (32-bit or 64-bit).
		- Click `nightly-gpl`.
		- Download most recent archive file.
		- Extract the file.  Move the `usr` directory to your C: drive.
		- Go back to your terminal window from above.
	```
	You already have this installed for thebrain
	rename C:\usr avconv
    set PATH=C:\avconv\bin;%PATH%
	```

7. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	- __Linux__ or __Mac__: 
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	./flac_to_wav.sh
	```
	- __Windows__: Download two files ([file 1](http://www.openslr.org/resources/12/dev-clean.tar.gz) and [file 2](http://www.openslr.org/resources/12/test-clean.tar.gz)) via browser and save in the  speech_recognition-Capstone` directory.  Extract them with an application that is compatible with `tar` and `gz` such as [7-zip](http://www.7-zip.org/) or [WinZip](http://www.winzip.com/). Convert the files from your terminal window.
	```
	move flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	powershell ./flac_to_wav.sh
	```

8. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

9. Create an IPython kernel
```
pip install ipykernel
```

10. Deactivate the environment
```
Linux or Mac: source deactivate speech_recognition 
Windows: deactivate speech_recognition

```

### Running the jupyter notebook
- __Linux__ or __Mac__: source activate speech_recognition
- __Windows__: activate speech_recognition
- jupyter notebook --ip="*" --port=`port_number` --no-browser
- Enter the following in your browser : http://`server:port_number`/tree/

## Contact 

If you require any help or have a suggestion, please contact [Piush Vaish](piushvaish@gmail.com)



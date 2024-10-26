I've installed Python 3.10.11 to use with this project. Since I have other Python versions installed and also want to keep this project's packages
seperate from other projects I've decided to use Python's venv (virtual environment). This is much easier to do with anaconda but in that case
pip and conda packages get mixed up.

Before beginning, I also need to install CUDA for the project to use my GPU (as KoljaB details in his README).
I've decided to use CUDA toolkit 12.4 since that is the latest version supported by pytorch 2.5.0 (which is the latest version of pytorch).
To do that download CUDA toolkit 12.4 from: https://developer.nvidia.com/cuda-12-4-0-download-archive
I will also need cuDNN from:                https://developer.nvidia.com/cudnn-downloads

To create a virtual env for a specific python version (without using conda) follow the steps below.
Use CMD and not PS (there may be differences for the commands between the two, I tested this on CMD).


----
ENVIRONMENT:

To begin with, navigate to the root folder where you want to create the env:
`cd "C:\Users\Bill\PurePython_envs"`

To create env based on Python 3.10 (you need to install it first):
`"C:\Program Files\Python310\python.exe" -m venv RealtimeSTT-3.10.11-12.4-pip`

To activate the env:
Go to "C:\Users\Bill\PurePython_envs\RealtimeSTT-3.10.11-12.4-pip\Scripts", open a CMD window and enter `activate.bat`
Or just enter in CMD `Scripts\activate` (you need to be in the env folder, in this case "C:\Users\Bill\PurePython_envs\RealtimeSTT-3.10.11-12.4-pip")
# Note: You can't just run the activate.bat file from the File Explorer directly because:
# When you double-click the activate.bat file from the File Explorer, it briefly opens a Command Prompt window, then immediately closes. This happens
# because .bat scripts are designed to run inside an active command line session. When you double-click from the Explorer, Windows opens a new
# Command Prompt, runs the script, and then closes it right after, without keeping the session open.

To run commands (e.g. to list pip installed packages):
`python.exe -m pip list`
# Note: You need to add `python.exe -m` before every command because (example for pip):
#	Virtual Environments:
#	* pip Directly:
#	    If you activate a virtual environment, the pip executable in that environment’s bin or Scripts directory is used.
#	    However, in some cases, especially with misconfigured environments, the wrong pip might be invoked.
#	
#	* python -m pip:
#	    When inside a virtual environment, using python -m pip ensures that pip runs within that environment, as it uses the
#       python interpreter from the active environment.

To deactivate the env:
Run `deactivate` or simply close the CMD window


----
PACKAGES:

Before beginning installing packages it might be a good idea to upgrade the default packages that come with the
environment (they should only be pip and maybe setuptools).
To do that run:
`python.exe -m pip list`                    To see all packages already installed (like I said, they should only be pip and maybe setuptools)
`python.exe -m pip install --upgrade pip`   To upgrade pip (similarly for setuptools)

After that, first package I install is pytorch. I want to install pytorch 2.5.0 with the CUDA 12.4 integration. To do that (info from https://pytorch.org/):
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

Then I install all the packages listed in the 'requirements.txt' file:
PyAudio==0.2.14                 `python.exe -m pip install PyAudio==0.2.14`
faster-whisper==1.0.3           `python.exe -m pip install faster-whisper==1.0.3`
pvporcupine==1.9.5              `python.exe -m pip install pvporcupine==1.9.5`
webrtcvad-wheels==2.0.14        `python.exe -m pip install webrtcvad-wheels==2.0.14`
halo==0.0.31                    `python.exe -m pip install halo==0.0.31`
torch                           Already installed in the first step
torchaudio                      Already installed in the first step
scipy==1.14.1                   `python.exe -m pip install scipy==1.14.1`
websockets==v12.0               `python.exe -m pip install websockets==12.0` *
websocket-client==1.8.0         `python.exe -m pip install websocket-client==1.8.0`
openwakeword>=0.4.0             `python.exe -m pip install openwakeword` **
numpy<2.0.0                     `python.exe -m pip install faster-whisper==1.0.3` ***

* I think the ==v12.0 is a typo in the 'requirements.txt' file, the correct format is ==12.0 as I've written
** The latest version of openwakeword is 6.0, and by running 'install openwakeword' the latest version will be installed which is >=0.4.0
*** If you installed the packages in order, then by this point numpy 1.26.3 is probably already installed. Nothing wrong with running the command of course.


Finally I install any other packages needed, like RealtimeSTT, keyboard, pynput, etc
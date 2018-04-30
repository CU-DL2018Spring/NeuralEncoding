# Neural Decoding

## Prerequisite

1. Package dependencies
```
$ pip install tableprint
$ conda install -c conda-forge deepdish
$ pip install pyret
```

2. Use tensorflow as backend of keras.
```
export KERAS_BACKEND=tensorflow
```

## Gcloud specific instructions

1. ssh into gcloud via personal account
5. `screen -S [name]`
6. `source activate tensorflow`
7. `export "KERAS_BACKEND=tensorflow`
8. `./NeuralEncoding/scripts/run.sh`
9. `ctrl-a ctrl-d`
10. ...
11. `screen -list`
12. If screen is still alive, `screen -r [name]`

To kill screen, `exit`
To logout of the current `ssh` session, `exit`, `exit`, then `logout`

## Data Set
The authors of McIntosh et al. shared a portion of the data set (5 of 37 cells) with us that was used in the original research. Our training data set consists of 359,802 frames of dimension 50x50 drawn from a set of 5,000 grayscale images and the RGC firing rates in response to these frames. The time interval between frames is 10 ms. The firing rate is put in 10 ms bins and smoothed by a Gaussian filter.


## Acknowledgement
The authors would like to thank Professor Drori and Ted Moskovitz for their guidance and helpful suggestions throughout the project. The authors would also like to thank Lane McIntosh for generously sharing his retinal data set with us.

## Authors
Yan-Song Chen,
Daniel Jaroslawicz,
Jake Lee,

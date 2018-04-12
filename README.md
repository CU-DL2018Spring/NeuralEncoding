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

## Acknowledgement

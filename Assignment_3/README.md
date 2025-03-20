## Install the Requirements
This project uses a few packages that can be automatically installed using pip. Run the following command on the requirements.txt file to install them.
```
pip install -r requirements.txt
```

## Run the Hyperparameter Search
Before training the final model, it's probably a good idea to search for the best combination of hyperparameters for the dataset. Do so by running the following script.
```
python search_hyperparameters.py
```

Now, view the results of the search.
```
python view_train_results.py training_results
```

Look through the results.csv file produced by the previous command and select the combination of your choosen. I typically look for low validation loss.

## Configure and Train
Take your desired parameters and modify the configuration in train.py, then run it.
```
python train.py
```

## Run the Test
Inside of the test.py file you can configure difference combinations of Non-maximum suppression and confidence threshold. Then run the script.
```
python test.py
```

## (optional) Compress the Output Video
Once the test.py file has ran it's course, there will be an uncompressed video file placed in the target directory. Use the helpful command to compress the video to a smaller size.

```
ffmpeg -i <input-mp4> -vf scale=1280:720 -c:v libx265 -crf 28 -b:v 1M -c:a aac -b:a 128k <output-mp4>
```

# Logo AI
Uses CNN to find out which logo is present in the image.

## QuickStart
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python model.py
python gen_train_set.py
python gen_test_set.py
python test.py
```

## To add new logos
Example: starplus (keep the name in lowercase and no spaces)
1. Create a 'images' folder in the project's root directory. (/images)
2. Create a folder with the name of the logo in the 'images' folder. (/images/starplus)
3. Add the images of the logo in the folder created in step 2. (/images/starplus/example.png)
4. Add an official image of the logo in the 'dataset/logos' folder with same name. (/dataset/logos/starplus.png)
5. Run the gen_train_set.py file to generate the perprocess the images and prepare the dataset for training. (python gen_train_set.py)
6. Run the gen_test_set.py file to generate the fake test set to test the ai against. (python gen_test_set.py)

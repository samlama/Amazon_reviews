

Step 1: Open the cloned repository and create a conda environment. Activate the new environment
```
conda create -n amazonreview python=3.10
```
```
conda activate amazonreview
```

Step 2: Install the requirements file
```
pip install -r requirements.txt
```

Step 3: Run the app
```
flask --app api.py run
```

Step 4: The app will run on port 5000. 
```
localhost:5000
```

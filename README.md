https://github.com/git-julian/nd0821-c3-starter-code.git.

Census Income Inference API

	Note: Working in a command line environment is recommended for ease of use with Git and DVC. If you’re on Windows, using WSL1 or WSL2 is recommended.

Environment Setup
	1.	Install Conda:
If you don’t already have Conda installed, download and install Miniconda or Anaconda.
	2.	Create a New Environment:
Use the supplied requirements.txt or create your own environment. For example:

conda create -n census-env python=3.8 scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge


	3.	Install Git:
Install Git either through Conda:

conda install git

or via your system’s package manager, for example on Ubuntu:

sudo apt-get install git

Repositories
	1.	Project Directory and Version Control:
	•	Create a directory for the project:

mkdir census-income-api
cd census-income-api


	•	Initialize Git:

git init


	•	As you work on the code, commit your changes frequently. Trained models you intend to use in production must be committed to GitHub.

	2.	Connect to GitHub:
	•	Connect your local repository to GitHub.
	•	Push your repository to GitHub using the remote URL:
https://github.com/git-julian/nd0821-c3-starter-code.git
	3.	GitHub Actions:
	•	Set up GitHub Actions on your repository to run at least pytest and flake8 on push.
	•	Ensure the Python version in the GitHub Action matches your development environment.

Data
	1.	Download Data:
	•	Download census.csv from the provided data folder.
	•	Commit this file using DVC:

dvc add data/census.csv
git add data/census.csv.dvc .gitignore
git commit -m "Add raw census data"


	2.	Data Cleaning:
	•	Open the CSV in pandas to inspect the data.
	•	Clean the data (e.g., remove extra spaces) using your favorite text editor or script.
	•	Commit the cleaned data to DVC under a new name to preserve the raw version.

Model
	1.	Training the Model:
	•	Use the starter code (in the repository) to write a machine learning model that trains on the clean data.
	•	The code typically splits the data into training and test sets, processes the data (using functions like process_data), trains the model (with train_model), and evaluates performance.
	•	Save the trained model and artifacts using DVC and Git (e.g., with joblib.dump).
	2.	Unit Tests:
	•	Write unit tests for at least three functions in your model code.
	•	Create tests for functions such as data processing, model training, and inference.
	3.	Slice Analysis:
	•	Write a function that outputs the model performance on slices of the data (for example, based on categorical features).
	•	This helps understand how the model performs on different subgroups.
	4.	Model Card:
	•	Write a model card (using the provided template) documenting model performance, assumptions, and limitations.

API Creation
	1.	Building the API:
	•	Create a RESTful API using FastAPI. Your API should implement:
	•	GET on the root (/) returning a welcome message.
	•	POST on /inference that performs model inference.
	•	Use type hinting and a Pydantic model for the POST request body.
	•	Example API implementation (e.g., app.py or starter/app.py):

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Census Income Inference API",
    description="This API provides a greeting at the root (GET) and model inference at the /inference endpoint (POST).",
    version="1.0"
)

class InferenceInput(BaseModel):
    features: List[float] = Field(
        ...,
        example=[2.0, 1.5, 1.0, 0.5, 1.0],
        description="A list of numerical features used for model inference."
    )

@app.get("/", summary="Greeting Endpoint", tags=["Root"])
def read_root() -> dict:
    return {"message": "Welcome to the Census Income Inference API!"}

@app.post("/inference", summary="Run Model Inference", tags=["Inference"])
def run_inference(input_data: InferenceInput) -> dict:
    threshold = 5.0
    prediction = 1 if sum(input_data.features) > threshold else 0
    return {"prediction": prediction}


	2.	API Unit Tests:
	•	Write three unit tests to test the API:
	•	One for the GET endpoint.
	•	Two for the POST endpoint (testing different predictions).

API Deployment on Render
	1.	Deploying on Render:
	•	Create an account on Render.com.
	•	In the Render dashboard, create a new Web Service.
	•	Link your GitHub repository (https://github.com/git-julian/nd0821-c3-starter-code.git) and select the branch to deploy (e.g., main or master).
	2.	Configure the Service:
	•	Build Command:

pip install -r requirements.txt


	•	Start Command:
	•	If your API file is in the repository root as app.py:

uvicorn app:app --host 0.0.0.0 --port $PORT


	•	If your API file is inside a folder (e.g., starter/app.py), update the command to:

uvicorn starter.app:app --host 0.0.0.0 --port $PORT


	•	Continuous Deployment (CD):
Render automatically enables CD when you link your GitHub repo.
Verify that CD is enabled in your service’s Settings tab and take a screenshot of the configuration (save it as continuous_deployment.png).

	3.	Testing Your Live API:
	•	Open your service’s live URL in a browser. The GET endpoint should display a welcome message.
Capture a screenshot (save it as live_get.png).
	•	Write and run a script using the requests module to POST data to your /inference endpoint (see the next section).
Capture the output screenshot as live_post.png.

Example POST Script (post_request.py)

import requests

def main():
    # Replace <your-render-app-url> with your actual Render URL.
    url = "https://<your-render-app-url>.onrender.com/inference"
    
    # Define the JSON payload that matches the API's expected schema.
    payload = {
        "features": [2.0, 1.5, 1.0, 0.5, 1.0]
    }
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

Replace <your-render-app-url> with your actual Render service URL.

Final Submission Checklist
	•	Continuous Deployment Screenshot:
continuous_deployment.png showing CD enabled on Render.
	•	Live GET Screenshot:
live_get.png showing the GET endpoint response in your browser.
	•	Live POST Screenshot:
live_post.png showing the terminal output of the POST script.
	•	GitHub Repository:
https://github.com/git-julian/nd0821-c3-starter-code.git
	•	DVC Tracked Data and Models:
Ensure your data and trained models are tracked using DVC.
	•	Model Card:
Include a model card that describes your model, its performance, and its limitations.

By following this README, a user should be able to set up the environment, manage the data and model, build and test the API locally, and deploy the API on Render with continuous delivery.
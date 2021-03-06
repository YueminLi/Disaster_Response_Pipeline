# Disaster Response Pipeline Project
Classify Disaster Response Messages

Udacity Data Scientist Nanodegree Project
## Project Motivation
This project analyzes social networking site messages collected after natural disasters. Its major goal is to build up data processing and classification pipeline to facilitate resource allocation process. 

## Installation
### Initial Setup
1. Install [Anaconda](https://www.anaconda.com) if you have not installed it. Otherwise, skip this step.
2. Update Anaconda by typing `conda update -all` in Anaconda Prompt.
3. Clone this repository to your local machine using: 
   
   `$ git clone https://github.com/YueminLi/Disaster_Response_Pipeline.git`
4. Delete these two files: 

   `data/DisasterResponse.db` and `models/classifier.pkl`
### Running Steps
1. Run the followihng commands in the project's root directory to set up your database and model.
   - To run ETL pipeline that cleans data and stores in database:
      
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves:
      
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app:

    `python run.py`
3. Go to http://0.0.0.0:3001/

## File Descriptions
- app
   - template
      - `master.html`: main page of web app
      - `go.html`: classification result page of web app
   - `run.py`: Flask file that runs app

- data
   - `disaster_categories.csv`: data to process, from [Figure Eight](https://www.figure-eight.com) 
   - `disaster_messages.csv`: data to process, from [Figure Eight](https://www.figure-eight.com)
   - `process_data.py`: process text data
   - `DisasterResponse.db`: database to save clean data to

- models
   - `train_classifier.py`: train machine learning models
   - `classifier.pkl`: saved model 

- LICENSE

- README.md

## Author
Yuemin Li 

Github: https://github.com/YueminLi

LinkedIn: https://www.linkedin.com/in/yuemin-li-89166333/

## License
Usage is provided under the MIT License. See LICENSE for the full details.

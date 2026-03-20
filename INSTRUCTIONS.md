#Instrctutions to Run the Project

## Prequistes
Have the following installed
- Python 3.10+
- pip
- Git

## Clone the Repository
- open terminal
- run: git clone https://github.com/your-repo/Knowledge-Stability-and-Risk-in-LLMs.git
cd Knowledge-Stability-and-Risk-in-LLMs

## Set Up Virtual Environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows


## install dependencies
- pip install -r requirements.txt

## Configure API Keys
- Create .env file in your root directory and add the contents from below in you .env file
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
- Ensure .env is not commited to GitHub

## Prepare Dataset
- Verify dataset exists and ensure questions are formatted correctly
data/questions_clean.csv

## Colleect Baseline Responses
- Run the script and it should output a .csv file
python Scripts/collect_responses.py


## Grade Baseline Responses
- Run the script and it should output a .csv file
python Scripts/grade_baseline.py



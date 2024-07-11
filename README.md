# Storyboard Automation

This project aims to develop a machine learning framework to automate the conversion of textual advertisement descriptions into visually compelling storyboards. This process enhances creativity and efficiency in digital advertising campaigns.

## Project Structure
    storyboard/
    │
    ├── data/ # Data files
    │ ├── raw/ # Raw data
    │ └── processed/ # Processed data
    │
    ├── notebooks/ # Jupyter notebooks for exploration and analysis
    │ └── EDA.ipynb
    │
    ├── src/ # Source code
    │ ├── init.py # Makes src a module
    │ ├── data_loader.py # Code for loading and processing data
    │ ├── models.py # Code for defining ML models
    │ ├── train.py # Code for training models
    │ ├── evaluate.py # Code for evaluating models
    │ └── utils.py # Utility functions
    │
    ├── tests/ # Unit tests
    │ ├── init.py
    │ └── test_data_loader.py
    │ └── test_models.py
    │
    ├── venv/ # Virtual environment directory
    │
    ├── .gitignore # Git ignore file
    ├── README.md # Project description
    └── requirements.txt # List of required packages

## Getting Started

1. **Clone the repository:**

   ```sh
   git clone https://github.com/dev-abuke/Automated_Ad_Storyboard_Synthesis.git
   ```

2. **Navigate to the project directory:**
   
   ```sh
   cd Automated_Ad_Storyboard_Synthesis
   ```

3. **Set up the virtual environment:**
   
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install the dependencies:**
   
   ```sh
   pip install -r requirements.txt
   ```


## Work Plan

- Implement more advanced image segmentation models.
- Integrate additional text analysis techniques.
- Develop a more sophisticated evaluation framework.
- Optimize the performance of the ML models.
- Expand the dataset with more diverse advertisement examples.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

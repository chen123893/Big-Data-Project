## Setup Instructions
1. Clone this repository.
2. Download the dataset from [Kaggle Transactions Fraud Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets).
3. Unzip the files and place them into the `data/raw/` directory.
4. Run the notebooks in sequential order (`01_...`, `02_...`).

Big-Data-Project/
|-- data/
|   |-- raw/          <-- Put the unzipped Kaggle files here
|   |-- processed/    <-- Save your cleaned, merged data here
|-- notebooks/        <-- Your Jupyter Notebooks go here
|-- app/
|   |-- streamlit_app.py  <-- Your Dashboard file
|-- models/           <-- Save your trained model .pkl files here
|-- requirements.txt  <-- List of libraries (pandas, scikit-learn, etc.)
|-- README.md         <-- Written project summary
#!/bin/bash
# Move to the project directory just in case
cd ~/Projects/personal/my_dashboard

# Activate the correct environment
source myenv/bin/activate

# Run the app
streamlit run app.py
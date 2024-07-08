from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the dictionary from the pickle file
with open('jobrecommendfinal (1).pkl', 'rb') as file:
    concatenated_data = pickle.load(file)

# Ensure the dataset contains the required columns
assert 'transformed_data' in concatenated_data.columns
assert 'designation' in concatenated_data.columns

def skillpredict(user_input):
    # Clone only the transformed_data column
    cloned_df = concatenated_data[['transformed_data']].copy()
    
    # Convert user input to DataFrame and append
    new_row = pd.DataFrame({'transformed_data': [user_input]})
    cloned_df = pd.concat([cloned_df, new_row], ignore_index=True)
    
    # Vectorize the data
    cv = TfidfVectorizer(max_features=100)
    vector = cv.fit_transform(cloned_df['transformed_data']).toarray()
    
    # Calculate cosine similarities
    last_row = vector[-1].reshape(1, -1)
    similarities = cosine_similarity(last_row, vector[:-1])
    
    # Get top 655 similar items
    distances1 = sorted(list(enumerate(similarities[0])), reverse=True, key=lambda x: x[1])[:655]
    
    roles = []
    for i in distances1:
        roles.append(concatenated_data.iloc[i[0]].designation)
    
    # Count the occurrences of each role
    values = Counter(roles)
    
    # Calculate the percentage of each role and round to two decimal places
    total = len(distances1)
    percentages = {role: round((count / total) * 100, 2) for role, count in values.items()}
    
    # Sort the percentages in descending order
    sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    output = [{"role": role, "percentage": percentage} for role, percentage in sorted_percentages]
    
    return output
    
    
@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ensure JSON data is present in the request
        if request.is_json:
            # Parse JSON data
            data = request.get_json()
            
            # Extract individual skills with default empty string
            skill1 = data.get('skill1', '')
            skill2 = data.get('skill2', '')
            skill3 = data.get('skill3', '')
            skill4 = data.get('skill4', '')
            skill5 = data.get('skill5', '')
            
            # Combine skills into a single string with spaces in between
            user_input = f"{skill1} {skill2} {skill3} {skill4} {skill5}".strip()
            
            if user_input:
                # Call the skillpredict function
                unique_designations = skillpredict(user_input)

                # Return JSON response
                return jsonify(unique_designations)
            else:
                return jsonify({'error': 'Invalid JSON data, missing skills'}), 400
        else:
            return jsonify({'error': 'Invalid request, JSON data expected'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Run the app on all available IP addresses and the specified port
    app.run(host='0.0.0.0', port=port)

import pickle

# Load the trained model using pickle
with open('spotify_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract the feature names
feature_names = model.feature_names_in_

# Save the feature names to a file
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

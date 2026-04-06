import pandas as pd
import time
import csv
from google import genai

# 1. Setup your Client
# Replace the text inside the quotes with your actual API key
client = genai.Client(api_key="AIzaSyCS2V2DN1zWaX7VM2dRPXrO7F_noOmAZYI")

# 2. Define the categories you want to expand
categories = [
    "Traffic_Harassment",
    "Tenant_Rights",
    "Cybercrime_Financial_Fraud",
    "Cybercrime_Harassment",
    "Consumer_Protection",
    "Employment_Dispute",
    "Property_Dispute",
    "Public_Nuisance",
    "Theft"
]

all_data = []

# 3. Ask the LLM to generate data
for category in categories:
    print(f"Generating data for: {category}...")
    
    prompt = f"""
    You are an expert dataset generator. 
    Generate 20 informal, everyday scenarios written by a distressed Indian citizen that fall under the legal category: '{category}'.
    The text should sound like a normal person explaining their problem. Include minor grammatical errors or slang occasionally for realism.
    Output ONLY the scenarios, one on each line. Do not include numbers, bullet points, or any introductory text.
    """
    
    try:
        # Using the new SDK syntax and the fast 2.5-flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Split the response into individual lines
        scenarios = response.text.strip().split('\n')
        
        for scenario in scenarios:
            # Clean up any stray quotes or bullets
            clean_scenario = scenario.strip(' "-*1234567890.')
            if clean_scenario:
                all_data.append([clean_scenario, category])
                
        # Pause to avoid hitting API rate limits
        time.sleep(2)
        
    except Exception as e:
        print(f"Error generating for {category}: {e}")

# 4. Save everything to a new CSV file
df = pd.DataFrame(all_data, columns=['User_Scenario', 'Legal_Category'])
df.to_csv('expanded_legal_dataset.csv', index=False, quoting=csv.QUOTE_ALL)

print("Dataset generation complete! Saved as 'expanded_legal_dataset.csv'.")
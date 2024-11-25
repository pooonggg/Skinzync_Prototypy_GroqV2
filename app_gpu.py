import os
# Set the environment variable to use GPU1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import re
import json
import random
import traceback
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma

app = Flask(__name__)

# Initialize LLM and vector database without device parameter
local_model = "llama-3.1-70b-versatile"
llm = ChatOllama(model=local_model)

loaded_vector_db = Chroma(
    persist_directory='vector_db',
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)

# Create a MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    loaded_vector_db.as_retriever(),
    llm
)

# Define the prompt template
template = """
You are a Cosmetic Formula Generator. Create 3 unique formulas based on the following specifications:

pH: {ph}
Viscosity (cps): {viscosity}
Appearance: {appearance}

Use the following context to help generate the formulas:
{context}

Requirements:
1. Generate exactly 3 formulas matching the given specifications.
2. Each formula must have at least 10 ingredients.
3. The sum of %w/w for all ingredients in each formula must equal exactly 100%.
4. If the sum of the ingredients exceeds 75%, use "di water" to adjust the total percentage to 100%.
5. Ensure that each formula has a diverse set of ingredients without excessive repetition across the formulas.
6. Use ingredients and functions from the `unique_ingredient_function.json` file.
7. Estimate viscosity contributions using the `viscosity_builder.json` file.
8. Each formula should be distinct in terms of its ingredient composition and function, while still meeting the pH, viscosity, and appearance requirements.

Output Format:
Respond ONLY with a valid JSON object structured as follows:

{
    "Formulas": [
        {
            "pH": <pH_value>,
            "Viscosity (cps)": <viscosity_value>,
            "Appearance": "<appearance_description>",
            "Ingredients": [
                {
                    "Ingredient": "<ingredient_name>",
                    "Phase": "<phase>",
                    "%w/w": <percentage>,
                    "Function": "<function>",
                    "Supplier": "<supplier>"
                },
                ...
            ]
        },
        ...
    ]
}

Ensure the output is a valid JSON object with no additional text or explanations. The formulas must each be unique in composition while meeting the required pH, viscosity, and appearance criteria.
"""

prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = (
    {
        "context": retriever,
        "ph": RunnablePassthrough(),
        "viscosity": RunnablePassthrough(),
        "appearance": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Function to generate formulas
def generate_formulas(ph, viscosity, appearance):
    return chain.invoke({
        "ph": ph,
        "viscosity": viscosity,
        "appearance": appearance
    })

# Rest of your code remains the same...

# Function to add the formulation to JSON file
def add_formulation_to_json(formulation_json_string, user_id):
    # Clean the JSON string
    cleaned = re.sub(r'^```json\s*', '', formulation_json_string)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    
    # Convert JSON string to Python dictionary
    formulation_data = json.loads(cleaned)
    
    # Define filename
    filename = 'json/LLM_Formulations.json'
    
    # Check if file exists
    if os.path.exists(filename):
        # If exists, read existing data
        with open(filename, 'r') as file:
            existing_data = json.load(file)
        
        # Append new data
        existing_data['Formulations'].append({
            'user_id': user_id,
            'formulas': formulation_data['Formulas']
        })
    else:
        # If not, create new data structure
        existing_data = {'Formulations': [{
            'user_id': user_id,
            'formulas': formulation_data['Formulas']
        }]}
    
    # Save back to file
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=2)
    
    print(f"Data has been added to {filename}")

# Function to process the LLM output and structure it as required
def process_formulation_output(formulation_json_string, user_id, formula_input):
    # Clean the JSON string
    cleaned = re.sub(r'^```json\s*', '', formulation_json_string)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    
    # Convert JSON string to Python dictionary
    formulation_data = json.loads(cleaned)
    
    # Initialize the formula_output dictionary
    formula_output = {}
    
    # Randomly generate Additional Properties
    properties_list = [
        "absorption time",
        "Advance delivery system",
        "Matte-Finish and Oil control",
        "Long lasting hydration",
        "Spreadability",
        "Ease of formulating"
    ]
    
    for idx, formula in enumerate(formulation_data['Formulas'], start=1):
        # Prepare Ingredients list with Supplier added
        ingredients = []
        for ingredient in formula['Ingredients']:
            ingredients.append({
                "Ingredient": ingredient["Ingredient"],
                "Phase": ingredient["Phase"],
                "%w/w": ingredient["%w/w"],
                "Function": ingredient["Function"],
                "Supplier": "-"  # Add Supplier as "-"
            })
        
        # Generate random values for Additional Properties (1-5)
        additional_properties = {prop: random.randint(1, 5) for prop in properties_list}
        
        # Construct the formula dictionary
        formula_output[f'formula_{idx}'] = {
            "pH": formula_input['pH'],
            "Viscosity (cps)": formula_input['Viscosity (cps)'],
            "Appearance": formula_input['Appearance'],
            "Ingredients": ingredients,
            "Additional Properties": additional_properties
        }
    
    return {'formula_output': formula_output}

# Set up session secret key
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Web page route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        ph = float(request.form.get('ph'))
        viscosity = float(request.form.get('viscosity'))
        appearance = request.form.get('appearance')
        
        formula_input = {
            'pH': ph,
            'Viscosity (cps)': viscosity,
            'Appearance': appearance
        }
        
        result = generate_formulas(ph=ph, viscosity=viscosity, appearance=appearance)
        
        # Process the result
        processed_result = process_formulation_output(result, user_id, formula_input)
        
        # Save the result
        add_formulation_to_json(result, user_id)
        
        # Store the data in session to pass to the next page
        session['user_id'] = user_id
        session['formula_input'] = formula_input
        session['formula_output'] = processed_result['formula_output']
        
        return redirect(url_for('formulation'))
    else:
        return render_template('index.html')

# Route for the formulation page
@app.route('/formulation')
def formulation():
    user_id = session.get('user_id')
    formula_input = session.get('formula_input')
    formula_output = session.get('formula_output')
    
    if not formula_output:
        return redirect(url_for('index'))
    
    return render_template('formulation.html', formula_input=formula_input, formula_output=formula_output)

# API endpoint
@app.route('/api/generate_formulas', methods=['POST'])
def api_generate_formulas():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input format'}), 400

    try:
        formula_input = data['formula_input']
        ph = float(formula_input['pH'])
        viscosity = float(formula_input['Viscosity (cps)'])
        appearance = formula_input['Appearance']
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except (TypeError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    try:
        result = generate_formulas(ph=ph, viscosity=viscosity, appearance=appearance)

        # Process the result to match the new output format
        processed_result = process_formulation_output(result, None, formula_input)

        # Save the result (user_id is not needed here)
        add_formulation_to_json(result, None)

        # Return the result without user_id
        return jsonify({
            'formula_input': formula_input,
            'formula_output': processed_result['formula_output']
        }), 200
    except Exception as e:
        # Log the exception traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# API documentation route
@app.route('/api/docs')
def api_docs():
    return render_template('api_docs.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=False, host='0.0.0.0', port=5000)

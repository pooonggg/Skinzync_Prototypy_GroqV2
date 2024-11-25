from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS

import re
import os
import json
import random
import traceback

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "https://skinzync.com",
            "https://www.skinzync.com",
            "http://api.skinzync.com"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Set up environment variables
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.1
)

# Initialize the vector database
embedding_function = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory='vector_db',
    embedding_function=embedding_function,
    collection_name="Building-Block"
)

# Create a retriever with adjusted settings
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Retrieve top 5 most similar documents
)

# Define the prompt template with placeholders for retrieved context
template = """
You are a Cosmetic Formula Generator. Create 3 unique formulas based on the following specifications:

Formulation Type: {formulationType}
Variant: {variant}
Dosage Form: {dosageForm}
Functions: {functions}
Appearance: {appearances}
Texture: {texture}
pH: {ph}
Viscosity Range (cps): {minViscosity} - {maxViscosity}

Additional Input:

- Each formula has specified phase compositions (%w/w) as follows:

{phase_compositions}

- Available ingredients for each phase:

{available_ingredients}

- Ingredient Details:

{ingredient_details}

Requirements:

1. Generate exactly 3 formulas matching the given specifications and phase compositions.
2. Each formula must have at least 10 ingredients.
3. The sum of %w/w for all ingredients in each formula must equal exactly 100%, with ingredients in each phase summing to the specified %w/w for that phase.
4. Select ingredients based on the input data and customer requirements.
5. Be aware of ingredient compatibility: some ingredients cannot be used together, while others have synergistic effects when combined.
6. The concentration of each ingredient must be within its specific allowable range.
7. Each phase can include multiple ingredients, but the total %w/w of ingredients in each phase must equal the specified percentage for that phase in each formula.
8. Ensure diversity of ingredients across the formulas without excessive repetition.
9. Each formula should be distinct in ingredient composition and function while meeting the given specifications.

Output Format:

Respond ONLY with a valid JSON object structured as follows:

{{
    "Formulas": [
        {{
            "Formulation Type": "<formulation_type>",
            "Variant": "<variant>",
            "Dosage Form": "<dosage_form>",
            "Functions": "<functions>",
            "Appearance": "<appearance>",
            "Texture": ["<texture1>", "<texture2>", ...],
            "pH": <pH_value>,
            "Viscosity (cps)": <viscosity_value_within_range>,
            "Ingredients": [
                {{
                    "Ingredient": "<ingredient_name>",
                    "Phase": "<phase>",
                    "%w/w": <percentage>,
                    "Function": "<function>",
                    "Supplier": "<supplier>",
                    "Note": "<note_if_any>"
                }},
                ...
            ]
        }},
        ...
    ]
}}

Ensure the output is a valid JSON object with no additional text or explanations. The formulas must each be unique in composition while meeting the required specifications.
"""

prompt_template = ChatPromptTemplate.from_template(template)

# Function to generate random phase compositions
def generate_phase_compositions(num_formulas=3):
    ingredient_ranges = {
        "Water": (65, 80),
        "Humectant": (3, 10),
        "Sensory enhancer/film-former": (1, 5),
        "Thickeners": (1, 3),
        "Emulsifiers": (2, 10),
        "Emollients": (5, 20),
        "Preservative": (1, 2),
        "Actives": (1, 5)
        # "pH adjuster": (0.5, 2)
    }
    phase_compositions = {}
    for i in range(1, num_formulas + 1):
        ingredient_percentages = {}
        total_percentage = 0
        for ingredient, (min_percent, max_percent) in ingredient_ranges.items():
            percent = random.uniform(min_percent, max_percent)
            ingredient_percentages[ingredient] = percent
            total_percentage += percent
        # Normalize percentages to sum to 100%
        scaling_factor = 100 / total_percentage
        for ingredient in ingredient_percentages:
            ingredient_percentages[ingredient] *= scaling_factor
            ingredient_percentages[ingredient] = round(ingredient_percentages[ingredient], 2)
        phase_compositions[f'formula_{i}'] = ingredient_percentages
    return phase_compositions

# Updated function to generate available ingredients based on user textures
def get_available_ingredients(user_textures):
    # Define the new data structure for ingredients with texture references and notes
    ingredients_data = {
        "Thickeners": [
            {
                "inci_name": "Xanthan gum",
                "textures": [
                    {"texture": "light"},
                    {"texture": "Rich"}
                ]
            },
            {
                "inci_name": "Carbomer",
                "textures": [
                    {"texture": "All"}
                ]
            },
            {
                "inci_name": "Ammonium Acryloyldimethyltaurate/VP Copolymer",
                "textures": [
                    {"texture": "All"}
                ]
            },
            {
                "inci_name": "Dibutyl Lauroyl Glutamide",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"}
                ]
            },
            {
                "inci_name": "PEG-240/HDI COPOLYMER BISDECYLTETRADECETH ETHER, BUTYLENE GLYCOL, WATER",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Caprylic/Capric Triglyceride, Sodium Acrylates Copolymer",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            }
        ],
        "Emulsifiers": [
            {
                "inci_name": "CETEARYL ALCOHOL",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "only 0.5-1%, add with Polysorbate 20 or ADEKA NOL GT-730 or Plantasens® Emulsifier CCT"}
                ]
            },
            {
                "inci_name": "Cetearyl Alcohol (and) Polysorbate 60 (and) PEG-150 Stearate (and) Steareth-20",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "only 0.5-1%, add with Polysorbate 20 or ADEKA NOL GT-730 or Plantasens® Emulsifier CCT"}
                ]
            },
            {
                "inci_name": "Cetearyl Alcohol (and) Polysorbate 60 (and) PEG-150 Stearate (and) Steareth-21",
                "textures": [
                    {"texture": "Silky"}
                ]
            },
            {
                "inci_name": "Cetearyl Alcohol (and) Polysorbate 60 (and) PEG-150 Stearate (and) Steareth-22",
                "textures": [
                    {"texture": "light", "note": "only 0.5-1%, add with Polysorbate 20 or ADEKA NOL GT-730 or Plantasens® Emulsifier CCT"}
                ]
            },
            {
                "inci_name": "Cetearyl Alcohol, Cetyl Palmitate, Sorbitan Palmitate, Sorbitan Oleate",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "with water 75-80%"}
                ]
            },
            {
                "inci_name": "Polysorbate 20",
                "textures": [
                    {"texture": "watery"}
                ]
            },
            {
                "inci_name": "Polysorbate 21",
                "textures": [
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Glyceryl Stearate (and) PEG-100 Stearate",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "only 0.5-1%, add with Polysorbate 20 or ADEKA NOL GT-730 or Plantasens® Emulsifier CCT"}
                ]
            },
            {
                "inci_name": "Glyceryl Stearate Citrate",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "only 0.5-1%, add with Polysorbate 20 or ADEKA NOL GT-730 or Plantasens® Emulsifier CCT"}
                ]
            },
            {
                "inci_name": "Cetearyl Alcohol and Coco-Glucoside",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Polyquaternium-37 & Mineral Oil & C11-15 Pareth",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Caprylic / Capric Triglyceride, Water, Glycerin, Lauryl Glucoside, Glyceryl Stearate, Sodium Lauroyl Lactylate, Cetearyl Alcohol, Sodium Stearoyl Lactylate",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Cetearyl Olivate, Sorbitan Olivate",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "with water 75-80%"}
                ]
            }
        ],
        "Emollients": [
            {
                "inci_name": "Isoamyl Laurate",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Cocos nucifera oil",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"}
                ]
            },
            {
                "inci_name": "ricinus communis seed oil (and) hydrogenated castor oil (and) copernicia cerifera wax",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"}
                ]
            },
            {
                "inci_name": "C12-15 alkyl benzoate",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "caprylic/capric triglyceride",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Squalane (and) Caprylic/Capric Triglyceride (and) Behenyl Behenate (and) Tribehenin",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"}
                ]
            },
            {
                "inci_name": "Olea europaea fruit oil",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"}
                ]
            },
            {
                "inci_name": "Vitis vinifera seed oil",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Isononyl isononanoate",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Butyrospermum Parkii (Shea) Butter",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"}
                ]
            }
        ],
        "Sensory Enhancer": [
            {
                "inci_name": "Fumed Silica",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "only 1% and Emollient should be only 5% and add Cyclopentasilcone instead"}
                ]
            },
            {
                "inci_name": "Cyclopentasiloxane",
                "textures": [
                    {"texture": "watery"},
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Hydrolyzed corn starch",
                "textures": [
                    {"texture": "Rich"},
                    {"texture": "Silky"},
                    {"texture": "light", "note": "only 0.5-1%"}
                ]
            },
            {
                "inci_name": "Nylon-12",
                "textures": [
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Oryza Sativa Starch",
                "textures": [
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            },
            {
                "inci_name": "Dimethicone",
                "textures": [
                    {"texture": "Silky"},
                    {"texture": "light"}
                ]
            }
        ]
    }

    # Categories that have the new structure
    categories_to_filter = ["Thickeners", "Emulsifiers", "Emollients", "Sensory Enhancer"]

    filtered_ingredients = {}

    for category, ingredients in ingredients_data.items():
        filtered_list = []
        for ingredient in ingredients:
            for texture_info in ingredient["textures"]:
                texture = texture_info["texture"].lower()
                # Check if 'All' is specified or texture matches user input
                if texture == "all" or texture in [t.lower() for t in user_textures]:
                    # Include the ingredient
                    ingredient_entry = {
                        "inci_name": ingredient["inci_name"],
                        "note": texture_info.get("note", "")
                    }
                    filtered_list.append(ingredient_entry)
                    break  # Avoid adding the same ingredient multiple times
        if filtered_list:
            filtered_ingredients[category] = filtered_list

    # Add the unmodified categories
    unmodified_categories = {
        "Humectant": [
            'Sodiumpolyaspartate', 'Sodium Acetyl hyaluronate',
            'Trehalose', 'Saccharomyces/Rice ferment filtrate PentyleneGlycol and Ethylhexylglycerin', 'Aloe Barbadensis Leaf Juice', 'Glycerin', 'Propylene glycol'
        ],
        "Preservatives": [
            'Phenoxyethanol', 'Benzyl alcohol ROTI®STAR Primary Standard',
            'METHYL PARABEN BP', 'Sodium benzoate', 'CMIT/MIT',
            'Germall Plus Liquid', '1,2-Hexanediol'
        ],
        # "pH Adjusters": [
        #     'Triethanolamine (TEA)', 'Citric Acid 50% Solution',
        #     'Lactic acid', 'Sodium hydroxide'
        # ],
        "Actives": []  # Will be filled based on functions
    }

    filtered_ingredients.update(unmodified_categories)

    return filtered_ingredients

# Function to retrieve ingredient details from RAG
def get_ingredient_details(functions):
    query = f"Provide detailed information about active ingredients for {functions} functions."
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No ingredient details found for the specified functions."
    # Combine the retrieved documents into a single string
    ingredient_details = "\n".join([doc.page_content for doc in docs])
    return ingredient_details

# Function to generate formulas
def generate_formulas(formula_input):
    phase_compositions = generate_phase_compositions(num_formulas=3)
    user_textures = formula_input.get('texture', [])
    available_ingredients = get_available_ingredients(user_textures)
    ingredient_details = get_ingredient_details(formula_input['functions'])

    # Prepare the messages
    messages = prompt_template.format_messages(
        formulationType=formula_input['formulationType'],
        variant=formula_input['variant'],
        dosageForm=formula_input['dosageForm'],
        functions=formula_input['functions'],
        appearances=formula_input['appearances'],
        texture=', '.join(formula_input['texture']),
        ph=formula_input['ph'],
        minViscosity=formula_input['minViscosity'],
        maxViscosity=formula_input['maxViscosity'],
        phase_compositions=json.dumps(phase_compositions, indent=4),
        available_ingredients=json.dumps(available_ingredients, indent=4),
        ingredient_details=ingredient_details
    )

    # Generate the response
    response = llm(messages)
    # Extract the content from the response
    if isinstance(response, AIMessage):
        content = response.content
    else:
        content = response[0].content  # Adjust based on the response type

    return content, phase_compositions

# Function to process the LLM output and structure it as required
def process_formulation_output(formulation_json_string, formula_input, phase_compositions):
    try:
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
            "Absorption Time",
            "Advance Delivery System",
            "Matte-Finish and Oil Control",
            "Long Lasting Hydration",
            "Spreadability",
            "Ease of Formulating"
        ]

        for idx, formula in enumerate(formulation_data['Formulas'], start=1):
            # Prepare Ingredients list with Supplier and Note added
            ingredients = []
            for ingredient in formula['Ingredients']:
                ingredients.append({
                    "Ingredient": ingredient["Ingredient"],
                    "Phase": ingredient["Phase"],
                    "%w/w": ingredient["%w/w"],
                    "Function": ingredient["Function"],
                    "Supplier": ingredient.get("Supplier", "-"),
                    "Note": ingredient.get("Note", "")
                })

            # Generate random values for Additional Properties (1-5)
            additional_properties = {prop: random.randint(1, 5) for prop in properties_list}

            # Construct the formula dictionary
            formula_output[f'formula_{idx}'] = {
                "Formulation Type": formula_input['formulationType'],
                "Variant": formula_input['variant'],
                "Dosage Form": formula_input['dosageForm'],
                "Functions": formula_input['functions'],
                "Appearance": formula_input['appearances'],
                "Texture": formula_input['texture'],
                "pH": formula_input['ph'],
                "Viscosity (cps)": formula['Viscosity (cps)'],
                "Ingredients": ingredients,
                "Additional Properties": additional_properties,
                "Phase Compositions": phase_compositions.get(f'formula_{idx}', {})
            }

        return {'formula_output': formula_output}
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}

# Set up session secret key
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Web page route
@app.route('/', methods=['GET', 'POST'])
def index():
    # Define choices for the dropdowns
    formulation_types = ['Facial-Leave-on']
    variants = ['Moisturizers']
    dosage_forms = ['Emulsions/Creams']
    functions_list = ['Hydrating','Exfoliating','Anti-aging','Brightening/Whitening','Soothing/Calming','Anti-acne','Pore-minimizing','Barrier repair','Anti-pollution']
    appearances = ['Opaque', 'Translucent']
    texture_choices = ['Watery', 'Light', 'Rich', 'Silky']

    if request.method == 'POST':
        # Get form data
        formula_input = {
            'formulationType': request.form.get('formulationType'),
            'variant': request.form.get('variant'),
            'minViscosity': request.form.get('minViscosity'),
            'maxViscosity': request.form.get('maxViscosity'),
            'ph': request.form.get('ph'),
            'dosageForm': request.form.get('dosageForm'),
            'functions': request.form.get('functions'),
            'appearances': request.form.get('appearances'),
            'texture': request.form.getlist('texture')  # Now a list
        }

        # Initialize errors dictionary
        errors = {}

        # Validate fields
        # Example validation (you can expand this as needed)
        try:
            formula_input['ph'] = float(formula_input['ph'])
            formula_input['minViscosity'] = float(formula_input['minViscosity'])
            formula_input['maxViscosity'] = float(formula_input['maxViscosity'])
        except ValueError:
            errors['numeric'] = 'pH and Viscosity must be numeric values.'

        if not (1 <= len(formula_input['texture']) <= 7):
            errors['texture'] = 'Select between 1 to 7 textures.'

        # Additional validation can be added here

        if not errors:
            # If no errors, proceed to generate formulas
            try:
                result, phase_compositions = generate_formulas(formula_input)
                # Process the result
                processed_result = process_formulation_output(result, formula_input, phase_compositions)
                if 'error' in processed_result:
                    errors['general'] = processed_result['error']
                else:
                    # Store the data in session to pass to the next page
                    session['formula_input'] = formula_input
                    session['formula_output'] = processed_result['formula_output']
                    return redirect(url_for('formulation'))
            except Exception as e:
                traceback.print_exc()
                errors['general'] = 'An error occurred while processing your request. Please try again later.'
        # If there are errors, render the form again with errors and previous data
        return render_template('index.html',
                               errors=errors,
                               formula_input=formula_input,
                               formulation_types=formulation_types,
                               variants=variants,
                               dosage_forms=dosage_forms,
                               functions_list=functions_list,
                               appearances=appearances,
                               texture_choices=texture_choices)
    else:
        return render_template('index.html',
                               errors={},
                               formula_input={},
                               formulation_types=formulation_types,
                               variants=variants,
                               dosage_forms=dosage_forms,
                               functions_list=functions_list,
                               appearances=appearances,
                               texture_choices=texture_choices)

# Route for the formulation page
@app.route('/formulation')
def formulation():
    formula_input = session.get('formula_input')
    formula_output = session.get('formula_output')

    if not formula_output:
        return redirect(url_for('index'))

    return render_template('formulation.html', formula_input=formula_input, formula_output=formula_output)


# API endpoint to generate formulas
@app.route('/api/generate_formulas', methods=['POST'])
def api_generate_formulas():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input format'}), 400

    try:
        formula_input = data['formula_input']

        # Validate and convert numeric inputs
        formula_input['ph'] = float(formula_input['ph'])
        formula_input['minViscosity'] = float(formula_input['minViscosity'])
        formula_input['maxViscosity'] = float(formula_input['maxViscosity'])

        # Ensure texture is a list with 1-7 values
        if not isinstance(formula_input['texture'], list) or not (1 <= len(formula_input['texture']) <= 7):
            return jsonify({'error': 'Texture must be a list with 1 to 7 values.'}), 400

        result, phase_compositions = generate_formulas(formula_input)

        # Process the result to match the new output format
        processed_result = process_formulation_output(result, formula_input, phase_compositions)

        if 'error' in processed_result:
            return jsonify({'error': processed_result['error']}), 500

        # Return the result
        return jsonify({
            'formula_input': formula_input,
            'formula_output': processed_result['formula_output']
        }), 200
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except (TypeError, ValueError) as e:
        return jsonify({'error': 'Invalid input type or format'}), 400
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

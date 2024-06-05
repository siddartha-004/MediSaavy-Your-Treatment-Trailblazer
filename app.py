import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import base64
from sklearn import preprocessing
from IPython.display import Image, display
from io import BytesIO
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from streamlit_option_menu import option_menu
from PIL import Image,UnidentifiedImageError
import requests
API_KEY='AIzaSyC0WEvOVTw1MtrdwL22OrHMCBWPtnC4cDo'
SEARCH_ENGINE_ID='53360f088bc304848'
search_query='tribhuvan kirti rasa'
url='https://www.googleapis.com/customsearch/v1'

sym_des = pd.read_csv("./symtoms_df.csv")
precautions = pd.read_csv("./precautions_df.csv")
workout = pd.read_csv("./workout_df.csv")
description = pd.read_csv("./description.csv")
medications = pd.read_csv('./medications.csv')
diets = pd.read_csv("./diets.csv")
disease_classes={'arthritis':0, 'diarrhea':1, 'gastritis':2,'migraine':3}
gender_classes={'female':0, 'male':1}
severity_classes={'HIGH':0, 'LOW':1,'NORMAL':2}
age_classes=[ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
       38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
       55, 56, 57, 58, 59, 60]
# Function to add external CSS


# Function to load image and convert to bytes for sidebar
def img_to_bytes(img_path):
    img_path = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_path).decode()
    return encoded
def img_to_base64(img):
    """
    Convert a PIL image to a base64 string.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to set up sidebar
def sidebar():
    st.sidebar.markdown(f'''<img src='data:image/png;base64,{img_to_bytes("m.jpg")}' class='img-fluid' width=225 height=280>''', unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div style="padding: 10px;">
            <h2 style="color: #1f77b4;">Allopathy Recommendation System</h2>
            <p style="font-size: 14px; line-height: 1.6;">
                This recommendation system aids healthcare professionals in making informed decisions by analyzing patient data to provide evidence-based diagnosis and treatment suggestions. It enhances clinical decision-making and patient care outcomes, ultimately improving the quality of healthcare delivery.
            </p>
            <p style="font-size: 14px; line-height: 1.6;">
                By leveraging machine learning techniques and clinical expertise, this system empowers healthcare providers to deliver more effective and efficient care, leading to better patient outcomes and increased healthcare efficiency.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Load medicine data and similarity vector
medicines_dict = pickle.load(open('./medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
similarity = pickle.load(open('./similarity.pkl', 'rb'))
#herbal=pickle.load(open('./drugTree.pkl','rb'))

# Recommendation function for medicines
def recommend(medicine):
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_medicines = [medicines.iloc[i[0]].Drug_Name for i in medicines_list]
    return recommended_medicines

# Define the allopathic medicines and their associated diseases
medication_diseases = {
    "Aspirin": ["Pain", "Fever", "Inflammation"],
    "Paracetamol": ["Pain", "Fever"],
    "Ibuprofen": ["Pain", "Fever", "Inflammation"],
    "Amoxicillin": ["Bacterial Infections (e.g., Ear Infections, Sinus Infections, Pneumonia)"],
    "Ciprofloxacin": ["Bacterial Infections (e.g., Urinary Tract Infections, Respiratory Infections)"],
    "Metformin": ["Type 2 Diabetes"],
    "Insulin": ["Diabetes (Type 1 and Type 2)"],
    "Atorvastatin": ["High Cholesterol", "Heart Disease"],
    "Lisinopril": ["Hypertension", "Heart Failure"],
    "Amlodipine": ["Hypertension", "Angina"],
    "Hydrochlorothiazide": ["Hypertension", "Edema"],
    "Losartan": ["Hypertension", "Heart Failure"],
    "Omeprazole": ["Gastroesophageal Reflux Disease (GERD)", "Ulcers"],
    "Pantoprazole": ["GERD", "Ulcers"],
    "Ranitidine": ["GERD", "Ulcers"],
    "Simvastatin": ["High Cholesterol", "Heart Disease"],
    "Levothyroxine": ["Hypothyroidism"],
    "Warfarin": ["Blood Clots", "Stroke Prevention"],
    "Clopidogrel": ["Blood Clots", "Heart Attack Prevention"],
    "Metoprolol": ["Hypertension", "Angina", "Heart Failure"],
    "Prednisone": ["Inflammatory Conditions (e.g., Asthma, Rheumatoid Arthritis)"],
    "Cetirizine": ["Allergies", "Hay Fever"],
    "Loratadine": ["Allergies", "Hay Fever"],
    "Furosemide": ["Edema", "Heart Failure"],
    "Albuterol": ["Asthma", "Chronic Obstructive Pulmonary Disease (COPD)"],
    "Sertraline": ["Depression", "Anxiety Disorders"],
    "Fluoxetine": ["Depression", "Anxiety Disorders"],
    "Escitalopram": ["Depression", "Anxiety Disorders"],
    "Paroxetine": ["Depression", "Anxiety Disorders"],
    "Tramadol": ["Pain (Moderate to Severe)"],
    "Morphine": ["Severe Pain", "Palliative Care"],
    "Codeine": ["Mild to Moderate Pain", "Cough"],
    "Oxycodone": ["Moderate to Severe Pain"],
    "Gabapentin": ["Neuropathic Pain", "Seizures"],
    "Pregabalin": ["Neuropathic Pain", "Fibromyalgia"],
    "Aripiprazole": ["Schizophrenia", "Bipolar Disorder"],
    "Quetiapine": ["Schizophrenia", "Bipolar Disorder"],
    "Risperidone": ["Schizophrenia", "Bipolar Disorder"],
    "Alprazolam": ["Anxiety Disorders", "Panic Disorders"],
    "Diazepam": ["Anxiety Disorders", "Muscle Spasms", "Seizures"]
}
unique_diseases = sorted(list(set(disease for diseases in medication_diseases.values() for disease in diseases)))

# Create the medication-disease efficacy matrix
medication_data = []
labels = []
for medication, diseases in medication_diseases.items():
    efficacy_row = [1 if disease in diseases else 0 for disease in unique_diseases]
    medication_data.append(efficacy_row)
    labels.append(medication)
medication_data = np.array(medication_data)

# Define the k-NN classifier
class MedicationRecommendation:
    def __init__(self, medication_data, labels):
        self.medication_data = medication_data
        self.labels = labels
        self.model = KNeighborsClassifier(n_neighbors=5, metric='jaccard')
    
    def train(self):
        self.model.fit(self.medication_data, self.labels)
    
    def get_recommendations(self, diseases, top_n=5):
        disease_indices = [1 if disease in diseases else 0 for disease in unique_diseases]
        scores = self.model.kneighbors([disease_indices], n_neighbors=top_n, return_distance=False)
        return [self.labels[idx] for idx in scores[0]]

# Initialize the medication recommendation system
recommendation_system = MedicationRecommendation(medication_data, labels)

# Train the model
recommendation_system.train()

def herbal_prediction(disease, age, gender, severity):
    # Get the numerical values from the dictionaries
    input_disease = disease_classes[disease]
    input_age = age
    input_gender = gender_classes[gender]
    input_severity = severity_classes[severity]

    # Create the input feature vector
    input_features = [[input_disease, input_age, input_gender, input_severity]]
    print(input_features)
    model = pickle.load(open("./drugTree.pkl","rb"))
    prediction=model.predict(input_features)
    # Make prediction
    #prediction = herbal.predict(input_features)
    return prediction



# Create Streamlit app
st.title("MediSavvy :)  Your Treatment Trailblazer")

# Sidebar
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    svc = pickle.load(open('./svc.pkl','rb'))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]
st.sidebar.markdown(f'''<img src='data:image/png;base64,{img_to_bytes("5755446.png")}' class='img-fluid' width=185 height=150>''', unsafe_allow_html=True)

with st.sidebar:
     option= option_menu('     MediSavvy :)  Your Treatment Trailblazer',

                              [
                              'Home','WellnessPro:Get suggestions',
                              'Recommend medicines for diseases','AYUCARE','Recommend Alternatives'],
                              
                              menu_icon='capsule',
                              icons=[ 'house', 'basket fill','heart-pulse-fill','cup-hot-fill','shuffle'],
                              default_index=0,
                              styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
sidebar()
# Main content based on selected option
if option == 'Recommend Alternatives':
    st.header('Recommend Alternatives')
    st.write("Our application suggests alternatives to specific medicines by analyzing their intended effects and matching them with alternative options. This feature aids users in finding suitable substitutes based on their needs and preferences, promoting informed medication decisions for better health outcomes.")
    selected_medicine_name = st.selectbox('Type your medicine name whose alternative is to be recommended', medicines['Drug_Name'].values)
    if st.button('Recommend Medicine'):
        recommendations = recommend(selected_medicine_name)
        st.write(f"Alternatives to {selected_medicine_name}:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec} - [Click here](https://pharmeasy.in/search/all?name={rec})")

elif option == 'Recommend medicines for diseases':
    st.header('Recommend medicines for Diseases')
    st.write("Our platform provides personalized medicine recommendations by analyzing reported diseases. This tool receives disease identification, and are presented with suitable medications. It's a user-friendly tool for informed health management, enhancing well-being.")
    input_diseases = st.multiselect("Select your symptoms",unique_diseases)
    if st.button("Get Recommendations"):
        if input_diseases and input_diseases != ['']:
            recommendations = recommendation_system.get_recommendations(input_diseases)
            st.write("Top recommendations for the given diseases:")
            for disease in input_diseases:
                st.write(f"{disease.strip()}: {recommendations}")
        else:
            st.write("Please enter at least one disease.")
elif option=='AYUCARE':
    st.header('Herbal Medibot')
    st.write("Leveraging sophisticated algorithms and extensive herbal knowledge, Herbal Medibot assists users in managing various health conditions through natural remedies and holistic approaches.")
    disease = st.selectbox('Select Disease', list(disease_classes.keys()))
    age = st.selectbox('Select Age', age_classes)
    gender = st.selectbox('Select Gender', list(gender_classes.keys()))
    severity = st.selectbox('Select Severity', list(severity_classes.keys()))
    if st.button('Predict'):
        prediction = herbal_prediction(disease, age, gender, severity)
        st.write("Best herbal medicine is here:")
        st.write(f'Prediction: {prediction}')
        st.write("Suggestion:")
        data=pd.read_csv('./Book1.csv',sep = ",", encoding='latin',header=None)
        prediction_str = prediction[0]

       # Remove the substring ' and '
        cleaned_prediction = prediction_str.replace(' and ', '')
        print(cleaned_prediction)
        prediction_row = data[data.iloc[:, 0] == cleaned_prediction]
        link=(prediction_row.iloc[0,1])
        print(link)
        response = requests.get(link)
        try:
            img = Image.open(BytesIO(response.content))
            width_percent = (100 / float(img.size[0]))
            new_height = int((float(img.size[1]) * float(width_percent)))
                
                # Resize the image
            img = img.resize((100, new_height), Image.ANTIALIAS)
        
            desc=(prediction_row.iloc[0,2])
            desc=desc.replace(' and ','')
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; border: 2px solid #F1597C;border-radius: 8px; padding: 10px;">
        <div style="flex: 1; padding-right: 10px;">
            <img src="data:image/png;base64,{img_to_base64(img)}" style="max-width: 100%; height: auto;">
        </div>
        <div style="flex: 2;">
            <h5>{desc}</h5>
        </div>
    </div>
                """,
                unsafe_allow_html=True
            )
        except:
            st.write("technical error")
        st.header('Image Gallery')
        

        params={
        'q':prediction,
        'key':API_KEY,
        'cx':SEARCH_ENGINE_ID,
        'searchType':'image'
        }
        response=requests.get(url,params=params)
        results=response.json()['items']
        image_links = []
        for item in results:
            print(item['link'])
            image_links.append(item['link'])
        cnt=0
        num_cols = 4
        cols = st.columns(num_cols)
       
        for url in image_links:
            clean_url = url.split('?')[0]
            print("links:"+clean_url)
            cnt+=1
        
            if(cnt==8):
                break
            try:
                response = requests.get(clean_url)
                img = Image.open(BytesIO(response.content))
                width_percent = (100 / float(img.size[0]))
                new_height = int((float(img.size[1]) * float(width_percent)))
                
                # Resize the image
                img = img.resize((100, new_height), Image.ANTIALIAS)
                col = cols[cnt % num_cols]
                with col:
                    #st.image(img, width=100)  # Customize width as needed
                    st.write(f"Similar Pic {cnt + 1}")
                    # Use HTML and CSS to add a border around the image container
                    st.markdown(
                f"""
                <div style="border: 2px solid #4CAF50; padding: 10px;border-radius: 6px">
                    <img src="data:image/png;base64,{img_to_base64(img)}" width="100">
                </div>
                """,
                unsafe_allow_html=True
            )
                cnt += 1
            except (requests.RequestException, UnidentifiedImageError):
            # Skip the current image and continue with the next one
                continue
        
elif option=='WellnessPro:Get suggestions':
    st.header('WellnessPro: Symptom Diagnosis and Personalized Health Plans')
    st.write("WellnessPro is an innovative application designed to assist users in diagnosing health conditions based on symptoms and providing personalized health plans tailored to their needs. Users can input their symptoms, and the application utilizes advanced algorithms to predict possible diseases and offer comprehensive health plans for management and treatment.")
    #symptoms = st.text_input("Enter your symptoms (comma-separated)")
    symptoms = st.multiselect("Select your symptoms", list(symptoms_dict.keys()))
    symptoms= ', '.join(symptoms)
    if st.button('get prescription'):
        if symptoms:

            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)

            desc, pre, med, die, wrkout = helper(predicted_disease)

            st.write("### Predicted Disease")
            st.write(predicted_disease)

            st.write("### Description")
            st.write(desc)

            st.write("### Precautions")
            for i, p_i in enumerate(pre[0], 1):
                st.write(f"{i}: {p_i}")

            st.write("### Medications")
            for i, m_i in enumerate(med, 1):
                st.write(f"{i}: {m_i}")

            st.write("### Workout")
            for i, w_i in enumerate(wrkout, 1):
                st.write(f"{i}: {w_i}")

            st.write("### Diets")
            for i, d_i in enumerate(die, 1):
                st.write(f"{i}: {d_i}")

            

elif option=='Home':
    st.header("Welcome to the Medicine Recommendation System")
    st.write("""
        This system helps healthcare professionals in making informed decisions by analyzing patient data to provide evidence-based diagnosis and treatment suggestions.
        
       
    """)
    image = Image.open('./we.png')
    st.image(image, caption='Recommended Medicines')


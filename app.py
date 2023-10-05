import streamlit as st
import openai
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from gspread_pandas import Spread, Client
from google.oauth2 import service_account

# Page title
st.markdown("""
# Jaundice Medical App
This app is designed to help detect symptoms of Jaundice early and have the user report to a medical professional immediately if necessary.
            
""")

#---------------------------------#
# About
expander_bar = st.expander("Read more")
expander_bar.markdown("""
Jaundice is a condition that occurs when excess amounts of bilirubin circulating in the blood stream dissolve in the subcutaneous fat 
            (the layer of fat just beneath the skin), causing a yellowish appearance of the skin and the whites of the eyes, among other things.
""")

# Integrating a database that can store data from users
# Create a Google Authentication connection object
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

# Catch that error that occurs when connecting to the Google sheets API sometimes.
try:
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = Client(scope=scope, creds=credentials)
    spreadsheetname = "Database"
    spread = Spread(spreadsheetname, client=client)

    ## Check that the connection worked
    # st.write(spread.url)

    # Call our spreadsheet
    sh = client.open(spreadsheetname)
    worksheet_list = sh.worksheets()

except Exception as error:
    st.warning("Reload the website. A temporary error has occurred.")

# Functions
# @st.cache()
# Get our worksheet names
def worksheet_names():
    sheet_names = []   
    for sheet in worksheet_list:
        sheet_names.append(sheet.title)  
    return sheet_names

# Get the sheet as dataframe
def load_the_spreadsheet(spreadsheetname):
    worksheet = sh.worksheet(spreadsheetname)
    df = pd.DataFrame(worksheet.get_all_records())
    return df

# Update to Sheet
def update_the_spreadsheet(spreadsheetname, dataframe):
    spread.df_to_sheet(dataframe, sheet=spreadsheetname, index = False)
    st.info('Updated to GoogleSheet')

# Get username
def get_username(x):
    return x


columns = ['user', 'babyName', 'dateOfBirth', 'timeOfBirth', 'weeksOfPregnancy', 'weightAtBirth', 'gender', 'bruisedAtBirth', 'motherBloodGroup',
           'fatherBloodGroup', 'diabetes', 'exclusivelyBreastfed', 'wellBreastfed', 'otherSiblingsJaundice', 'hasFever', 
           'bruisingScalp', 'skinColor', 'stableTemperature']

df = pd.DataFrame(columns=columns)
# Create instances of the worksheets
sheet_names = worksheet_names()
df1 = load_the_spreadsheet(sheet_names[1])

# get users' usernames from the existing dataframe
usernames = set(df1.user.apply(get_username))

# Get usernames as unique identifiers
now = datetime.now()
new = st.radio("First time visiting this app?", ("", "No", "Yes"))

if new == "No":
    user = st.text_input("Enter your username to retrieve your information.")
    if user in usernames:
        dob = df1[df1['user'] == user]['dateOfBirth'].values[0]  # getting the value from a pandas Series
        st.info("You can proceed")

        # Subsequent checks
        seen_doctor = st.radio("Have you seen a doctor", ("Yes", "No"))

        # Stage 2
        st.write("Take your baby under natural light and select all the options that you notice.")
        eye_color = st.radio("Are his/her eyes yellow in color?", ("No", "Yes"))

        st.write("Still under natural light, press down on your baby's forehead and nose for two seconds and release;")
        skin_color = st.radio("Is the skin only slightly lighter than the normal color?", ("No", "Yes"))
        face_color = st.radio("Is the  face yellow?", ("No", "Yes"))

        st.write("Repeat the procedure for the baby's chest area")
        chest_color = st.radio("Is the chest area yellow?", ("No", "Yes"))

        st.write("Repeat the procedure for the baby's feet area")
        feet_color = st.radio("Are the feet yellow?", ("No", "Yes"))


        # Conditions for output recommendation
        datetime_str = datetime.strptime(dob, '%d/%m/%Y')
        baby_age_years = (now - datetime_str).days

        # Condition 1A
        if seen_doctor == 'No' and baby_age_years < 2 and (eye_color == "Yes" or chest_color == "Yes" or feet_color == "Yes"):
            st.write("## RECOMMENDATION")
            st.write("Your baby is at risk of Kernicterus and potentially cerebral palsy, let your doctor know the situation immediately.")
            st.write("### EXPLANATION")
            st.write("""
                        Your baby's bilirubin level is rising faster than he/she can handle. 
                        The bilirubin might cross the blood-brain barrier and lead to fatalities. Take the baby to the hospital to get help.\n

                        These are some of the information your doctor will need, ensure you have clear answers ready:
                        - Date of birth
                        - Weeks of pregnancy before birth (born before 37 weeks)
                        - Weight at birth (Less than 2.5kg)
                        - Gender
                        - Was your baby bruised at birth?
                        - What is your blood group?
                        - What is your husband's blood group?
                        - Do you have diabetes?
                        - Is your baby  exclusively breastfed?
                        - Is she breastfeeding well? (8 to 12 feedings a day)
                        - Do any of her older siblings have Jaundice at birth that lead to phototherapy or blood transfusion ?
                        - Does your baby have a fever
                        - Did your baby have bruising or swelling under the scalp ?
                        - Have you noticed any changes in your baby skin color?
            """)

        # Condition 1B
        elif seen_doctor == 'Yes' and baby_age_years < 2 and (eye_color == "Yes" or chest_color == "Yes" or feet_color == "Yes"):
            treatment = st.radio("What treatment method did your doctor suggest?", ("Medications", "Injections", "Phototherapy", "Blood exchange"))
            if treatment in ["Medications", "Injections"]:
                st.write("## RECOMMENDATION")
                st.write("""Your baby is still at risk, let the doctor know the situation and any other symptoms. 
                                    Ask if your baby can get a phototherapy or blood transfusion instead.""")
            elif treatment in ["Phototherapy", "Blood exchange"]:
                st.write("## RECOMMENDATION")
                st.write("""Insist that your baby be admitted and make sure to check for his/her jaundice regularly and inform the doctor 
                                    regularly of the progress.""")
            st.write("### EXPLANATION")
            st.write("""Your baby's bilirubin level is rising faster than he/she can handle. 
                        The bilirubin might cross the blood-brain barrier and lead to fatalities. Take the baby to the hospital to get help.\n

                        These are some of the information your doctor will need, ensure you have clear answers ready:
                        - Date of birth
                        - Weeks of pregnancy before birth (born before 37 weeks)
                        - Weight at birth (Less than 2.5kg)
                        - Gender
                        - Was your baby bruised at birth?
                        - What is your blood group?
                        - What is your husband's blood group?
                        - Do you have diabetes?
                        - Is your baby  exclusively breastfed?
                        - Is she breastfeeding well? (8 to 12 feedings a day)
                        - Do any of her older siblings have Jaundice at birth that lead to phototherapy or blood transfusion ?
                        - Does your baby have a fever
                        - Did your baby have bruising or swelling under the scalp ?
                        - Have you noticed any changes in your baby skin color?
            """)
    
        # Condition 2
        elif seen_doctor == 'Yes' and baby_age_years > 2 and eye_color == "Yes" and chest_color == "No" and feet_color == "No":
            treatment = st.radio("Did your doctor advise any of these?", ("Medications", "Injections", "Phototherapy", "Blood exchange", "I've not seen a doctor"))
            if treatment in ["Medications", "Injections", "Phototherapy", "Blood exchange"]:
                st.write("## RECOMMENDATION")
                st.write("Make sure to follow the doctor's advice and feed your baby regularly.")
                st.write("### EXPLANATION")
                st.write("""Your baby's jaundice is normal. 75% of babies have this type of Jaundice, it is mild and not a cause for worry. 
                                 Proper feeding can help your baby overcome this; if he/she isnt breast breastfeeding enough, 
                                 consider formula feeding based on your doctor's advice.""")
            else:
                st.write("## RECOMMENDATION")
                st.write("""Ensure your baby is feeding well (at least 8 times a day) and stooling properly(at least 5 times a day). 
                                    See a doctor if your baby also seems tired, isn't feeding well and isn't sleeping enough.""")
                st.write("### EXPLANATION")
                st.write("""Your baby's jaundice is most likely physiological, 75% of babies have this type of Jaundice, 
                                    it is mild and not a cause for worry. Proper feeding can help your baby overcome this; 
                                    if he/she isnt breast breastfeeding enough, consider formular feeding based on your doctor's advice.""")
                
        # Condition 3
        elif seen_doctor == 'Yes' and baby_age_years > 2 and eye_color == "No" and (chest_color == "Yes" or feet_color == "Yes"):
            treatment = st.radio("Did your doctor advise any of these?", ("Medications", "Injections", "Phototherapy", "Blood exchange", "I've not seen a doctor"))
            if treatment in ["Medications", "Injections", "Phototherapy", "Blood exchange"]:
                st.write("## RECOMMENDATION")
                st.write("Your baby's jaundice is on the rise still. Insist on your baby getting hospitalized and watched closely.")
                st.write("### EXPLANATION")
                st.write("""Jaundice already showing on the chest and feet indicates it is rising quickly. 
                                    Your doctor needs to monitor the baby closely to ensure the treatment is really working.""")
            else:
                st.write("## RECOMMENDATION")
                st.write("You are putting your baby at risk. Take him/her to the doctors immediately.")
                st.write("### EXPLANATION")
                st.write("""Jaundice already showing on the chest and feet indicates it is rising quickly. 
                                    Your doctor needs to monitor the baby closely to ensure the treatment is really working.""")
                
        else:
            st.write("## RECOMMENDATION")
            st.write("Take your baby to the doctor. No need to panic, this is just a precautionary measure")

    elif len(user) > 1 and user not in usernames:
        st.info("This username is not registered. Check spellings and capitalizations.", icon="🚨")
elif new == "Yes":
    user = st.text_input("Enter a unique username to track your information. There should be no spaces in the username.")
    if len(user.split()) > 1:
        st.info("There should be no spaces in the username", icon="🚨")
    elif user in usernames or len(user) == 1:
        st.info("This username already exists or the username is too short", icon="🚨")
    elif len(user) > 1:
        st.info("You can proceed")

        baby_name = st.text_input("What is the name of the baby?")
        dob = st.text_input("What is the baby's date of birth (DD/MM/YYYY)? Example: 24/05/2023", "24/05/2023")  # Default value as second argument 
        tob = st.text_input("What is the baby's time of birth (HH:MM:SS)? Example: 15:56:00")
        wop = st.text_input("Weeks of pregnancy before birth? Example: enter '37' for 37 weeks of pregnancy")
        wab = st.text_input("Weight of the baby at birth? Example: enter '2.5' for 2.5kg")
        gender = st.radio("Gender of the baby", ("Male", "Female"))
        bab = st.radio("Was your baby bruised at birth", ("No", "Yes"))
        wbg = st.radio("Blood group of the mother", ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"))
        hbg = st.radio("Blood group of the father", ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"))
        diabetes = st.radio("Does the wife have diabetes", ("No", "Yes"))
        ebf = st.radio("Is your baby exclusively breastfed", ("No", "Yes"))
        wbf = st.radio("Is your baby well breastfed", ("No", "Yes"))
        osj = st.radio("Do any of her older siblings have Jaundice at birth that lead to phototherapy or blood transfusion?", ("No", "Yes"))
        hf = st.radio("Does your baby have fever", ("No", "Yes"))
        bs = st.radio("Did your baby have bruising or swelling under the scalp?", ("No", "Yes"))
        sc = st.radio("Have you noticed any changes in your baby skin color?", ("No", "Yes"))
        temp = st.radio("Has your baby's temperature been stable?", ("No", "Yes"))


        # Next, we run the first time check here since we're sure that this is the first time they're using the app
        st.header("Diagnosis")
        st.subheader("1. Physical Examination")
        sleeping_through_feedings = st.radio("Is he/she consistently sleeping through feedings?", ("No", "Yes"))
        sleeping_19hrs = st.radio("Is he/she sleeping more than 19hrs?", ("No", "Yes"))
        avg_wet_diapers = st.radio("Average number of wet diapers daily?", ("Less than 5", "Greater than 5"))
        cries = st.radio("Is she crying high pitched cries?", ("No", "Yes"))
        vomitting = st.radio("Is your baby vomiting?", ("No", "Yes"))
        breastfed = st.radio("How many times have you breast-fed your baby in the last 24hrs?", ("0-5", "5-10", "10-15"))

        # Stage 2
        st.write("Take your baby under natural light and select all the options that you notice.")
        eye_color = st.radio("Are his/her eyes yellow in color?", ("No", "Yes"))

        st.write("Still under natural light, press down on your baby's forehead and nose for two seconds and release;")
        skin_color = st.radio("Is the skin only slightly lighter than the normal color?", ("No", "Yes"))
        face_color = st.radio("Is the  face yellow?", ("No", "Yes"))

        st.write("Repeat the procedure for the baby's chest area")
        chest_color = st.radio("Is the chest area yellow?", ("No", "Yes"))

        st.write("Repeat the procedure for the baby's feet area")
        feet_color = st.radio("Are the feet yellow?", ("No", "Yes"))


        # Conditions for output recommendation
        datetime_str = datetime.strptime(dob, '%d/%m/%Y')
        baby_age_years = (now - datetime_str).days

        # Condition 1
        if baby_age_years < 2 and (eye_color == "Yes" or chest_color == "Yes" or feet_color == "Yes"):
            st.write("## RECOMMENDATION")
            st.write("See a doctor immediately. This is an emergency case.")
            st.write("### EXPLANATION")
            st.write("""
                                Your baby most likely has Pathologic Jaundice, this means there is an underlying condition causing it. 
                                Go and see a healthcare professional immediately to check the cause of the jaundice.
                                These are some of the information your doctor will need, ensure you have clear answers ready:
                                - Date of birth
                                - Weeks of pregnancy before birth (born before 37 weeks)
                                - Weight at birth (Less than 2.5kg)
                                - Gender
                                - Was your baby bruised at birth?
                                - What is your blood group?
                                - What is your husband's blood group?
                                - Do you have diabetes?
                                - Is your baby  exclusively breastfed?
                                - Is she breastfeeding well? (8 to 12 feedings a day)
                                - Do any of her older siblings have Jaundice at birth that lead to phototherapy or blood transfusion ?
                                - Does your baby have a fever
                                - Did your baby have bruising or swelling under the scalp ?
                                - Have you noticed any changes in your baby skin color?
                                - Has your baby's temperature been stable?

            """)

        # Condition 2
        elif baby_age_years > 2 and eye_color == "Yes" and chest_color == "No" and feet_color == "No":
            st.write("## RECOMMENDATION")
            st.write("""Ensure your baby is feeding well (at least 8 times a day) and stooling properly(at least 5 times a day). 
                                See a doctor if your baby also seems tired, isn't feeding well and isn't sleeping enough.""")
            st.write("### EXPLANATION")
            st.write("""Your baby's jaundice is most likely physiological, 75% of babies have this type of Jaundice, 
                             it is mild and not a cause for worry. Proper feeding can help your baby overcome this; 
                             if he/she isnt breast breastfeeding enough, consider formular feeding based on your doctor's advice.""")
            
        # Condition 3
        elif baby_age_years > 2 and eye_color == "No" and (chest_color == "Yes" or feet_color == "Yes"):
            st.write("## RECOMMENDATION")
            st.write("""Your baby's jaundice is on the rise. See a doctor immediately. Your baby might require phototherapy.""")
            st.write("### EXPLANATION")
            st.write("""Jaundice already showing on the chest and feet indicates it is rising quickly. 
                                Please go to your doctor and explain all the other symptoms your baby suffers.\n
                                Here is a list of questions your doctor might ask. Have the answers in mind:
                                - Date of birth
                                - Weeks of pregnancy before birth (born before 37 weeks)
                                - Weight at birth (Less than 2.5kg)
                                - Gender
                                - Was your baby bruised at birth?
                                - What is your blood group?
                                - What is your husband's blood group?
                                - Do you have diabetes?
                                - Is your baby  exclusively breastfed?
                                - Is she breastfeeding well? (8 to 12 feedings a day)
                                - Do any of her older siblings have Jaundice at birth that lead to phototherapy or blood transfusion ?
                                - Does your baby have a fever
                                - Did your baby have bruising or swelling under the scalp ?
                                - Have you noticed any changes in your baby skin color?
                                - Has your baby's temperature been stable?
                                """)
            
        else:
            st.write("## RECOMMENDATION")
            st.write("Take your baby to the doctor. No need to panic, this is just a precautionary measure")
            
        # Create button to confirm write operation to the dataframe
        if st.button("Press this button to confirm all answers"):
            df.loc[len(df)] = [user, baby_name, dob, tob, wop, wab, gender, bab, wbg, hbg, diabetes, ebf, wbf, osj, hf, bs, sc, temp]

            # combine old records with new ones
            df = pd.concat([df1, df], ignore_index=True)

            # update the googlesheet
            update_the_spreadsheet("InitialProfile", df)

else:
    pass


# Computer vision to detect Jaundice on the skin
good = "All good. No signs of jaundice."
medium = "Watch your baby for more symptoms. No serious issue yet."
bad = "See a medical professional immediately! Your baby might be suffering from jaundice."

st.header("Skin detection")
st.write("""Jaundice may go unnoticed for a while if not carefully monitored. In this section, upload a photo of your baby so we can
         determine if there's a likelihood that the baby has Jaundice on the skin.
         """)
st.write("""#### Instructions""")
st.write("1. Upload a photo of your baby that shows the area of the body you want to scan.")
st.write("2. Ensure that the area of the skin you want to scan covers 80% of the whole image")
st.write("3. If unsure of which part of the body to scan, you could do them separately (for example, face first, chest area next then legs and feet).")

# code part starts
st.write("""### Choose image input method""")
options = ['Camera', 'Upload']
option = st.radio('Method', options, index=1)

img = None

if option == "Upload":
    uploaded_file = st.file_uploader("Upload your image in .jpg format", type=["jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        cv.imwrite('output.jpg', cv.cvtColor(img_array, cv.COLOR_RGB2BGR))

        img = cv.imread('output.jpg')
    else:
        st.write("Please upload an image of the baby you want to scan.")
elif option == 'Camera':
    image = st.camera_input('Capture Image', key='FirstCamera', 
                            help="""This is a basic camera that takes a photo to scan whether a baby has Jaundice. 
                                    Don\'t forget to allow access in order for the app to be able to use the devices camera.""")
    if image is not None:
        bytes_data = image.getvalue()
        img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
    else:
        st.write("Please take a snapshot of the baby you want to scan.")

st.sidebar.write("# RECOMMENDATION")
# After getting the image, run the computer vision code to scan for Jaundice
if img is not None:
    # Constants for finding range of skin color in YCrCb
    min_YCrCb = np.array([0,130,100], np.uint8)
    max_YCrCb = np.array([255,180,130], np.uint8)

    # Convert image to YCrCb
    imageYCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Get how much parts of the image are shaded
    # Calculate the total number of pixels in the mask
    total_pixels = skinRegion.size

    # Calculate the number of black pixels (pixels outside the color range)
    black_pixels = np.sum(skinRegion == 0)

    # Calculate the proportion of black shading
    black_shading_proportion = black_pixels / total_pixels

    if black_shading_proportion > 0.6:
        st.sidebar.write(bad)
    elif black_shading_proportion > 0.4:
        st.sidebar.write(medium)
    else:
        st.sidebar.write(good) 


# Eye detection
st.header("Eye detection")
st.write("""Jaundice may also appear in the form of yellow coloring of the eyes. Upload an image of the baby's face to check.
         """)
st.write("""#### Instructions""")
st.write("1. Upload a photo of your baby's face. The face/head in the must be upright and eyes must be open.")
st.write("2. If the image is rejected, try again. Remember, the face must be upright and eyes open")

st.write("""### Choose image input method""")
options = ['Camera', 'Upload']
option = st.radio('Method ', options, index=1)

eye_images = None

# function to get cropped images of the eyes
def get_eye_images(img):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

    # Convert the image to grayscale for face detection
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

    # loop over the detected faces
    if len(faces) == 1:
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # detects eyes of within the detected face area (roi)
        eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(30, 30))

        image_objects = []

        # draw a rectangle around eyes if there are two eyes
        if len(eyes) == 2:
            for idx, (ex,ey,ew,eh) in enumerate(eyes):
                cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,255), 2)
                cropped_eye = roi_color[ey:ey+eh, ex:ex+ew]
                image_objects.append(cropped_eye)
            # st.image(img, channels='BGR', caption='Image')
            return image_objects
        
        else:
            num_eyes = len(eyes)
            st.info("Detected " + str(num_eyes) + " eyes in the image. Try again")

    else:
        num_faces = len(faces)
        st.info("Detected " + str(num_faces) + " faces in the image. Try again")


if option == "Upload":
    uploaded_file = st.file_uploader("Upload your image in .jpg format ", type=["jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        cv.imwrite('output.jpg', cv.cvtColor(img_array, cv.COLOR_RGB2BGR))

        img = cv.imread('output.jpg')
        eye_images = get_eye_images(img)
        
    else:
        st.write("Please upload an image of the baby you want to scan.")
elif option == 'Camera':
    image = st.camera_input('Capture Image', key='FirstCamera', 
                            help="""This is a basic camera that takes a photo to scan whether a baby has Jaundice. 
                                    Don\'t forget to allow access in order for the app to be able to use the devices camera.""")
    if image is not None:
        bytes_data = image.getvalue()
        img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
        eye_images = get_eye_images(img)

    else:
        st.write("Please take a snapshot of the baby you want to scan.")

# After getting the image, run the computer vision code to scan for Jaundice
if eye_images is not None:
    total_black_proportion = 0
    for img in eye_images:
        # Constants for finding range of skin color in YCrCb
        min_YCrCb = np.array([0,130,100], np.uint8)
        max_YCrCb = np.array([255,180,130], np.uint8)

        # Convert image to YCrCb
        imageYCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

        # Find region with skin tone in YCrCb image
        skinRegion = cv.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        # output_image = cv.bitwise_and(img, img, mask=skinRegion)
        # st.image(output_image, channels='BGR', caption='Image1')

        # Get how much parts of the image are shaded
        # Calculate the total number of pixels in the mask
        total_pixels = skinRegion.size

        # Calculate the number of black pixels (pixels outside the color range)
        black_pixels = np.sum(skinRegion == 0)

        # Calculate the proportion of black shading and add to total
        black_shading_proportion = black_pixels / total_pixels
        total_black_proportion += black_shading_proportion

    # get the average black shading proportion and check for jaundice
    average_black_proportion = total_black_proportion / 2
    if average_black_proportion > 0.4:
        st.sidebar.write(bad)
        # st.sidebar.write(average_black_proportion)
        # st.sidebar.write(total_black_proportion)
    elif average_black_proportion > 0.2:
        st.sidebar.write(medium)
        # st.sidebar.write(average_black_proportion)
        # st.sidebar.write(total_black_proportion)
    else:
        st.sidebar.write(good) 
        # st.sidebar.write(average_black_proportion)
        # st.sidebar.write(total_black_proportion)



# Integrating openai
st.header("Any Questions? Ask!")

# set the GPT-3 api key
openai.api_key = st.secrets["api_key"]

prompt = st.text_input("Enter your question here")

try:
    if st.button("Get result"):
        # Use GPT-3 to get an answer
        response = openai.Completion.create(engine="text-davinci-003",
                                                prompt=prompt,
                                                temperature=0,
                                                top_p=1,
                                                max_tokens=60,
                                                frequency_penalty=0,
                                                presence_penalty=0)
        
        # Print the result
        res = response["choices"][0]["text"]
        st.info(res)

except Exception as error: 
    st.warning(error)

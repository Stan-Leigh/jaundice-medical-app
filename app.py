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

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
client = Client(scope=scope, creds=credentials)
spreadsheetname = "Database"
spread = Spread(spreadsheetname, client=client)

## Check that the connection worked
# st.write(spread.url)

# Call our spreadsheet
sh = client.open(spreadsheetname)
worksheet_list = sh.worksheets()

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
    return x.split()[0]


columns = ['users', 'sbt', 'ut', 'sc', 'dwf', 'hpc']
df = pd.DataFrame(columns=columns)
# Create instances of the worksheets
sheet_names = worksheet_names()
df1 = load_the_spreadsheet(sheet_names[0])
df2 = load_the_spreadsheet(sheet_names[1])

# get users' usernames from the existing dataframe
usernames = set(df1.users.apply(get_username))

# Get usernames as unique identifiers
now = datetime.now()
new = st.radio("First time visiting this app?", ("", "No", "Yes"))

if new == "No":
    user = st.text_input("Enter your username to retrieve your information.")
    if user in usernames:
        username = user + " " + str(now)
        st.info("You can proceed")
    elif len(user) > 1 and user not in usernames:
        st.info("This username is not registered. Check spellings and capitalizations.", icon="🚨")
elif new == "Yes":
    user = st.text_input("Enter a unique username to track your information. There should be no spaces in the username.")
    if len(user.split()) > 1:
        st.info("There should be no spaces in the username", icon="🚨")
    elif user in usernames or len(user) == 1:
        st.info("This username already exists or the username is too short", icon="🚨")
    elif len(user) > 1:
        username = user + " " + str(now)
        st.info("You can proceed")
else:
    pass

# Streamlit code continues
st.header("Symptoms")
st.subheader("1. Skin Blanched test")
st.write("""
        Press gently on the baby's forehead or nose to blanch the skin (make it pale). 
        Release the pressure, and observe if the skin returns to its yellow color.
""")
sbt = st.radio("Does the skin return to its original color in less than two seconds?", ("Less than two seconds", "More than two seconds"))

st.subheader("2. Urine test")
st.write("""
        Try to compare the current diaper with previous ones to see if there is a significant change in urine color. 
        This can help you identify any sudden darkening of the urine.
         
        If you notice that your baby's urine is dark yellow or orange, 
        it could be an indication of increased bilirubin levels in the blood.
""")
urine_intensity = st.slider('What is the intensity of the color of the urine?', 0, 100, 0, 20)
hex_values = ['#fff064', '#ffde1a', '#ffce00', '#ffa700', '#ff8d00', '#ff7400']
color_hex = int(urine_intensity / 20)
color = st.color_picker('Color intensity', hex_values[color_hex])
ut = "No" if color_hex < 3 else "Yes"

st.subheader("3. Stool color")
st.write("""
        Observe your baby's stool color when they are healthy and not experiencing jaundice. 
        Note the typical color of their stools during this time.
         
        If you notice a significant change in stool color, 
        where the stool becomes considerably paler or takes on a clay-like appearance, it may be an indicator of jaundice.
""")
sc = st.radio("Is there significant change in stool color (Paler or clay-like appearance)?", ("No", "Yes"))

st.subheader("4. Difficulty in waking and feeding")
st.write("""
        Observe your baby and watch for signs of lethargy i.e. extreme tiredness, sluggishness, and lack of energy or motivation,
        drowsiness, and a reduced level of consciousness. The baby may appear weak, and less responsive than usual. 
         
        Feeding can become challenging because the baby lacks the energy and alertness to suckle or drink properly.
""")
dwf = st.radio("Are there signs that indicate that the baby has difficulty waking and feeding?", ("No", "Yes"))

st.subheader("5. High pitched crying")
st.write("""
        Observe your baby and listen for changes in their crying behaviour. Remember that babies cry for various reasons, 
        and it's a normal part of their communication. They might cry due to hunger, discomfort, needing a diaper change, 
        fatigue, or feeling overwhelmed.
         
        Pay attention and see if you can detect any change in their behaviour or if the cries are high pitched.
""")
hpc = st.radio("Are there changes in the crying behaviour or is the baby's cries high pitched?", ("No", "Yes"))


# Sidebar menu
# Show the registered usernames
st.sidebar.selectbox('Registered Usernames', usernames)


st.sidebar.write("## RECOMMENDATION")
good = "All good. No signs of jaundice."
medium = "Watch your baby for more symptoms. No serious issue yet."
bad = "See a medical professional immediately! Your baby might be suffering from jaundice."

# Logic
def recommendation(sbt, ut, sc, dwf, hpc):
    strike = 0

    if sbt == "More than two seconds":
        strike += 1
    if ut == "Yes":
        strike += 1 
    if sc == "Yes":
        strike += 1 
    if dwf == "Yes":
        strike += 1  
    if hpc == "Yes":
        strike += 1  

    if strike > 2:
        st.sidebar.write(bad)
    elif strike > 0:
        st.sidebar.write(medium)
    else:
        st.sidebar.write(good) 


# Create button to confirm write operation to the dataframe
if st.button("Press this button to confirm all answers"):
    df.loc[len(df)] = [username, sbt, ut, sc, dwf, hpc]

    # combine old records with new ones
    df = pd.concat([df1, df], ignore_index=True)

    # update the googlesheet
    update_the_spreadsheet("JaundiceReport", df)
    
    # Show recommendation on the sidebar
    recommendation(sbt, ut, sc, dwf, hpc)
    st.write(df[df['users'].str.contains(user)])


# Computer vision to detect Jaundice on the skin
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

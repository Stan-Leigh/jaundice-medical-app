import streamlit as st
import openai
import pandas as pd
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
ut = st.radio("Is the baby's urine color dark yellow or orange?", ("No", "Yes"))

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
# Logic for everything
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
        st.sidebar.write("See a medical professional immediately! Your baby might be suffering from jaundice.")
    elif strike > 0:
        st.sidebar.write("Watch your baby for more symptoms. No serious issue yet.")
    else:
        st.sidebar.write("All good. No signs of jaundice.") 


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

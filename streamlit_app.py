
import pvlib
from pvlib import location
from pvlib import irradiance
import pandas as pd
from matplotlib import pyplot as plt
import math
import vocmax
import time
import numpy as np


from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import streamlit as st
import pandas as pd
import numpy as np

st.title('Technical and Financial Model of a PV system')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
            
lat = st.number_input('Insert the latitude')
#st.write('The current number is ', lat)
lon= st.number_input('Insert the longitude')
#st.write('The current number is ', lon)

NREL_API_KEY = 'Hmf84x6KdrFhF4FGdtH8MRD2bWObpR7YYUvwhgd3'  # <-- please set your NREL API key here
NREL_API_KEY = st.text_input('NREL_API_KEY', 'Hmf84x6KdrFhF4FGdtH8MRD2bWObpR7YYUvwhgd3')
#st.write('The NREL_API_KEY is', NREL_API_KEY)

email='atiqureee@gmail.com'
email = st.text_input('email', 'atiqureee@gmail.com')
#st.write('The email is', email)


weather, metadata=pvlib.iotools.get_psm3(lat, lon, NREL_API_KEY, email=email, names='2010', interval=60, attributes=('air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 'surface_albedo', 'surface_pressure', 'wind_direction', 'wind_speed'), leap_day=False, full_name='pvlib python', affiliation='pvlib python', map_variables=None, timeout=30)






st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)

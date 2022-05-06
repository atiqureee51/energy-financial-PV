
import pvlib
from pvlib import location
from pvlib import irradiance
import pandas as pd
import matplotlib.pyplot as plt
import math
#import vocmax
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
lat=29.99
#st.write('The current number is ', lat)
lon= st.number_input('Insert the longitude')
#st.write('The current number is ', lon)
lon=-91.89

NREL_API_KEY2 = 'Hmf84x6KdrFhF4FGdtH8MRD2bWObpR7YYUvwhgd3'  # <-- please set your NREL API key here
NREL_API_KEY='lOHyZqGMvZfGXResGGPtdvceWLyUnCtabZ1Ngbkt'
#NREL_API_KEY = st.text_input('NREL_API_KEY', 'Hmf84x6KdrFhF4FGdtH8MRD2bWObpR7YYUvwhgd3')
#st.write('The NREL_API_KEY is', NREL_API_KEY)

email1='atiqureee@gmail.com'
email='atiqureee111@gmail.com'
#email = st.text_input('email', 'atiqureee@gmail.com')
#st.write('The email is', email)


weather, metadata=pvlib.iotools.get_psm3(lat, lon, NREL_API_KEY, email=email, names='2010', interval=60, attributes=('air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 'surface_albedo', 'surface_pressure', 'wind_direction', 'wind_speed'), leap_day=False, full_name='pvlib python', affiliation='pvlib python', map_variables=None, timeout=30)

## 3. Enter the following financial information:
installed_cost=4328468.17    #($/kWdc)
federal_tax_credit=0.00*installed_cost  #(% of installed cost) 30
state_tax_credit=0.00*installed_cost   #(% of installed cost) 10
rebates= 26          #(itc OR Investment tax credit % of installed cost)
interest_rate=4       #(annual percentage rate)
project_life=25       #(years)
annual_maintenance_costs= 15000         #($)
years_to_inverter_replacement=10   #(years)
inverter_cost=10000          #($/kWac)
salvage_value=0.05*installed_cost          #(%)
variable_operating_costs =0.0     #($/kWh) 0.1
electricity_rate=0.10  #$/kWh      # $/kWh    0r 100 in $/MWh retail rate or PPA price ($/kWh)
term_of_loan=25   # years
interest_rate= 4  #(annual percentage rate)
amount_financed=100  ##(%)
annual_payments=3  #(%) 
discount_rate=interest_rate  
year=project_life
voc=variable_operating_costs   ##$/kWh
foc=annual_maintenance_costs
salvage_cost=salvage_value

#print('installed_cost=',installed_cost)

total_loss={'soiling':2, 'shading':0, 'snow':0, 'mismatch':2, 'wiring':2, 'connections':0, 'lid':0, 'nameplate_rating':0, 'age':0, 'availability':0}
mod=pvlib.pvsystem.retrieve_sam('SandiaMod')
#mod.to_csv('/content/module.csv') 
module=mod.SunPower_SPR_300_WHT__2007__E__.to_dict()
#

# 6. Select an inverter from the SAM database
#invdb=pvlib.pvsystem.retrieve_sam('SandiaInverter')
invdb=pvlib.pvsystem.retrieve_sam('SandiaInverter')
#invdb.to_csv('/content/inverter.csv') 
inverter=invdb.Huawei_Technologies_Co___Ltd___SUN2000_33KTL_US__480V_.to_dict()


max_string_design_voltage = inverter['Vdcmax']
min_db_temp_ashrae=-3.7     #ASHRAE_Extreme_Annual_Mean_Minimum_Design_Dry_Bulb Temperature (Tmin)
max_db_temp_ashrae= 36.6    #ASHRAE 2% Annual Design Dry Bulb Temperature (Tmax)#

module['Bvoco%/C']=(module['Bvoco']/module['Voco'])*100
module['Bvmpo%/C']=(module['Bvmpo']/module['Vmpo'])*100
module['Aimpo%/C']=(module['Aimp']/module['Impo'])*100
module['TPmpo%/C']=module['Bvmpo%/C']+module['Aimpo%/C']
max_module_voc= module['Voco']*(1+((min_db_temp_ashrae-25)*module['Bvoco%/C']/100))  #Temperature corrected maximum module Voc
max_module_series=int(max_string_design_voltage/max_module_voc) #maximum number of modules in series



dc_ac_ratio=inverter['Pdco']/inverter['Paco']
inverter_STC_watts=inverter['Paco']*dc_ac_ratio
single_module_power=module['Vmpo']*module['Impo']
T_add= 25 # temp adder
min_module_vmp= module['Vmpo']*(1+((T_add+max_db_temp_ashrae-25)*module['TPmpo%/C']/100))  #Temperature corrected maximum module Voc
min_module_series_ideal=math.ceil(inverter['Mppt_low']*1.2/min_module_vmp) #maximum number of modules in series
min_module_series_okay=math.ceil(inverter['Mppt_low']*dc_ac_ratio/min_module_vmp) #maximum number of modules in series



source_circuit_STC_power=[]
parallel_string=[]
circuit_dc_ac_ratio=[]
diff_ratio=[]
series_module=[]
for i in range(min_module_series_okay,max_module_series+1,1):
  #print(i)
  source_circuit=single_module_power*i
  max_string=int(inverter['Pdco']/source_circuit)
  ratio=max_string*single_module_power*i/inverter['Paco']
  diff_r=abs(dc_ac_ratio-ratio)
  diff_ratio.append(diff_r)
  circuit_dc_ac_ratio.append(ratio)
  parallel_string.append(max_string)
  source_circuit_STC_power.append(source_circuit)
  series_module.append(i)
min_index_dc_ac_ratio = diff_ratio.index(min(diff_ratio))
print(min_index_dc_ac_ratio)
max_parallel_string=parallel_string[min_index_dc_ac_ratio]
no_of_series_module=series_module[min_index_dc_ac_ratio]


#print(source_circuit_STC_power)
#print(parallel_string)
#print(circuit_dc_ac_ratio)
#print('max_parallel_string:',max_parallel_string)
#print('no_of_series_module',no_of_series_module)

st.write('max_parallel_string:',max_parallel_string)
st.write('no_of_series_module',no_of_series_module)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)

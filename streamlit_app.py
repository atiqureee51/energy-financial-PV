# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:07:05 2022

@author: Atiqur rahaman
"""


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

#st.write('installed_cost=',installed_cost)

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
  #st.write(i)
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
st.write(min_index_dc_ac_ratio)
max_parallel_string=parallel_string[min_index_dc_ac_ratio]
no_of_series_module=series_module[min_index_dc_ac_ratio]


#st.write(source_circuit_STC_power)
#st.write(parallel_string)
#st.write(circuit_dc_ac_ratio)
#st.write('max_parallel_string:',max_parallel_string)
#st.write('no_of_series_module',no_of_series_module)

st.write('max_parallel_string:',max_parallel_string)
st.write('no_of_series_module',no_of_series_module)




# b. Calculate number of inverters needed to meet annual energy production goal

number_of_inverters_needed=math.ceil(system_size*1E6/inverter['Pdco'])
st.write('number_of_inverters_needed:',number_of_inverters_needed)

# 8. Calculate the total AC and DC system size, and DC/AC ratio
total_AC_system_size=inverter['Paco']*number_of_inverters_needed
total_DC_system_size=inverter['Pdco']*number_of_inverters_needed
dc_ac_ratio=total_DC_system_size/total_AC_system_size
st.write('dc_ac_ratio:',dc_ac_ratio)



##9. Enter the type of racking to be used (i.e. fixed tilt, single-axis tracking, etc.)
#   a. Enter azimuth and tilt angle for array
#   b. Enter Ground Coverage Ratio or distance between rows

#a. Enter azimuth and tilt angle for array
racking_parameters = {
    'racking_type': 'fixed_tilt',
    'surface_tilt': 30,
    'surface_azimuth': 180,
    'albedo':0.2,
    'gcr': 0.60 
}


##10. Calculate the following performance information

#a. Annual energy production

#calculates the solar data (zenith,azimuth,eot,elevation,etc)
sol_data = pvlib.solarposition.get_solarposition(time=weather.index,latitude=lat,longitude=lon,altitude=alt,temperature=weather['Temperature'])
sol_data['dni_extra']=pvlib.irradiance.get_extra_radiation(weather.index)

#we can confirm the tz is correctly localized by looking at the solar elevation values
sol_data.head(5)
#calculate environmental data (poa components, aoi, airmass)
env_data = pvlib.irradiance.get_total_irradiance(surface_tilt=racking_parameters['surface_tilt'], surface_azimuth=racking_parameters['surface_azimuth'], 
                                                 solar_zenith=sol_data['apparent_zenith'],
                                                 solar_azimuth=sol_data['azimuth'], dni=weather['DNI'],ghi=weather['GHI'],
                                                 dhi=weather['DHI'], dni_extra=sol_data['dni_extra'], model='haydavies')
env_data['aoi'] = pvlib.irradiance.aoi(surface_tilt=racking_parameters['surface_tilt'], surface_azimuth=racking_parameters['surface_azimuth'], solar_zenith=sol_data['apparent_zenith'],
                                       solar_azimuth=sol_data['azimuth'])
env_data['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith=sol_data['apparent_zenith'])
env_data['am_abs'] = pvlib.atmosphere.get_absolute_airmass(airmass_relative=env_data['airmass'], pressure=(weather['Pressure']*100))
env_data.head(5)
#calculate the effective irradiance - needs module so that it can have fraction of diffuse irradiance used by the module
weather['effective_irradiance'] = pvlib.pvsystem.sapm_effective_irradiance(poa_direct=env_data['poa_direct'],poa_diffuse=env_data['poa_diffuse'],
                                                   airmass_absolute=env_data['am_abs'], aoi=env_data['aoi'], module=module)

tmp=(pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'])
weather['cell_temperature'] = pvlib.temperature.sapm_cell(poa_global=env_data['poa_global'], temp_air=weather['Temperature'],
                                                   wind_speed=weather['Wind Speed'], a=tmp['a'], b=tmp['b'], deltaT=tmp['deltaT'])


weather['Solar Elevation'] = sol_data['apparent_elevation']
weather.replace(0, np.nan, inplace=True)
#weather.replace(np.nan,0, inplace=True)
#day_weather = weather.loc[(weather['Solar Elevation'] > 0) & (weather['Solar Elevation'] < 90)] 



temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']


#https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.PVSystem.html#pvlib.pvsystem.PVSystem
system=pvlib.pvsystem.PVSystem(arrays=None, surface_tilt=racking_parameters['surface_tilt'], surface_azimuth=racking_parameters['surface_azimuth'], 
                        albedo=0.2, surface_type=None, module='SunPower_SPR_300_WHT__2007__E__', 
                        module_type=None, module_parameters=module, 
                        temperature_model_parameters=temperature_model_parameters, modules_per_string=no_of_series_module, 
                        strings_per_inverter=max_parallel_string, inverter='Huawei_Technologies_Co___Ltd___SUN2000_33KTL_US__480V_', inverter_parameters=inverter, 
                        racking_model='open_rack', losses_parameters=total_loss , name=None)



#https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.run_model_from_effective_irradiance.html#pvlib.modelchain.ModelChain.run_model_from_effective_irradiance
mc = ModelChain(system, location,losses_model='pvwatts')
#mc.run_model(weather)
mc.run_model_from_effective_irradiance(weather)
mc.results.aoi
mc.results.cell_temperature
mc.results.dc.to_dict()
ac_result=mc.results.ac.to_dict()
#ac_result.loc[ac_result<0]=0
#ac_result.loc[ac_result<0]
ac_result
energy_production=(pd.DataFrame.from_dict(ac_result, orient='index', columns=['energy_production'])*number_of_inverters_needed)/1000  ##in kW
energy_production.replace( np.nan, 0, inplace=True)
energy_month=energy_production.resample('M').sum()*(3600/3600)   # kWh
energy_month
#annual_energy_production=energy_month.resample('Y').sum()*(1-final_loss/100)   # kWh
annual_energy_production=energy_month.resample('Y').sum()  # MWh
annual_energy_production
st.write('annual_energy_production in kWh',annual_energy_production.to_numpy())












st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)

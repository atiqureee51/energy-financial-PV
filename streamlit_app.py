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

import time
import numpy as np


from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import streamlit as st
import pandas as pd
import numpy as np

from distutils import errors
from distutils.log import error
import altair as alt
from itertools import cycle

#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode



st.title('Technical and Financial Model of a PV system')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

#@st.cache(allow_output_mutation=True)


def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

#data_load_state = st.text('Loading data...')
data = load_data(10000)
#data_load_state.text("Done! (using st.cache)")

#if st.checkbox('Show raw data'):
#    st.subheader('Raw data')
#    st.write(data)


#st.subheader('Raw data')
## 1. Enter the location of the PV system and obtain a weather file for that location

st.subheader('Enter the location of the PV system and obtain techno-economic analysis for that location')

lat = st.sidebar.number_input('Insert the latitude, example =29.99 ',value=29.99,)
#lat=29.99
#st.write('The current number is ', lat)
lon=st.sidebar.number_input('Insert the longitude, example = -91.89',value=-91.89)
#st.write('The current number is ', lon)
#lon=-91.89

alt=st.sidebar.number_input('Insert the altitude, example =13 ',value=13)
## 2. Enter the approximate desired annual energy production (kWh/year)
system_size= 5 # in MW in dc


email1='atiqureee@gmail.com'
#email='atiqureee111@gmail.com'
email = st.sidebar.text_input('email for the NREL API KEY', 'atiqureee@gmail.com')
#st.write('The email is', email)


#NREL_API_KEY2 = 'Hmf84x6KdrFhF4FGdtH8MRD2bWObpR7YYUvwhgd3'  # <-- please set your NREL API key here
NREL_API_KEY='qguVH9fdgUOyRo1jo6zzOUXkS6a96vY1ct45RpuK'  
NREL_API_KEY = st.sidebar.text_input('Go to https://developer.nrel.gov/signup/ to get the NREL_API_KEY', 'qguVH9fdgUOyRo1jo6zzOUXkS6a96vY1ct45RpuK')
#st.write('The NREL_API_KEY is', NREL_API_KEY)




weather, metadata=pvlib.iotools.get_psm3(lat, lon, NREL_API_KEY, email, names='2010', interval=60, attributes=('air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 'surface_albedo', 'surface_pressure', 'wind_direction', 'wind_speed'), leap_day=False, full_name='pvlib python', affiliation='pvlib python', map_variables=None, timeout=30)

if st.checkbox('Show raw weather data'):
    st.subheader('Raw downloaded weather data')
    st.write(weather)
# Setting up columns

#c1,c2= st.columns([1,1])

# Widgets: checkbox (you can replace st.xx with st.sidebar.xx)
#if c2.checkbox("Show Dataframe"):
#    st.subheader("The weather dataset:")
#    st.dataframe(data=weather)
    #st.table(data=weather)

#weather.to_csv('/data/weather.csv')
#c1.download_button("Download CSV File", data='/data/weather.csv', file_name="weather.csv", mime='text/csv')

##chart
chart_data = pd.DataFrame(weather["GHI"],weather["Temperature"],weather["Wind Speed"])

st.line_chart(weather["GHI"])

##map
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
from plotly.subplots import make_subplots


df_map = pd.DataFrame({'lat': [lat],'lon': [lon]})
#df_map["lat"]=lat
#df_map["lon"]=lon

st.map(df_map)






## 3. Enter the following financial information:
installed_cost=st.number_input('Insert the installed_cost in $ ',value=4328468.17 )  
federal_tax_credit_percent=st.number_input('Insert the federal_tax_credit in % ',value=0 )
federal_tax_credit=federal_tax_credit_percent/100*installed_cost  #(% of installed cost) 30
state_tax_credit_percent=st.number_input('Insert the state_tax_credit in % ',value=0 )
state_tax_credit=state_tax_credit_percent/100*installed_cost   #(% of installed cost) 10
rebates_percent=st.number_input('Insert rebate in % ',value=0 )
rebates= rebates_percent/100          #(itc OR Investment tax credit % of installed cost)

interest_rate=st.number_input('Insert the interest in % ',value=4 )/100       #(annual percentage rate)
project_life=st.number_input('Insert the project life in years ',value=25 )       #(years)
annual_maintenance_costs= st.number_input('Insert the annual maintenace cost in $ ',value=15000 )         #($)
years_to_inverter_replacement=st.number_input('Insert the years to replace inverters ',value=10 )   #(years)
inverter_cost=st.number_input('Insert the inverter cost in $ ',value=10000 )           #($/kWac)
salvage_value=(st.number_input('Insert the salvage value in % of installed cost ',value=5 )/100)*installed_cost          #(%)
variable_operating_costs =st.number_input('Insert the variable operating cost in $/kWh ',value=0 )     #($/kWh) 0.1
electricity_rate=st.number_input('Insert the electricity rate in $/kWh ',value=0.1 )  #$/kWh      # $/kWh    0r 100 in $/MWh retail rate or PPA price ($/kWh)
term_of_loan=st.number_input('Insert the loan term in years ',value=25 )  # years
amount_financed=st.number_input('Insert the amount financed in % ',value=100 )/100  ##(%)
annual_payments=st.number_input('Insert the number of annual payment ',value=3 )  #(%) 
discount_rate=interest_rate  
year=project_life
voc=variable_operating_costs   ##$/kWh
foc=annual_maintenance_costs
salvage_cost=salvage_value

#st.write('installed_cost=',installed_cost)

total_loss={'soiling':2, 'shading':0, 'snow':0, 'mismatch':2, 'wiring':2, 'connections':0, 'lid':0, 'nameplate_rating':0, 'age':0, 'availability':0}

#import st_aggrid
#from st_aggrid import AgGrid
#st.write('losses values in percentage')
#lossdataframe = pd.DataFrame({'soiling': [2], 'shading':[0], 'snow':[0], 'mismatch':[2], 'wiring':[2], 'connections':[0], 'lid':[0], 'nameplate_rating':[0], 'age':[0], 'availability':[0]})
#grid_return = AgGrid(lossdataframe, editable=True)
#new_df = grid_return['data']


#st.dataframe(new_df) 

## mod selection
mod=pvlib.pvsystem.retrieve_sam('SandiaMod')

module=mod.SunPower_SPR_300_WHT__2007__E__.to_dict()
clist = mod.T
clist.reset_index(inplace=True)
select_module = st.sidebar.selectbox("Select a module:",clist, index=467)
st.write('module list', clist)

select_module2=clist[clist['index'] == select_module]

index_value=clist[clist['index'] == select_module].index.tolist()
index_mod = np.asarray(index_value)
index_mod=index_mod[0]
st.write('index',index_mod)

module=select_module2.T.to_dict()
st.write('module',module)

module=module[index_mod]



# 6. Select an inverter from the SAM database

invdb=pvlib.pvsystem.retrieve_sam('SandiaInverter')
 
inverter=invdb.Huawei_Technologies_Co___Ltd___SUN2000_33KTL_US__480V_.to_dict()
clist = invdb.T
clist.reset_index(inplace=True)
select_inverter = st.sidebar.selectbox("Select a inverter:",clist, index=1337)
st.write('inverter list', clist)

select_inverter2=clist[clist['index'] == select_inverter]
index_value=clist[clist['index'] == select_inverter].index.tolist()
index_inv = np.asarray(index_value)
index_inv=index_inv[0]
st.write('index',index_inv)


inverter=select_inverter2.T.to_dict()
st.write('inverter',inverter)


inverter=inverter[index_inv]




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
st.write('min_module_series_okay',min_module_series_okay)

st.write('max_module_series',max_module_series)

source_circuit_STC_power=[]
parallel_string=[]
circuit_dc_ac_ratio=[]
diff_ratio=[]
series_module=[]

if max_module_series>=min_module_series_okay:
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
else:
            i=min_module_series_okay
            source_circuit=single_module_power*i
            max_string=int(inverter['Pdco']/source_circuit)
            ratio=max_string*single_module_power*i/inverter['Paco']
            #diff_r=abs(dc_ac_ratio-ratio)
            min_index_dc_ac_ratio = ratio
            st.write(min_index_dc_ac_ratio)
            max_parallel_string=max_string
            no_of_series_module=min_module_series_okay 
            if max_string==0:
                      max_parallel_string=1  



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
                        albedo=0.2, surface_type=None, module=None, 
                        module_type=None, module_parameters=module, 
                        temperature_model_parameters=temperature_model_parameters, modules_per_string=no_of_series_module, 
                        strings_per_inverter=max_parallel_string, inverter=None, inverter_parameters=inverter, 
                        racking_model='open_rack', losses_parameters=total_loss , name=None)



#https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.run_model_from_effective_irradiance.html#pvlib.modelchain.ModelChain.run_model_from_effective_irradiance
mc = ModelChain(system, location,losses_model='pvwatts')
#mc.run_model(weather)
mc.run_model_from_effective_irradiance(weather)
mc.results.aoi
st.write('cell temperature in C', mc.results.cell_temperature)
#mc.results.dc.to_dict()
ac_result=mc.results.ac.to_dict()
#ac_result.loc[ac_result<0]=0
#ac_result.loc[ac_result<0]
ac_result
energy_production=(pd.DataFrame.from_dict(ac_result, orient='index', columns=['energy_production in kWh'])*number_of_inverters_needed)/1000  ##in kW
energy_production.replace( np.nan, 0, inplace=True)
energy_month=energy_production.resample('M').sum()*(3600/3600)   # kWh
energy_month
#annual_energy_production=energy_month.resample('Y').sum()*(1-final_loss/100)   # kWh
annual_energy_production=energy_month.resample('Y').sum()  # MWh
annual_energy_production
st.write('annual_energy_production in kWh',annual_energy_production.to_numpy())







##b. Energy Yield
dc_result=mc.results.dc.p_mp.to_dict()
energy_production_dc=(pd.DataFrame.from_dict(dc_result, orient='index', columns=['energy_production_dc'])*number_of_inverters_needed)/1000 ##in kW
energy_production_dc.replace( np.nan, 0, inplace=True)
annual_energy_production_dc=energy_production_dc.resample('Y').sum()*(3600/3600)
Energy_Yield=((annual_energy_production_dc.values)/(single_module_power*max_parallel_string*no_of_series_module*number_of_inverters_needed))*1000
st.write('Energy_Yield in kWh/m2:',Energy_Yield)

## c. Final Yield
Final_Yield=((annual_energy_production.values)/(single_module_power*max_parallel_string*no_of_series_module*number_of_inverters_needed))*1000
st.write('Final_Yield in kWh/m2:',Final_Yield)


## d. Reference Yield
poa_wh_m2=(env_data['poa_global']*(3600/3600))  ##wh/m2
poa_sum=poa_wh_m2.resample('Y').sum().to_numpy()
Reference_Yield=poa_sum/1000 ##kwh/m2
st.write('Reference_Yield kWh/m2',Reference_Yield)
#st.write('poa sum kWh/m2',poa_sum)

## e. Performance Ratio
Performance_Ratio=Final_Yield/Reference_Yield
st.write('Performance_Ratio in %',Performance_Ratio*100)


##f. Capacity Factor
Capacity_Factor=((annual_energy_production.values)/(single_module_power*max_parallel_string*no_of_series_module*number_of_inverters_needed*8760))*1000
st.write('Capacity Factor in %',Capacity_Factor*100)

## g. System Efficiency
System_Efficiency=((annual_energy_production.values)/(poa_sum*max_parallel_string*no_of_series_module*number_of_inverters_needed*module['Area']))*1000
st.write('System_Efficiency in %',System_Efficiency*100)


##11. Calculate the following financial information
## a. Estimated installed cost (before credits and rebates)


#st.write('installed_cost',installed_cost)
# b. Total Capital Cost

Total_Capital_Cost=installed_cost-federal_tax_credit-state_tax_credit

st.write('Total_Capital_Cost in $',Total_Capital_Cost)

# c. Net annual savings




energy_cost_first_year=annual_energy_production.values*electricity_rate
RPWF=(1-((1+discount_rate)**(-year)))/discount_rate
single_pwf=(1/((1+discount_rate)**year))
Utility_energy_cost_LCC=energy_cost_first_year*RPWF

st.write('Utility_energy_cost_LCC in $',Utility_energy_cost_LCC)

maintenance_value=annual_maintenance_costs*RPWF
#st.write('maintenance_costs_value',maintenance_value)
salvage_value=salvage_cost*single_pwf
#st.write('salvage_value',salvage_value)


PV_Life_cycle_cost=Total_Capital_Cost+maintenance_value-salvage_value
st.write('PV_Life_cycle_cost in $',PV_Life_cycle_cost)

Net_LCC_cost=PV_Life_cycle_cost-Utility_energy_cost_LCC
st.write('Net_LCC_cost in $',Net_LCC_cost)


Net_annual_savings=energy_cost_first_year
st.write('Net_annual_savings in $',Net_annual_savings)

##e. LCOE
fixed_charge_rate=1/RPWF


LCOE=(((Total_Capital_Cost*fixed_charge_rate)+foc)/annual_energy_production.values)+voc
st.write('LCOE in $/kWh',LCOE)
## d. Simple Payback Period

Simple_Payback_Period=Total_Capital_Cost/(annual_energy_production.values*electricity_rate)
st.write('Simple_Payback_Period in years',Simple_Payback_Period)


#f. Net Present Value of Project
#Net_Present_Value_of_Project=Total_Capital_Cost*single_pwf
#st.write('Net_Present_Value_of_Project',Net_Present_Value_of_Project)

## Python program explaining pv() function
import numpy_financial as npf

#            rate            values     
a =  npf.npv(discount_rate,[-Total_Capital_Cost, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate, 
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate, 
                           annual_energy_production.values*electricity_rate, 
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate, 
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate, 
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate, annual_energy_production.values*electricity_rate,
                           annual_energy_production.values*electricity_rate])
st.write("Net Present Value(npv) in $: ", a)



#energy_month=energy_month.T

st.subheader('monthly production')
#hist_values1 = np.histogram(energy_month, bins=12, range=(0,12))
st.bar_chart(energy_month,width=4)



#st.subheader('Number of pickups by hour')
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#st.bar_chart(hist_values)

# Some number in the range 0-23
#hour_to_filter = st.slider('hour', 0, 23, 17)
#filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

#st.subheader('Map of all pickups at %s:00' % hour_to_filter)
#st.map(filtered_data)

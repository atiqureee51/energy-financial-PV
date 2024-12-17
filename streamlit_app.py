# -*- coding: utf-8 -*-
"""
Enhanced Streamlit App for PV Techno-Economic Analysis
Now with configurable temperature model selection:
 - Choose system type (ground, roof, floating, agrivoltaics)
 - Choose temperature model family (sapm, pvsyst, custom)
 - Choose from predefined models or input custom a, b, deltaT
"""

import streamlit as st
import pandas as pd
import numpy as np
import pvlib
from pvlib import location
from pvlib import irradiance
import math
import requests
import io
from json import JSONDecodeError
import warnings
import numpy_financial as npf
import altair as alt
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import Polygon
from pyproj import Transformer

st.set_page_config(page_title="PV Techno-Economic Analysis", layout='wide')

# Pre-defined Temperature Models from pvlib
TEMPERATURE_MODEL_PARAMETERS = {
    'sapm': {
        'open_rack_glass_glass': {'a': -3.47, 'b': -0.0594, 'deltaT': 3},
        'close_mount_glass_glass': {'a': -2.98, 'b': -0.0471, 'deltaT': 1},
        'open_rack_glass_polymer': {'a': -3.56, 'b': -0.0750, 'deltaT': 3},
        'insulated_back_glass_polymer': {'a': -2.81, 'b': -0.0455, 'deltaT': 0},
    },
    'pvsyst': {
        'freestanding': {'u_c': 29.0, 'u_v': 0.0},
        'insulated': {'u_c': 15.0, 'u_v': 0.0}
    }
}

currency_conversion = {"USD": 1, "BDT": 110}  # Approximate rate

def_elec_rate_bd = 0.08
def_elec_rate_us = 0.12
def_elec_rate_global = 0.10

def get_psm_url(lon):
    NSRDB_API_BASE = "https://developer.nrel.gov"
    PSM_URL1 = NSRDB_API_BASE + "/api/nsrdb/v2/solar/psm3-download.csv"
    MSG_URL = NSRDB_API_BASE + "/api/nsrdb/v2/solar/msg-iodc-download.csv"
    HIMAWARI_URL = NSRDB_API_BASE + "/api/nsrdb/v2/solar/himawari-download.csv"

    if -16 < lon < 91:
        return MSG_URL
    elif 91 <= lon < 182:
        return HIMAWARI_URL
    else:
        return PSM_URL1

def parse_psm3(fbuf, map_variables=False):
    metadata_fields = fbuf.readline().split(',')
    metadata_fields[-1] = metadata_fields[-1].strip()
    metadata_values = fbuf.readline().split(',')
    metadata_values[-1] = metadata_values[-1].strip()
    metadata = dict(zip(metadata_fields, metadata_values))
    metadata['Local Time Zone'] = int(metadata['Local Time Zone'])
    metadata['Time Zone'] = int(metadata['Time Zone'])
    metadata['Latitude'] = float(metadata['Latitude'])
    metadata['Longitude'] = float(metadata['Longitude'])
    metadata['Elevation'] = int(metadata['Elevation'])
    columns = fbuf.readline().split(',')
    columns[-1] = columns[-1].strip()
    columns = [col for col in columns if col != '']
    dtypes = dict.fromkeys(columns, float)
    dtypes.update(Year=int, Month=int, Day=int, Hour=int, Minute=int)
    dtypes['Cloud Type'] = int
    dtypes['Fill Flag'] = int
    data = pd.read_csv(
        fbuf, header=None, names=columns, usecols=columns, dtype=dtypes,
        delimiter=',', lineterminator='\n')
    dtidx = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    tz = 'Etc/GMT%+d' % -metadata['Time Zone']
    data.index = pd.DatetimeIndex(dtidx).tz_localize(tz)

    if map_variables:
        VARIABLE_MAP = {
            'GHI': 'ghi',
            'DHI': 'dhi',
            'DNI': 'dni',
            'Clearsky GHI': 'ghi_clear',
            'Clearsky DHI': 'dhi_clear',
            'Clearsky DNI': 'dni_clear',
            'Solar Zenith Angle': 'solar_zenith',
            'Temperature': 'temp_air',
            'Relative Humidity': 'relative_humidity',
            'Dew point': 'temp_dew',
            'Pressure': 'pressure',
            'Wind Direction': 'wind_direction',
            'Wind Speed': 'wind_speed',
            'Surface Albedo': 'albedo',
            'Precipitable Water': 'precipitable_water',
        }
        data = data.rename(columns=VARIABLE_MAP)
        metadata['latitude'] = metadata.pop('Latitude')
        metadata['longitude'] = metadata.pop('Longitude')
        metadata['altitude'] = metadata.pop('Elevation')

    return data, metadata

def get_psm3_data(latitude, longitude, api_key, email, names='tmy', interval=60,
                  attributes=('air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 'surface_albedo',
                              'surface_pressure', 'wind_direction', 'wind_speed'),
                  leap_day=False, full_name='pvlib python',
                  affiliation='pvlib python', timeout=30):
    longitude_str = ('%9.4f' % longitude).strip()
    latitude_str = ('%8.4f' % latitude).strip()
    params = {
        'api_key': api_key,
        'full_name': full_name,
        'email': email,
        'affiliation': affiliation,
        'reason': 'pvlib python',
        'mailing_list': 'false',
        'wkt': f'POINT({longitude_str} {latitude_str})',
        'names': names,
        'attributes': ','.join(attributes),
        'leap_day': str(leap_day).lower(),
        'utc': 'false',
        'interval': interval
    }

    if any(prefix in names for prefix in ('tmy', 'tgy', 'tdy')):
        URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-tmy-download.csv"
    elif interval in (5,15):
        URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-5min-download.csv"
    else:
        URL = get_psm_url(float(longitude))

    response = requests.get(URL, params=params, timeout=timeout)
    if not response.ok:
        try:
            errors = response.json()['errors']
        except JSONDecodeError:
            errors = response.content.decode('utf-8')
        raise requests.HTTPError(errors, response=response)

    fbuf = io.StringIO(response.content.decode('utf-8'))
    data, metadata = parse_psm3(fbuf, map_variables=True)
    return data, metadata

def compute_area_of_polygon(latlon_list):
    if len(latlon_list) < 3:
        return 0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    coords_3857 = [transformer.transform(pt[1], pt[0]) for pt in latlon_list]
    polygon_3857 = Polygon(coords_3857)
    return polygon_3857.area

def generate_kpi_metrics(annual_energy_kwh, electricity_rate, total_capital_cost, simple_payback):
    total_annual_savings = annual_energy_kwh * electricity_rate
    return total_annual_savings, simple_payback, total_capital_cost

def get_default_electricity_rate(lat, lon):
    # Bangladesh approx lat:20.7 to 26.6, lon:88.0 to 92.7
    if 20.7 <= lat <= 26.6 and 88.0 <= lon <= 92.7:
        return def_elec_rate_bd
    # USA approx lat:24.5 to 49.3, lon:-124.8 to -66.9
    if 24.5 <= lat <= 49.3 and -124.8 <= lon <= -66.9:
        return def_elec_rate_us
    return def_elec_rate_global

# ----------------------------------------------
# Sidebar Inputs
# ----------------------------------------------
st.sidebar.title("User Inputs")

region = st.sidebar.radio("Select Region", ["Bangladesh", "USA"])
if region == "Bangladesh":
    default_lat = 23.8103
    default_lon = 90.4125
    default_currency = "BDT"
    default_installed_cost_usd = 7500.0
    default_elec_rate = def_elec_rate_bd
    lat = default_lat
    lon = default_lon
    currency = default_currency
else:
    default_lat = 30.2672
    default_lon = -97.7431
    default_currency = "USD"
    default_installed_cost_usd = 6000.0
    default_elec_rate = def_elec_rate_us
    lat = default_lat
    lon = default_lon
    currency = default_currency

cur_factor = currency_conversion[currency]

with st.sidebar.expander("Location and Weather Options"):
    lat = st.number_input('Latitude', value=lat, format="%.6f")
    lon = st.number_input('Longitude', value=lon, format="%.6f")
    alt = st.number_input('Altitude (m)', value=10)
    email = st.text_input('Email for NREL API Key', 'atiqureee@gmail.com')
    NREL_API_KEY = st.text_input('NREL API Key', 'qguVH9fdgUOyRo1jo6zzOUXkS6a96vY1ct45RpuK')

# System type: ground, roof, floating, agrivoltaics
system_type = st.sidebar.radio("System Type", ["Ground-mounted PV", "Roof-based PV", "Floating Solar", "Agrivoltaics"])

# Default model selection based on system_type
if system_type == "Ground-mounted PV":
    default_model_family = 'sapm'
    default_sapm_key = 'open_rack_glass_polymer'
elif system_type == "Roof-based PV":
    default_model_family = 'sapm'
    default_sapm_key = 'close_mount_glass_glass'
elif system_type == "Floating Solar":
    default_model_family = 'sapm'
    default_sapm_key = 'open_rack_glass_polymer'
else:  # Agrivoltaics
    default_model_family = 'sapm'
    default_sapm_key = 'open_rack_glass_polymer'

with st.sidebar.expander("System Configuration"):
    sizing_method = st.radio("Sizing Method", ["Manual System Size", "Area-based System Size"])
    manual_system_size_kw = st.number_input('System Size (kW DC)', value=5.0)
    packing_factor = st.number_input('Packing Factor (0-1)', value=0.8, min_value=0.0, max_value=1.0)
    mod_db = pvlib.pvsystem.retrieve_sam('SandiaMod')
    inv_db = pvlib.pvsystem.retrieve_sam('SandiaInverter')
    module_list = mod_db.columns.tolist()
    inverter_list = inv_db.columns.tolist()
    module_name = st.selectbox("Select Module", module_list, index=467)
    inverter_name = st.selectbox("Select Inverter", inverter_list, index=1337)

with st.sidebar.expander("Temperature Model"):
    # Choose model family: sapm, pvsyst, custom
    model_family = st.selectbox("Model Family", ["sapm", "pvsyst", "custom"], index=0 if default_model_family == 'sapm' else 2)
    
    if model_family == 'sapm':
        # show sapm keys + custom
        sapm_keys = list(TEMPERATURE_MODEL_PARAMETERS['sapm'].keys())
        sapm_keys.append("Custom SAPM")
        selected_sapm_key = st.selectbox("SAPM Model", sapm_keys, index=sapm_keys.index(default_sapm_key) if default_sapm_key in sapm_keys else (len(sapm_keys)-1))
        if selected_sapm_key == "Custom SAPM":
            a = st.number_input('a', value=-3.56)
            b = st.number_input('b', value=-0.075)
            deltaT = st.number_input('deltaT', value=3)
            temperature_model_parameters = {'a': a, 'b': b, 'deltaT': deltaT}
        else:
            temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm'][selected_sapm_key]

    elif model_family == 'pvsyst':
        # pvsyst keys
        pvsyst_keys = list(TEMPERATURE_MODEL_PARAMETERS['pvsyst'].keys())
        pvsyst_keys.append("Custom PVSyst")
        selected_pvsyst_key = st.selectbox("PVSyst Model", pvsyst_keys)
        if selected_pvsyst_key == "Custom PVSyst":
            u_c = st.number_input('u_c', value=29.0)
            u_v = st.number_input('u_v', value=0.0)
            # pvsyst model differs from sapm. We'll approximate a, b, deltaT for code compatibility:
            # Actually pvsyst uses u_c and u_v in a different formula.
            # For simplicity in code, we can store these separately and handle them later.
            # We'll store them in a dictionary and handle them carefully in code.
            temperature_model_parameters = {'u_c': u_c, 'u_v': u_v, 'model': 'pvsyst'}
        else:
            parameters = TEMPERATURE_MODEL_PARAMETERS['pvsyst'][selected_pvsyst_key]
            temperature_model_parameters = {'u_c': parameters['u_c'], 'u_v': parameters['u_v'], 'model': 'pvsyst'}

    else:
        # Custom model_family
        # Let user input a,b,deltaT as in SAPM style
        a = st.number_input('a', value=-3.56)
        b = st.number_input('b', value=-0.075)
        deltaT = st.number_input('deltaT', value=3)
        temperature_model_parameters = {'a': a, 'b': b, 'deltaT': deltaT, 'model': 'custom'}

with st.sidebar.expander("Financial Inputs"):
    installed_cost = st.number_input('Installed Cost', value=default_installed_cost_usd)*cur_factor
    federal_tax_credit_percent = st.number_input('Federal Tax Credit (%)', value=0.0)
    state_tax_credit_percent = st.number_input('State Tax Credit (%)', value=0.0)
    rebates_percent = st.number_input('Rebate (%)', value=0.0)
    interest_rate = st.number_input('Discount/Interest Rate (%)', value=4.0)/100
    project_life = st.number_input('Project Life (years)', value=25)
    annual_maintenance_costs = st.number_input('Annual Maintenance ($)', value=100.0)*cur_factor
    years_to_inverter_replacement = st.number_input('Years to Inverter Replacement', value=10)
    inverter_cost = st.number_input('Inverter Cost ($)', value=500.0)*cur_factor
    salvage_percent = st.number_input('Salvage Value (%) of installed cost', value=5.0)
    salvage_value = salvage_percent/100 * installed_cost
    voc = st.number_input('Variable Operating Cost ($/kWh)', value=0.0)*cur_factor

with st.sidebar.expander("Electricity Rate"):
    adjusted_default_rate = get_default_electricity_rate(lat, lon)*cur_factor
    electricity_rate = st.number_input('Electricity Rate ($/kWh)', value=adjusted_default_rate)

st.title("PV Techno-Economic Analysis Dashboard")

st.subheader("Select Location and Draw Area on Map")
m = folium.Map(location=[lat, lon], zoom_start=6, tiles=None)
folium.TileLayer('Esri.WorldImagery', name='Satellite', attr="Esri").add_to(m)
Draw(export=True, filename="data.json").add_to(m)
map_data = st_folium(m, width=700, height=500)

if 'last_clicked' in map_data and map_data['last_clicked'] is not None:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']

polygon_area = 0
possible_modules = 0
if 'all_drawings' in map_data and map_data['all_drawings'] is not None:
    for feature in map_data['all_drawings']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            latlon_list = [(c[1], c[0]) for c in coords]
            polygon_area = compute_area_of_polygon(latlon_list)

st.write(f"Chosen Location: Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt} m")
if polygon_area > 0:
    st.write(f"Drawn Polygon Area: {polygon_area:.2f} m²")

try:
    with st.spinner("Fetching weather data from NREL..."):
        weather, metadata = get_psm3_data(lat, lon, NREL_API_KEY, email, names="2019", interval=60)
except Exception as e:
    st.error(f"Error fetching weather data: {e}")
    st.stop()

module = mod_db[module_name].to_dict()
inverter = inv_db[inverter_name].to_dict()

if sizing_method == "Area-based System Size" and polygon_area > 0:
    module_area = module['Area']
    possible_modules = math.floor((polygon_area * packing_factor) / module_area)
    system_size_w = possible_modules * (module['Vmpo'] * module['Impo'])
    system_size_kw = system_size_w/1000
else:
    system_size_kw = manual_system_size_kw

system_size_mw = system_size_kw/1000

# Check if using pvsyst or sapm or custom
# If pvsyst, we need to convert to 'a','b','deltaT' equivalent or handle differently.
# For simplicity, if pvsyst chosen, let's just approximate using a known formula:
# PVsyst model: Tmodule = Tambient + (u_c + u_v * wind_speed)*Irradiance/800 (approx)
# For now, we will just assume we can directly use the sapm_cell method if model = 'sapm' or 'custom'.
# If 'pvsyst', we must implement a custom cell temp calculation. Let's do that now:

def calculate_cell_temp_pvsyst(poa_global, temp_air, wind_speed, u_c, u_v):
    # Approx: Tmodule = Tair + (poa_global/1000)*u_c + (poa_global/1000)*u_v*wind_speed
    # Actually, PVsyst formula: Tmodule = Tair + (poa_global * (u_c + u_v*wind_speed)/800)
    # We'll use this formula:
    return temp_air + (poa_global*(u_c + u_v*wind_speed)/800)

def get_cell_temperature(env_data, weather, params):
    if 'model' in params and params['model'] == 'pvsyst':
        # pvsyst model
        return calculate_cell_temp_pvsyst(env_data['poa_global'], weather['temp_air'], weather['wind_speed'], params['u_c'], params['u_v'])
    else:
        # assume SAPM/custom SAPM-like
        return pvlib.temperature.sapm_cell(poa_global=env_data['poa_global'],
                                           temp_air=weather['temp_air'],
                                           wind_speed=weather['wind_speed'],
                                           a=params['a'],
                                           b=params['b'],
                                           deltaT=params['deltaT'])

min_db_temp_ashrae = -3.7
max_db_temp_ashrae = 36.6
module['Bvoco%/C']=(module['Bvoco']/module['Voco'])*100
module['Bvmpo%/C']=(module['Bvmpo']/module['Vmpo'])*100
module['Aimpo%/C']=(module['Aimp']/module['Impo'])*100
module['TPmpo%/C']=module['Bvmpo%/C']+module['Aimpo%/C']
max_module_voc= module['Voco']*(1+((min_db_temp_ashrae-25)*module['Bvoco%/C']/100))
max_string_design_voltage = inverter['Vdcmax']
max_module_series=int(max_string_design_voltage/max_module_voc)
dc_ac_ratio=inverter['Pdco']/inverter['Paco']
single_module_power=module['Vmpo']*module['Impo']
T_add=25
min_module_vmp= module['Vmpo']*(1+((T_add+max_db_temp_ashrae-25)*module['TPmpo%/C']/100))
min_module_series_okay=math.ceil(inverter['Mppt_low']*dc_ac_ratio/min_module_vmp)

if max_module_series<min_module_series_okay:
    no_of_series_module=min_module_series_okay
    max_parallel_string=1
else:
    diff_ratio = []
    series_module = []
    for i in range(min_module_series_okay, max_module_series+1):
        source_circuit=single_module_power*i
        max_string=int(inverter['Pdco']/source_circuit)
        if max_string<1:
            max_string=1
        ratio=max_string*single_module_power*i/inverter['Paco']
        diff_r=abs(dc_ac_ratio-ratio)
        diff_ratio.append(diff_r)
        series_module.append((i, max_string))
    idx = diff_ratio.index(min(diff_ratio))
    no_of_series_module, max_parallel_string = series_module[idx]

number_of_inverters_needed=math.ceil(system_size_mw*1E6/inverter['Pdco'])
total_AC_system_size=inverter['Paco']*number_of_inverters_needed
total_DC_system_size=inverter['Pdco']*number_of_inverters_needed
dc_ac_ratio = total_DC_system_size/total_AC_system_size

location_obj = location.Location(lat, lon, 'Etc/GMT', alt)
surface_tilt = 30
surface_azimuth = 180
total_loss = {'soiling':2, 'shading':0, 'snow':0, 'mismatch':2, 'wiring':2, 'connections':0, 'lid':0, 'nameplate_rating':0, 'age':0, 'availability':0}

sol_data = pvlib.solarposition.get_solarposition(time=weather.index,latitude=lat,longitude=lon,altitude=alt,temperature=weather['temp_air'])
sol_data['dni_extra']=pvlib.irradiance.get_extra_radiation(weather.index)
env_data = pvlib.irradiance.get_total_irradiance(surface_tilt=surface_tilt,
                                                 surface_azimuth=surface_azimuth,
                                                 solar_zenith=sol_data['apparent_zenith'],
                                                 solar_azimuth=sol_data['azimuth'], 
                                                 dni=weather['dni'], ghi=weather['ghi'],
                                                 dhi=weather['dhi'], dni_extra=sol_data['dni_extra'], model='haydavies')
env_data['aoi'] = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, sol_data['apparent_zenith'], sol_data['azimuth'])
env_data['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith=sol_data['apparent_zenith'])
env_data['am_abs'] = pvlib.atmosphere.get_absolute_airmass(env_data['airmass'], pressure=(weather['surface_pressure']*100))

weather['cell_temperature'] = get_cell_temperature(env_data, weather, temperature_model_parameters)

weather['effective_irradiance'] = pvlib.pvsystem.sapm_effective_irradiance(
    poa_direct=env_data['poa_direct'],
    poa_diffuse=env_data['poa_diffuse'],
    airmass_absolute=env_data['am_abs'], 
    aoi=env_data['aoi'], module=module
)

system_obj = pvlib.pvsystem.PVSystem(
    arrays=None, surface_tilt=surface_tilt, surface_azimuth=surface_azimuth, 
    albedo=0.2, module_parameters=module, 
    inverter_parameters=inverter,
    # For losses model and others:
    racking_model='open_rack', losses_parameters=total_loss,
    modules_per_string=no_of_series_module, strings_per_inverter=max_parallel_string
)

mc = pvlib.modelchain.ModelChain(system_obj, location_obj, losses_model='pvwatts')
mc.run_model_from_effective_irradiance(weather)
ac_result=mc.results.ac.copy()
ac_result[ac_result<0]=0

energy_production_kW = (ac_result*number_of_inverters_needed)/1000
annual_energy_production_kWh=energy_production_kW.resample('Y').sum().values[0]

federal_tax_credit=federal_tax_credit_percent/100*installed_cost  
state_tax_credit=state_tax_credit_percent/100*installed_cost
Total_Capital_Cost=installed_cost-federal_tax_credit-state_tax_credit

RPWF=(1-((1+interest_rate)**(-project_life)))/interest_rate
single_pwf=(1/((1+interest_rate)**project_life))
Utility_energy_cost_LCC=(annual_energy_production_kWh*electricity_rate)*RPWF

maintenance_value=annual_maintenance_costs*RPWF
salvage_val=salvage_value*single_pwf
PV_Life_cycle_cost=Total_Capital_Cost+maintenance_value-salvage_val
Net_LCC_cost=PV_Life_cycle_cost-Utility_energy_cost_LCC

Simple_Payback_Period=Total_Capital_Cost/(annual_energy_production_kWh*electricity_rate)
fixed_charge_rate=1/RPWF
LCOE=(((Total_Capital_Cost*fixed_charge_rate)+annual_maintenance_costs)/annual_energy_production_kWh)+voc
annual_savings, payback, capex = generate_kpi_metrics(annual_energy_production_kWh, electricity_rate, Total_Capital_Cost, Simple_Payback_Period)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Generation (kWh)", f"{annual_energy_production_kWh:,.0f}")
col2.metric("Annual Savings", f"{annual_savings:,.0f} {currency}")
col3.metric("Payback Period (yrs)", f"{payback:.2f}")
col4.metric("LCOE ($/kWh)", f"{LCOE:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Location & Weather", "System Design", "Financials", "Performance Analysis"])

with tab1:
    st.subheader("Weather Data")
    if st.checkbox('Show raw weather data'):
        st.dataframe(weather)
    monthly_ghi = weather['ghi'].resample('M').sum()
    ghi_chart = alt.Chart(monthly_ghi.reset_index()).mark_line(point=True).encode(
        x='index:T', y='ghi:Q'
    ).properties(title='Monthly GHI')
    st.altair_chart(ghi_chart, use_container_width=True)

with tab2:
    st.write(f"System Size: {system_size_kw:.2f} kW (DC)")
    st.write(f"Modules in series: {no_of_series_module}")
    st.write(f"Strings per inverter: {max_parallel_string}")
    st.write(f"Inverters needed: {number_of_inverters_needed}")
    st.write(f"Total DC Size: {total_DC_system_size/1e6:.2f} MW")
    st.write(f"Total AC Size: {total_AC_system_size/1e6:.2f} MW")
    st.write(f"DC/AC Ratio: {dc_ac_ratio:.2f}")
    if polygon_area > 0 and sizing_method == "Area-based System Size":
        st.write(f"Based on drawn area, approx. {possible_modules} modules fit.")
    st.write(f"System Type: {system_type}")
    st.write(f"Temperature Model Family: {model_family}")
    if model_family == 'sapm':
        if 'a' in temperature_model_parameters:
            st.write(f"Using SAPM model parameters: a={temperature_model_parameters['a']}, b={temperature_model_parameters['b']}, deltaT={temperature_model_parameters['deltaT']}")
        else:
            # no a,b,deltaT if user selected a pre-defined key but didn't show them 
            # They are in temperature_model_parameters anyway:
            st.write("Using predefined SAPM parameters:")
            st.write(temperature_model_parameters)
    elif model_family == 'pvsyst':
        st.write("Using PVSyst parameters:")
        st.write(temperature_model_parameters)
    else:
        st.write("Using Custom model:")
        st.write(temperature_model_parameters)

with tab3:
    st.write("**Financial Summary**")
    st.write(f"Total Capital Cost: {Total_Capital_Cost:,.0f} {currency}")
    st.write(f"Utility Energy Cost LCC: {Utility_energy_cost_LCC:,.0f} {currency}")
    st.write(f"PV Life Cycle Cost: {PV_Life_cycle_cost:,.0f} {currency}")
    st.write(f"Net LCC Cost: {Net_LCC_cost:,.0f} {currency}")
    st.write(f"Simple Payback Period: {Simple_Payback_Period:.2f} years")
    st.write(f"LCOE: {LCOE:.3f} {currency}/kWh")
    annual_cashflow = annual_energy_production_kWh*electricity_rate
    cash_flows = [-Total_Capital_Cost] + [annual_cashflow]*(project_life)
    npv_val = npf.npv(interest_rate, cash_flows)
    st.write(f"Net Present Value (NPV): {npv_val:,.0f} {currency}")

with tab4:
    monthly_prod = (energy_production_kW.resample('M').sum())
    monthly_chart = alt.Chart(monthly_prod.reset_index()).mark_bar().encode(
        x=alt.X('index:T', title='Month'),
        y=alt.Y('ac', title='Energy (kWh)'),
    ).properties(title='Monthly Energy Production')
    st.altair_chart(monthly_chart, use_container_width=True)

    poa_wh_m2=(env_data['poa_global'])
    poa_sum=poa_wh_m2.resample('Y').sum().values[0]
    Reference_Yield=poa_sum/1000
    Final_Yield=(annual_energy_production_kWh/(single_module_power*max_parallel_string*no_of_series_module*number_of_inverters_needed))*1000
    Performance_Ratio=Final_Yield/Reference_Yield
    Capacity_Factor=((annual_energy_production_kWh)/(single_module_power*max_parallel_string*no_of_series_module*number_of_inverters_needed*8760))*1000

    st.write(f"Reference Yield (kWh/m²): {Reference_Yield:.2f}")
    st.write(f"Final Yield (kWh/m²): {Final_Yield:.2f}")
    st.write(f"Performance Ratio (%): {Performance_Ratio*100:.2f}")
    st.write(f"Capacity Factor (%): {Capacity_Factor*100:.2f}")

st.markdown("---")
st.write("**Note:** This is a demonstration tool. Adjust inputs as needed for accurate and region-specific analysis. Powered by pvlib Python.")

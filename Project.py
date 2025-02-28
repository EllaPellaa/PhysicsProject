import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import folium
import streamlit as st
from streamlit_folium import st_folium
from scipy.signal import butter,filtfilt

df = pd.read_csv("https://raw.githubusercontent.com/EllaPellaa/PhysicsProject/refs/heads/main/LinearAcceleration.csv")

st.title('Kävelymatka')

# ----------- Alipäästösuodatin --------------

def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# ----------- Filtterin parametrit --------------
T = df['Time (s)'][len(df['Time (s)'])-1] - df['Time (s)'][0] #Koko datan pituus
N = len(df['Time (s)']) #Datapisteiden lukumäärä
fs = N/T #Näytteenottotaajuus (olettaen vakioksi)
nyq = fs/2 #Nyqvistin taajuus
order = 3 #Kertaluku
cutoff = 1/(0.2) #Cut-off taajuus


# ----------- Suodatetaan data --------------
df['filter_a_y'] =  butter_lowpass_filter(df['Linear Acceleration y (m/s^2)'], cutoff, fs, nyq, order)

# ----------- Lasketaan askeleet nollakohtien ylityksen perusteella --------------
filt_signal = df['filter_a_y']
jaksot = 0
for i in range(N-1):
    if filt_signal[i]/filt_signal[i+1] < 0: #True jos nollan ylitys, False jos ei ole
        jaksot = jaksot + 1

askeleet = round(jaksot / 2)

#Printataan askelmäärä sivulle
st.write('Askelmäärä laskettuna suodatuksen avulla: ', askeleet , " askelta.")


# ----------- Tehospektri --------------
f = df['filter_a_y'] #Signaali
t = df['Time (s)'] #Aika
N = len(f) #Havaintojen määrä
dt = np.max(t)/N #Näytteenottoväli

#Fourier-analyysi
fourier = np.fft.fft(f,N) #Fourier-muunnos
psd = fourier*np.conj(fourier)/N #Tehospektri
freq = np.fft.fftfreq(N,dt) #Taajuudet
L = np.arange(1,int(N/2)) #Negatiivisten ja nollataajuuksien rajaus


# ----------- Lasketaan askelmäärä fourier-analyysin avulla --------------

#Määritellään tehokkain taajuus
f_max = freq[L][psd[L] == np.max(psd[L])][0]

f_max = freq[L][psd[L] == np.max(psd[L])][0] #Taajuuden arvo, silloin kun tehon arvo saa maksimin. 
T = 1/f_max #Askeleeseen kuluva aika, eli jaksonaika

askelmäärä = np.max(t)*f_max

#Printataan sivulle
st.write('Askelmäärä laskettuna fourier-analyysin avulla:  ', round(askelmäärä) , " askelta.")


# ----------- GPS-data ja haversine --------------

df2 = pd.read_csv('https://raw.githubusercontent.com/EllaPellaa/PhysicsProject/refs/heads/main/Location.csv')

#Haversine formula
def haversine(lon1, lat1, lon2, lat2):
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a))
  r = 6371
  return c * r


# ----------- Lasketaan nopeus, matka, ja askelpituus --------------
df2['dist'] = np.zeros(len(df2)) #Distance
df2['time_diff'] = np.zeros(len(df2)) #Distance

for i in range(len(df2)-1):
  df2.loc[i,'dist'] = haversine(df2['Longitude (°)'][i], df2['Latitude (°)'][i], df2['Longitude (°)'][i+1], df2['Latitude (°)'][i+1])

  df2.loc[i,'time_diff'] = df2['Time (s)'][i+1] - df2['Time (s)'][i]
  df2['velocity'] = df2['dist']/df2['time_diff']
  df2['tot_dist'] = np.cumsum(df2['dist'])
  
keskinopeus = df2['velocity'].mean() * 1000
kokonaismatka = df2['tot_dist'].max() * 1000
askelpituus = kokonaismatka / round(askelmäärä) * 100

st.write("Keskinopeus on: ", round(keskinopeus, 1), "m/s")
st.write("Kokonaismatka: ", round(kokonaismatka), " metriä.")
st.write("Askelpituus: ", round(askelpituus), "cm")


# ----------- Piirretään kuvaaja suodatetusta datasta --------------
st.title("Suodatetun kiihtyvyysdatan y-komponentti")
st.line_chart(df, x = 'Time (s)', y = 'filter_a_y', y_label = 'Suodatettu y (m/s^2)', x_label = 'Aika (s)')


# ----------- Piirretään kuvaaja tehospektristä --------------
st.title("Tehospektri")
chart_data = pd.DataFrame(np.transpose(np.array([freq[L],psd[L].real])), columns=["freq", "psd"])
st.line_chart(chart_data, x = 'freq', y = 'psd' , y_label = 'Teho',x_label = 'Taajuus [Hz]')

# ----------- Kartta --------------

start_lat = df2['Latitude (°)'].mean()
start_lon = df2['Longitude (°)'].mean()
my_map = folium.Map(location = [start_lat, start_lon], zoom_start = 17)

folium.PolyLine(df2[['Latitude (°)','Longitude (°)']], color = 'red', weight = 2.5, opacity = 1).add_to(my_map)

st.title("Karttakuva")

st_map = st_folium(my_map, width=900, height=650)

#!/usr/bin/env python
# coding: utf-8

# # DATA PROGRAMMING IN PYTHON FINAL PROJECT
# ### Names: Michael Lee, Andrew McQueen, Munju Kam, Stephen Schmidt, Cindy Shen

# Data Source: https://catalog.data.gov/dataset/cable-office-speed-test
# 
# Description: The data includes ISP speed test results collected from user input 
# throughout the county. Speed test results are updated weekly. The dataset includes 
# survey information about internet performance in Montgomery County, Maryland.

# ## Import Packages & Reading CSV File into Data Frame

# In[1]:


import os
import pandas as pd
import numpy as np
from ipywidgets import widgets, interactive, Layout


# In[2]:


os.getcwd() #check working directory


# In[3]:


cable_speed_df = pd.read_csv("Cable_Office_Speed_Test.csv", header = 'infer')
cable_speed_df


# In[4]:


#CREATE DATA FRAME COPY
cs_df = cable_speed_df.copy()


# #### Note: Use "cs_df" for data cleaning/preprocessing and analysis. Do not make any changes to original data frame "cable_speed_df"

# ## Data Characteristics

# In[5]:


#Check data information
cs_df.info()


# ## Data Preprocessing/Cleaning

# ### Joining City Population Data to Data Frame

# In[6]:


unique_zip = cs_df.Zip.unique()


# City Name Source: https://mde.state.md.us/programs/LAND/RecyclingandOperationsprogram/Documents/www.mde.state.md.us/assets/document/zip_codes.pdf
# 
# Population Source: https://www.maryland-demographics.com/cities_by_population

# In[7]:


#Load city names data
city_pop_df = pd.read_excel("city_population_data.xlsx", header = 'infer')
city_pop_df = city_pop_df[city_pop_df['Zip'].isin(unique_zip)]
city_pop_df.head(n = 15)


# In[8]:


#Left join cs_df with city_names_df
cs_df = cs_df.merge(city_pop_df, on = 'Zip', how = 'left')


# In[9]:


#Review data
cs_df


# In[10]:


cs_df[cs_df.city_name.isnull()].Zip.unique()


# ### Updating Column Names and Values

# In[11]:


#Alter space with underscore for column names
cs_df.columns = [c.replace(' ', '_') for c in cs_df.columns]


# In[12]:


#Review unique provider
cs_df.Provider.unique()


# In[13]:


#Update provider values
cs_df['Provider'].replace({"ATT": "AT&T", "I don’t know": "Unknown"}, inplace = True)

print("ATT" in cs_df.Provider, "I don’t know" in cs_df.Provider)


# In[14]:


#Drop uncesseary columns
cs_df = cs_df.drop(["Additional_Comments","What_kind_of_service_plan_do_you_have?","Where_are_you_accessing_the_internet","What_are_you_using_the_internet_for?","Indoor","Do_you_buy_mobile_data_alone_or_bundle_with_talk_and_text?","How_much_is_your_monthly_bill?","Is_your_data_plan_unlimited?"], axis=1)


# ### Handling Null Values

# In[15]:


#Exclude rows with no internet provider
cs_df = cs_df[cs_df['Provider'].notna()]
cs_df = cs_df[cs_df['How_are_you_connected'].notna()]
print(len(cs_df))


# In[16]:


cs_df.groupby("Provider").Price_Per_Month.count()


# In[17]:


#Fill null values in the columns in column list with the average value for its provider, or by the column average
col_list = ['Price_Per_Month', 'Advertised_Download_Speed', 'Satisfaction_Rating', 
            'Advertised_Price_Per_Mbps', 'Actual_Price_Per_Mbps', 'Ping']
for col_name in col_list:
    cs_df[col_name] = cs_df[col_name].fillna(cs_df.groupby('Provider')[col_name].transform('mean'))
    cs_df[col_name] = cs_df[col_name].fillna(cs_df[col_name].mean())
    print(col_name, cs_df[col_name].notnull().sum(), sep = "---")


# ### Final Data Frame Characteristics

# In[18]:


def displaydf(dataframe):
    return print(dataframe.info())


# In[19]:


displaydf(cs_df)
#cs_df.info()


# ## QUESTIONS:
# ### 1. Which variables are most important in determining service quality?
# ### 2. How does plan price affect speed?
# ### 3. What machine learning model is best predicting satisfaction?
# ### 4. How does price and performance change over time?
# ### 5. How does city population affect speed?

# ## Exploratory Analysis

# In[20]:


#Review satisfaction rating by provider
provider=cs_df.groupby(["Provider"]).Price_Per_Month.mean()
provider = provider.sort_values(ascending=True)
provider


# In[21]:


#Create user-defined function for bar graph
def createbar(column):
    return column.plot(kind="barh")


# In[22]:


createbar(provider)
#provider.plot(kind="barh")


# In[23]:


#Review satisfaction rating by price
satisfaction_price=cs_df.groupby(["Satisfaction_Rating"]).Price_Per_Month.mean()
satisfaction_price = satisfaction_price.sort_values(ascending=True)
satisfaction_price


# In[24]:


#Review Download speed by connection type
connectiond=cs_df.groupby(["How_are_you_connected"]).Download_Speed.mean()
connectiond = connectiond.sort_values(ascending=True)
connectiond


# In[25]:


createbar(connectiond)
#connectiond.plot(kind="barh")


# In[26]:


#Review upload speed by connection type
connectionu=cs_df.groupby(["How_are_you_connected"]).Upload_Speed.mean()
connectionu = connectionu.sort_values(ascending=True)
connectionu


# In[27]:


createbar(connectionu)
#connectionu.plot(kind="barh")


# In[28]:


#Extract year from Date Column
cs_df["Year"] = cs_df["Date"].str[-4:]
cs_df["Month"] = cs_df["Date"].str[:2]
cs_df["Time1"] = cs_df["Time"].str[:2]
cs_df


# In[29]:


#Review Price by year
price_year=cs_df.groupby(["Year"]).Price_Per_Month.mean()
price_year


# In[30]:


def createline(column):
    return column.plot(kind="line")


# In[31]:


createline(price_year)
#price_year.plot(kind="line")


# In[32]:


#Review download speed by year
speed_year=cs_df.groupby(["Year"]).Download_Speed.mean()
speed_year


# In[33]:


createline(speed_year)
#speed_year.plot(kind="line")


# In[34]:


#Review download speed by month
speed_month=cs_df.groupby(["Month"]).Download_Speed.mean()
speed_month


# In[35]:


createline(speed_month)
#speed_month.plot(kind="line")


# In[36]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

year16 = cs_df[cs_df['Year'] == '2016'].groupby(["Month"]).Download_Speed.mean()
year17 = cs_df[cs_df['Year'] == '2017'].groupby(["Month"]).Download_Speed.mean()
year18 = cs_df[cs_df['Year'] == '2018'].groupby(["Month"]).Download_Speed.mean()
year19 = cs_df[cs_df['Year'] == '2019'].groupby(["Month"]).Download_Speed.mean()
year20 = cs_df[cs_df['Year'] == '2020'].groupby(["Month"]).Download_Speed.mean()
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#Subsitute missing information with 0
s1 = pd.Series([0,0,0,0,0,0,0,0,0,0])
year16 = s1.append(year16)

s2 = pd.Series([0,0,0,0,0])
year17 = s2.append(year17)

#ploting
plt.plot(month, year16, 'o-y')
plt.plot(month, year17, 'o-c')
plt.plot(month, year18, 'o-g')
plt.plot(month, year19, 'o-b')
plt.plot(month, year20, 'o-r')
#set axis
plt.xlabel("Month")
plt.ylabel("download_speed")
#legend
plt.legend(['2016', '2017', '2018', '2019', '2020'])
plt.show()


# In[37]:


#Review download speed by time
speed_time=cs_df.groupby(["Time1"]).Download_Speed.mean()
speed_time


# In[38]:


#createline(speed_time)
speed_time.plot(kind="line")


# In[39]:


cs_df.plot(kind='scatter', x='Download_Speed', y='population', figsize=(10,10))


# ### Interactive Visualization

# In[40]:


#Interactive boxplot to view internet performance by provider and location (user defined function #1)
w_provider = widgets.Dropdown(
    description = 'provider',
    options = ['All Providers'] + sorted(list(set(cs_df.Provider))),
    value = 'All Providers',
    style = {"description_width": '50px'},
    layout = Layout(width="20%")
)

w_city = widgets.Dropdown(
    description = 'city',
    options = ['All Cities'] + sorted(list(set(cs_df.city_name))),
    value = 'All Cities',
    style = {"description_width": '50px'},
    layout = Layout(width="20%")
)

w_test_type = widgets.ToggleButtons(
    description = 'type',
    options = ['Business', 'Home', 'Mobile', 'Work', 'All Types'],
    value = 'All Types',
    style = {"description_width": '50px'},
    layout = Layout(width='100%')
)

def view(ttype, provider, city):
    if provider == "All Providers":
        df_tmp = cs_df
    else: 
        df_tmp = cs_df[cs_df.Provider == provider]
        
    if city == "All Cities":
        df_tmp = df_tmp
    else:
        df_tmp = df_tmp[df_tmp.city_name == city]
    
    if ttype == 'All Types':
        df_tmp = df_tmp
    else:
        df_tmp = df_tmp[df_tmp.How_Are_You_Testing == ttype]
        
    title = "Internet Performance by {} in {} for {}".format(provider, city, ttype)
    df_tmp[["Price_Per_Month"]].plot(kind = "box", title = title, figsize = (10,5))
    
i = interactive(view, provider = w_provider, city = w_city, ttype = w_test_type)
display(i)


# In[41]:


#importing Modules
#!pip install -U textblob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textblob import TextBlob


# In[42]:


#Market Price Analysis
series1=cs_df.groupby(["Provider"]).Response.sum()
series1 = series1.sort_values(ascending=True)
series1


# In[43]:


series2=cs_df.groupby(["Provider"]).Price_Per_Month.mean()
series2 = series2.sort_values(ascending=True)


# In[44]:


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
ax1 = axs[0]
ax2 = axs[1]

ax1.pie(series1.values, labels=series1.index, autopct='%.1f')
ax1.set(title="Market Survey")

ax2.barh(series2.index, series2.values)
ax2.set(ylabel="Provider", title="Average_Price_Per_Month")

fig.suptitle("Provider Analysis By Price")
plt.show()


# In[45]:


# Provider Performance Analysis
series3=cs_df.groupby(["Provider"]).Download_Speed.mean()
series3 = series3.sort_values(ascending=True)
series4=cs_df.groupby(["Provider"]).Upload_Speed.mean()
series4 = series4.sort_values(ascending=True)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
ax1 = axs[0]
ax2 = axs[1]

ax1.barh(series3.index, series3.values)
ax1.set(ylabel="Provider", title="Average_Download_Speed")

ax2.barh(series4.index, series4.values)
ax2.set(ylabel="Provider", title="Average_Upload_Speed")

fig.suptitle("Provider Analysis By Download_Speed & Upload_Speed")
plt.show()


# In[46]:


# Provider Performance Analysis
series5=cs_df.groupby(["Provider"]).Actual_Price_Per_Mbps.mean()
series5 = series5.sort_values(ascending=True)
series6=cs_df.groupby(["Provider"]).Ping.mean()
series6 = series6.sort_values(ascending=True)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
ax1 = axs[0]
ax2 = axs[1]

ax1.barh(series5.index, series5.values)
ax1.set(ylabel="Provider", title="Average_Actual_Price_Per_Mbps")

ax2.barh(series6.index, series6.values)
ax2.set(ylabel="Provider", title="Average_Ping")

fig.suptitle("Provider Analysis By Actual_Price_Per_Mbps & Ping")
plt.show()


# In[47]:


#Create correlation matrix with visualization (except zipcode)
pd.plotting.scatter_matrix(cs_df[["Response", "Date", "Time", "How_Are_You_Testing", "Provider", "How_are_you_connected","Price_Per_Month","Advertised_Download_Speed","Satisfaction_Rating","Download_Speed","Upload_Speed","Advertised_Price_Per_Mbps","Actual_Price_Per_Mbps","Ping"]], figsize=(10,10), diagonal="hist");


# In[48]:


# import modules
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16,10))  
# plotting correlation heatmap
dataplot = sb.heatmap(cs_df.corr(), cmap="YlGnBu", annot=True)
mp.show()


# In[49]:


#Price_Per_Month over time , very random no pattern of trend to follow
cs_df["Date"] = cs_df["Date"].apply(pd.to_datetime)
cs_df.info()


# In[50]:


cs_df.set_index("Date" , inplace=True)


# In[51]:


cs_df["Price_Per_Month"].plot(figsize=(16,8) , title = 'Price_Per_Month over time')


# In[52]:


#Participated surveying People mean resample
cs_df["Response"].resample('Y').mean().plot(kind='bar',figsize = (10,4))
plt.title('Participated Survey Trend')


# In[53]:


#Actual_Price_Per_Mbps mean resample
cs_df["Actual_Price_Per_Mbps"].resample('Y').mean().plot(kind='bar',figsize = (10,4))
plt.title('Actual_Price_Per_Mbps Trend')


# In[54]:


#Download_Speed mean resample
cs_df["Download_Speed"].resample('Y').mean().plot(kind='bar',figsize = (10,4))
plt.title('Download_Speed Performance Trend')


# In[55]:


#Upload_Speed mean resample
cs_df["Upload_Speed"].resample('Y').mean().plot(kind='bar',figsize = (10,4))
plt.title('Upload_Speed Performance Trend')


# In[56]:


#Ping mean resample
cs_df["Ping"].resample('Y').mean().plot(kind='bar',figsize = (10,4))
plt.title('Ping Performance Trend')


# ## Machine Learning Models

# ### Add Target Column, Assign Features and Target, Split Data into Training & Testing

# In[57]:


#create target column
cs_df["satisfaction"] = cs_df.Satisfaction_Rating.apply(lambda x: 0 if x <= 2.5 else 1)


# In[58]:


cs_df[cs_df.satisfaction == 0]


# In[59]:


features = ["Price_Per_Month", "Download_Speed", "Upload_Speed", "Ping"]
target = "satisfaction"


# In[60]:


X = cs_df[features]
y = cs_df[target]


# In[61]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


# ### Supervised Learning

# In[62]:


summary = dict()
models = dict()


# #### K Nearest Neighbor (KNN)

# In[63]:


from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors=1)
knc


# In[64]:


knc.fit(X_train, y_train)
knc.score(X_train, y_train),knc.score(X_test, y_test)


# In[65]:


summary["k-NNs"] = round(knc.score(X_test, y_test), 3)
models["k-NNs"] = knc
summary


# #### Decision Tree

# In[66]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=0)
dtc


# In[67]:


dtc.fit(X_train, y_train)
dtc.score(X_train, y_train),dtc.score(X_test, y_test)


# In[68]:


summary["Decision Trees"] = round(dtc.score(X_test, y_test), 3)
models["Decision Trees"] = dtc
summary


# #### Random Forest

# In[69]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)
rfc


# In[70]:


rfc.fit(X_train, y_train)
rfc.score(X_train, y_train),rfc.score(X_test, y_test)


# In[71]:


summary["Random Forest"] = round(rfc.score(X_test, y_test), 3)
models["Random Forest"] = rfc
summary


# #### Linear Support Vector Machines (SVM)

# In[72]:


from sklearn.svm import LinearSVC

lsvc = LinearSVC(random_state=0)
lsvc


# In[73]:


lsvc.fit(X_train, y_train)
lsvc.score(X_train, y_train),lsvc.score(X_test, y_test) 


# In[74]:


summary["Linear SVMs"] = round(lsvc.score(X_test, y_test), 3)
models["Linear SVMs"] = lsvc
summary


# #### Kernelized Support Vector Machines (SVM)

# In[75]:


from sklearn.svm import SVC

svc = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=0)
svc


# In[76]:


svc.fit(X_train, y_train)
svc.score(X_train, y_train),svc.score(X_test, y_test) 


# In[77]:


summary["Kernelized SVMs"] = round(svc.score(X_test, y_test), 3)
models["Kernelized SVMs"] = svc
summary


# #### Neural Networks

# In[78]:


from sklearn.neural_network import MLPClassifier

mlpc = MLPClassifier(hidden_layer_sizes=(10,), random_state=0)
mlpc


# In[79]:


mlpc.fit(X_train, y_train)
mlpc.score(X_train, y_train),mlpc.score(X_test, y_test)  


# In[80]:


summary["Neural Networks"] = round(mlpc.score(X_test, y_test), 3)
models["Neural Networks"] = mlpc
summary


# #### Model Evaluation

# In[81]:


summary


# In[82]:


best_model = max(summary, key = summary.get)
final_model = models[best_model]
final_model


# In[83]:


#features = ["Price_Per_Month", "Actual_Price_Per_Mbps", "Download_Speed", "Upload_Speed", "Ping"]
def prediction():
    prediction_values = []
    print("Enter price per month: ")
    prediction_values.append(float(input()))
    print("Enter download speed: ")
    prediction_values.append(float(input()))
    print("Enter upload speed: ")
    prediction_values.append(float(input()))
    print("Enter ping: ")
    prediction_values.append(float(input()))
    
    prediction_df = pd.DataFrame(columns = features)
    prediction_df.loc[0] = prediction_values
    final_prediction = final_model.predict(prediction_df)[0]
   
    if final_prediction == 1:
        result = "Satisfied"
    elif final_prediction == 0:
        result = "Unsatisfied"
    return "Prediction using {} Model: {} Customer".format(best_model, result)


# In[84]:


prediction()


# ### Unsupervised Learning

# In[85]:


#Defining Features
features = ["Price_Per_Month","Actual_Price_Per_Mbps","Download_Speed","Upload_Speed","Ping"]
X = cs_df[features]
X


# #### K-Means Clustering

# We are going to perform clustering analysis on the dataset and use the elbow method to determine the optimal number of clusters

# In[86]:


from sklearn.cluster import KMeans

distortions = []
models = {}


# In[87]:


K = range(1,10)

for k in K:
    kmeans = KMeans(n_clusters = k, random_state = 0)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)
    models[k] = kmeans


# #### Model Evaluation

# In[88]:


import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Cluster Evaluation')
plt.show()


# In[89]:


final_k = 3
final_model = models[final_k]
final_model


# In[90]:


final_model.fit(X)
cs_df["label"] = final_model.predict(X)


# In[91]:


cs_df.label.value_counts()


# In[92]:


cluster_1st, cluster_2nd, cluster_3rd = cs_df.label.value_counts().index
cluster_1st, cluster_2nd, cluster_3rd


# #### Cluster 1

# In[93]:


cs_df[cs_df.label == cluster_1st].sample(n=10, random_state=0)


# #### Cluster 2

# In[94]:


cs_df[cs_df.label == cluster_2nd].sample(n=10, random_state=0)


# #### Cluster 3

# In[95]:


cs_df[cs_df.label == cluster_3rd] #only 4 rows in this cluster


# ### Cluster Interpretation
# ##### Cluster1: Highest paying group (High performace in download/upload speed)
# ##### Cluster2: Average paying group (Average performance but has lowest satisfaction rating)
# ##### Cluster3: Lowest paying group (Low performance in download/upload speed)

# In[96]:


cs_df.groupby("label").mean()


# In[97]:


#Front-end user-defined function to show users their options with preferred price and speed.

def myOptions():
    print("Enter your city: ")
    cty = str(input())
    #check if city exists in data
    if cty not in cs_df["city_name"].unique():
        return "Invalid city."
    print("Enter your preferred price: ")
    prefprice = float(input())
    print("Enter your preferred download speed: ")
    prefdown = float(input())
    
    #setting new dataframe
    newdf = cs_df[cs_df.city_name == cty]
    newdf2 = newdf[(newdf.Price_Per_Month <= prefprice) & (newdf.Download_Speed >= prefdown) & (newdf.Provider != "Other")][["Provider", "Price_Per_Month", "Download_Speed", "Upload_Speed", "Actual_Price_Per_Mbps"]]
    
    
    #check if dataframe is empty
    if len(newdf2.index) == 0:
        return "No users with these parameters."
    
    #get price/mbps for each provider
    perMbpsMean = newdf2.groupby("Provider").Actual_Price_Per_Mbps.mean()
    perMbpsMeanDf = perMbpsMean.to_frame()
    bestDeal = "{:.2f}".format(float(perMbpsMean.min()))
    
    #return best provider, given constraints
    bestDealProvider = str(perMbpsMeanDf.idxmin())
    bestDealProvider = bestDealProvider[25:][:-14]

    avgPrice = float(newdf2[(newdf2.Provider == bestDealProvider)].Price_Per_Month.mean())
    avgDown = float(newdf2[(newdf2.Provider == bestDealProvider)].Download_Speed.mean())
    finalPrice = "{:.2f}".format(avgPrice)
    finalDown = "{:.2f}".format(avgDown)

    print(cty + " users had the best service with " + str(bestDealProvider) + ", paying $" + str(bestDeal) + " per Mbps.")
    print("In " + cty + ", " + str(bestDealProvider) + " has an average price of $" + str(finalPrice) + " per month.")
    print("In " + cty + ", " + str(bestDealProvider) + " has an average download speed of " + str(finalDown) + " Mbps.")
    createbar(perMbpsMean)


# In[98]:


myOptions()


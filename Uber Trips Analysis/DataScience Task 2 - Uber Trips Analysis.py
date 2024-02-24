#!/usr/bin/env python
# coding: utf-8

# # Data Science Task 1
# 
# ## Task: Uber Trips Analysis
# 
# ## Objective: Analyze Uber rides to detect patterns and relationships between different variables using Python

# ### Tasks:
# 
# ### 1. Read the dataset and display records to understand its structure.
# ### 2. Uncover hidden patterns in the data using data analysis techniques.
# ### 3. Identify relationships between variables such as date, time, location, and ride demand.
# ### 4. Visualize the relationships through graphs and charts.

# ### Steps
# 
# #### 1- Import necessary libraries
# #### 2- Read the dataset using Pandas
# #### 3- Explore the dataset properties
# #### 4- Visualize the relationship between different variables and draw insights

# In[1]:


pip install pandas matplotlib seaborn


# #### 1- Import necessary libraries

# In[2]:


#To read the dataset
import pandas as pd

#For visualization
import matplotlib.pyplot as plt
import seaborn as sns


# ### 2- Read the dataset using Pandas

# In[3]:


# Load the dataset
uber_data = pd.read_csv('dataset/uber_trips_data.csv')

# Display the first few records to understand its structure
print("First few records of the dataset:")
uber_data.head()


# ##### This data contains data about date and time, latitude and longitude, and a Base column that contains code affiliated with the uber pickup.

# In[4]:


#Display the last 5 records
uber_data.tail()


# ### 3- Data exploration and analysis

# In[5]:


#Find the shape of the dataset
uber_data.shape


# In[6]:


#Understand the dataset properties
uber_data.info()


# In[7]:


# Descriptive statistics summary
print("Descriptive statistics summary:")
print(uber_data.describe())


# In[8]:


# Check for missing values
print("\nMissing values:")
print(uber_data.isnull().sum())


# In[9]:


# Correlation analysis
print("\nCorrelation analysis:")
correlation_matrix = uber_data.corr()
print(correlation_matrix)


# #### Let's break the Date/Time column to "day", "hour", & "weekday".

# In[10]:


#Change the "Date/Time" column's data type from string to datetime
uber_data["Date/Time"] = uber_data["Date/Time"].map(pd.to_datetime) 


# In[11]:


#Convert "Date/Time" column from string data type into DateTime
uber_data["day"] = uber_data["Date/Time"].apply(lambda x: x.day)
uber_data["hour"] = uber_data["Date/Time"].apply(lambda x: x.hour)
uber_data["weekday"] = uber_data["Date/Time"].apply(lambda x: x.weekday())
uber_data.head(5)


# ### 4- Identify any other patterns or anomalies based on your domain knowledge and data understanding

# In[12]:


## Pattern Detection 

# Peak Ride Hours
# Example: Peak ride hours
peak_hours = uber_data.groupby('hour')['hour'].count()
print("\nPeak ride hours:")
print(peak_hours)


# ##### According to this distribution we can analyse the following :
# 
# ###### 1. Peak Hours: The analysis shows that the highest number of Uber rides occurs during the late afternoon and the evening hours, particularly between 5 PM and 9 PM. This suggests that these hours are peak hours of ride demand, likely due to commuters returning home from work or individuals going out for leisure activities in the evening.
# 
# ###### 2. Morning Rush: There is also a significant number of rides during the morning hours, particularly between 7 AM and 10 AM, which corresponds to the morning rush hour. his spike in ride demand suggests that many users rely on Uber for their daily commute to work or other daytime activities.
# 
# ###### 3. Late Night Hours: While the number of rides decreases during the late night and early morning hours, there is still a considerable demand for Uber rides during these times, especially between 11 PM and 3 AM. This could be attributed to people returning home after social gatherings or late-night events.
# 
# ###### 4. Off-Peak Hours: The early morning hours between 12AM and 5 AM have the lowest ride counts, which is expected as it represents the time when most people are asleep, and there is less demand for transportation.
# 
# ###### 5. Consistent Demand: Throughout the day, there's a consistent demand for Uber rides, with variations depending on the time of day. The data suggests that Uber services are utilized consistently across different hours, albeit with fluctuations in demand.

# In[13]:


# Example: Distribution of rides by day of the week
day_of_week_distribution = uber_data['weekday'].value_counts()
print("\nDistribution of rides by day of the week:")
print(day_of_week_distribution)


# ##### According to this distribution we can analyse the following :
# 
# ###### 1. Weekday Distribution: The analysis shows the count of Uber rides for each day of the week. From the output, it's evident that the highest number of rides occurs on Monday (weekday 1), followed closely by Friday (weekday 5) and Thursday (weekday 4). This suggests that weekdays, particularly early in the week, have higher ride demand compared to weekends.
# 
# ###### 2. Weekend Distribution: The lowest number of rides occurs on Saturday (weekday 0), indicating relatively lower ride demand on weekends. Sunday (weekday 6) also has fewer rides compared to weekdays, but it still has a significant number of rides, likely due to leisure activities and social events.
# 
# ###### 3. Consistent Weekday Demand: Across Monday to Friday, there's a relatively consistent demand for Uber rides, with slight variations between weekdays. This suggests that Uber services are utilized consistently throughout the weekdays, likely driven by commuting to work or other weekday activities.
# 
# ###### 4. Decreased Weekend Demand: On weekends (Saturday and Sunday), there's a noticeable decrease in ride counts compared to weekdays. This decrease in demand is expected as weekends typically involve fewer work-related activities and may result in decreased transportation needs.

# In[14]:


# Example: Distribution of rides by day of the month
day_of_month_distribution = uber_data['day'].value_counts()
print("\nDistribution of rides by day of the month:")
print(day_of_month_distribution)


# ##### According to this distribution we can analyse the following :
# 
# ###### 1. Day of the Month Distribution: The analysis shows the count of Uber rides for each day of the month. From the output, it's evident that there are fluctuations in ride counts across different days of the month.
# 
# ###### 2. Peak Days: Certain days of the month stand out with higher ride counts, such as the 13th, 5th, 19th, and 6th, which have the highest number of rides. These peaks in ride counts may coincide with specific events, promotions, or high-demand periods.
# 
# ###### 3. Variability: While some days experience higher ride counts, there's variability in ride demand throughout the month. This variability may be influenced by factors such as day of the week, holidays, paydays, or other external factors impacting transportation needs.
# 
# ###### 4. Lowest Ride Counts: Conversely, there are also days with lower ride counts, such as the 1st of the month, which has the lowest number of rides. These days may correspond to periods of decreased activity or reduced transportation demand.

# ### 5- Identify relationships between variables such as date, time, location, and ride demand.

# In[15]:


## Relationship Identification 

# Example 1: Relationship between time (hour) and ride demand
hourly_demand_relationship = uber_data.groupby('hour').size()
print("\nRelationship between time (hour) and ride demand:")
print(hourly_demand_relationship)


# ##### The analysis based on this relationship:
# 
# ###### 1. Relationship between Hour and Ride Demand: The analysis shows the count of Uber rides for each hour of the day. From the output, it's evident that there is a strong relationship between the time of day (hour) and ride demand.
# 
# ###### 2. Peak Hours: There are clear peak hours of ride demand, particularly during the late afternoon and early evening hours, between 5 PM and 9 PM. These hours correspond to the highest number of rides, indicating peak demand for Uber services during this time.
# 
# ###### 3. Morning Rush: There's also a noticeable increase in ride demand during the morning hours, between 7 AM and 9 AM, which corresponds to the morning rush hour. This spike in ride demand suggests that many users rely on Uber for their daily commute to work or other daytime activities.
# 
# ###### 4. Off-Peak Hours: The early morning hours between 12 AM and 6 AM generally have lower ride demand compared to other times of the day, indicating off-peak hours when demand for rides is relatively low.

# In[16]:


# Example 2: Relationship between location and ride demand (if location data is available)
# For example:
location_demand_relationship = uber_data.groupby(['Lat', 'Lon']).size()
print("\nRelationship between location and ride demand:")
print(location_demand_relationship)


# ##### The analysis based on this relationship:
# 
# ###### 1. Relationship between Location and Ride Demand: The analysis groups the Uber rides by latitude and longitude coordinates, providing a count of rides for each unique location. Each pair of latitude and longitude represents a specific location from which Uber rides were requested.
# 
# ###### 2. Spatial Distribution: The output shows the count of rides for each unique location, indicating the spatial distribution of ride demand across different geographical areas. Locations with higher ride counts suggest areas of higher demand for Uber services, while locations with lower ride counts may indicate areas with less demand.
# 
# ###### 3. Heatmap or Visualization: To better understand the relationship between location and ride demand, it's beneficial to visualize the data using techniques such as heatmaps or scatter plots. These visualizations can provide insights into hotspots of ride demand, popular pickup/drop-off locations, and spatial patterns in ride distribution.
# 
# ###### 4. Optimization Opportunities: Understanding the relationship between location and ride demand can help optimize driver allocation, identify areas for service expansion, and improve overall service coverage. By focusing resources on areas with high demand, Uber can enhance efficiency and customer satisfaction.

# #### Visualization

# In[17]:


# Group the data by latitude and longitude, and calculate the ride counts for each location
location_demand_relationship = uber_data.groupby(['Lat', 'Lon']).size().reset_index(name='Ride Count')

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(location_demand_relationship.pivot_table(index='Lat', columns='Lon', values='Ride Count'), cmap='viridis')
plt.title('Heatmap of Ride Demand by Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[18]:


# Example 3: Correlation analysis between variables
correlation_matrix = uber_data.corr()
print("\nCorrelation matrix between variables:")
print(correlation_matrix)


# #####  The analysis of the relationship between location (latitude and longitude) and ride demand:
# 
# ###### 1. Relationship between Location and Ride Demand: The analysis groups Uber rides by latitude and longitude coordinates, allowing us to examine the spatial distribution of ride demand across different geographical areas. Each pair of latitude and longitude represents a specific location from which Uber rides were requested.
# 
# ###### 2. Spatial Distribution: The output provides the count of rides for each unique location, indicating the spatial distribution of ride demand. Locations with higher ride counts suggest areas of greater demand for Uber services, while locations with lower ride counts may indicate areas with lower demand.
# 
# ###### 3. Heatmap Visualization: To better understand the relationship between location and ride demand, visualizations such as heatmaps or scatter plots can be generated. These visualizations can reveal hotspots of ride demand, identify popular pickup/drop-off locations, and uncover spatial patterns in ride distribution.
# 
# ###### 4. Optimization Opportunities: Understanding the relationship between location and ride demand is crucial for optimizing driver allocation, identifying areas for service expansion, and improving overall service coverage. By focusing resources on areas with high demand, Uber can enhance operational efficiency and customer satisfaction.

# #### Visualization

# In[1]:


# Correlation heatmap
plt.subplot(2, 2, 4)
correlation_matrix = uber_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()


# ### 6- Visualize the relationships through graphs and charts. 

# In[20]:


# Visualize the relationships
plt.figure(figsize=(12, 6))

# Example 1: Plot ride demand over day
sns.set(rc={'figure.figsize':(12, 10)})
sns.distplot(uber_data["day"])


# ##### By looking at the daily trips we can say that the Uber trips are rising on the working days and decreases on the weekends.

# In[21]:


# Example 2: Plot ride demand by hour of the day
plt.subplot(2, 2, 2)
hourly_demand_plot = uber_data.groupby('hour').size().plot()
plt.title('Ride Demand by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Ride Demand')

plt.tight_layout()
plt.show()


# ##### According to the hourly data, the Uber trips decreases after midnight and then start increasing after 5 am and the trips keep rising till late evening such that this is the busiest hour for Uber then the trips start decreasing.

# In[22]:


sns.distplot(uber_data["hour"])


# ##### According to the hourly data, the Uber trips decreases after midnight and then start increasing after 5 am and the trips keep rising till 6 pm such that 6 pm is the busiest hour for Uber then the trips start decreasing.

# In[23]:


sns.distplot(uber_data["weekday"])


# ##### In the above figure 0 indicates Sunday, on Sundays the Uber trips and more than Saturdays so we can say people also use Uber for outings rather than for just going to work. On Saturdays, the Uber trips are the lowest and on Mondays, they are the highest.

# In[24]:


# Example 3: Plot ride counts by day of the week
plt.figure(figsize=(10, 6))
sns.countplot(data=uber_data, x='weekday')
plt.title('Ride Counts by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Ride Count')
plt.show()


# ##### In the above figure 0 indicates Sunday, on Sundays the Uber trips and more than Saturdays so we can say people also use Uber for outings rather than for just going to work. On Saturdays, the Uber trips are the lowest i.e., below 120000 rides and on Mondays, they are the highest, i.e., more than 160000 rides.

# In[25]:


# Example 4: Plot ride counts by hour, segmented by day of the week
plt.figure(figsize=(12, 6))
sns.countplot(data=uber_data, x='hour', hue='weekday')
plt.title('Ride Counts by Hour, Segmented by Day of the Week')
plt.xlabel('Hour of the Day')
plt.ylabel('Ride Count')
plt.legend(title='Day of Week')
plt.show()


# ##### From the chart, we can observe several trends
# 
# ###### 1. Early morning hours (12 AM to 5 AM) have the lowest ride counts across all days, with a slight uptick at 5 AM.
# 
# ###### 2. Weekdays (Monday through Friday, days 1-5) show similar patterns with a sharp increase in rides during the morning rush hour (approximately 6 AM to 9 AM) and a second peak during the evening rush hour (approximately 4 PM to 7 PM).
# 
# ###### 3. Sunday (day 0) has a distinct spike at 2 AM that is not present on other days, which could be due to late-night activities on Saturday leading into early Sunday morning.
# 
# ###### 4. The weekend days (Sunday and Saturday, days 0 and 6) show a different pattern from weekdays, with a more gradual increase in ride counts throughout the day and extended activity into the late evening, especially noticeable on Saturday.
# 
# ###### 5. Saturday (day 6) has higher ride activity in the evening hours, extending up to 11 PM, which is not as pronounced on other days.
# 
# ##### These patterns suggest that ride behavior varies between weekdays and weekends, with weekends having a more spread out distribution of rides throughout the day and into the night. Weekdays have concentrated peaks correlating with typical work commute times.

# In[26]:


# Example 5: Heatmap of ride counts by hour and day of the week
hour_day_counts = uber_data.groupby(['hour', 'weekday']).size().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(hour_day_counts, cmap='Blues', linecolor='white', linewidth=1)
plt.title('Heatmap of Ride Counts by Hour and Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Hour of the Day')
plt.show()


# ##### The analysis of the relationship depicted in the heatmap:
# 
# ###### 1. Daily Patterns: The heatmap shows a clear daily pattern where certain hours of the day have higher ride counts. For example, there are peaks during what could be typical rush hour times in the morning and late afternoon/evening. This suggests that Uber usage is higher during times when people are likely commuting to and from work.
# 
# ###### 2. Weekday vs. Weekend: There is a discernible difference in the pattern of ride counts between weekdays (0-4) and weekends (5-6). Weekdays generally have higher ride counts during rush hours, while the weekends tend to have a more even distribution of rides throughout the day, with a notable increase in the late evening and nights, which may be due to leisure activities.
# 
# ###### 3. Hourly Trends: During the nighttime hours, from around 0 to 5 (midnight to early morning), the ride counts are generally lower, which is expected as this is typically when most people are at home. However, there is a slight increase during the early hours on days 5 and 6, which could suggest that people are using Uber services late into the night on weekends, possibly after social events.
# 
# ###### 4. Midday Activity: There seems to be a consistent mid-level demand for rides during the midday hours across all days of the week, which could be due to a variety of activities such as errands, lunch breaks, or non-commute related travel.

# In[27]:


# Correlation of Weekday and Hour
df = uber_data.groupby(["weekday", "hour"]).apply(lambda x: len(x))
df = df.unstack()
sns.heatmap(df, annot=False)


# ##### The heatmap which is another representation of the Uber rides data, grouped by weekdays and hours, similar to the previous one but with a different color scheme. This heatmap uses a color gradient from dark purple to white, with darker colors indicating fewer rides and lighter colors indicating more rides. Here's the relationship analysis:
# 
# ###### 1. Peak Hours on Weekdays: The heatmap shows that on weekdays (0-4), there are two distinct time periods with higher ride counts, likely corresponding to the morning and evening rush hours. These times are represented by lighter shades, particularly around 8-9 AM and 5-6 PM. This pattern is consistent with typical workday commuting behavior.
# 
# ###### 2. Evening Activity on Weekends: On days 5 and 6, which represent the weekend, there is a noticeable increase in lighter shades during the evening hours, starting from around 5 PM and extending later into the night compared to weekdays. This suggests that on weekends, Uber rides are more popular in the evening, possibly due to social activities and nightlife.
# 
# ###### 3. Late Night and Early Morning Rides: The darkest shades occur during the late-night and early morning hours (from midnight to 5 AM), indicating the lowest ride counts. However, on weekend early mornings (particularly on day 6), there is a slight increase in activity, which can be attributed to people returning home from late-night events.
# 
# ###### 4. Midday Consistency: Throughout the week, there is a consistent level of demand for rides during the midday hours (from around 10 AM to 3 PM), as indicated by the medium shades in the heatmap. This could be related to non-commute travel such as errands, lunch outings, or tourism-related activities.

# In[28]:


# Visualize the Density of Uber trips according to the regions 
uber_data.plot(kind='scatter', x='Lon', y='Lat', alpha=0.4, s=uber_data['day'], label='Uber Trips',
figsize=(12, 8), cmap=plt.get_cmap('jet'))
plt.title("Uber Trips Analysis")
plt.legend()
plt.show()


# ###### 1. Geographical Distribution: The plot shows a clear concentration of Uber trips in a specific area, indicating a city or urban region where Uber services are frequently used. The highest density of points is along a diagonal band, suggesting a correlation between the longitude and latitude in trip occurrences, which could align with the city's geography or street layout.
# 
# ###### 2. Density of Trips: There are regions with very high densities of points, especially in the middle of the plot, where the points are darker due to overlap. This suggests that these areas are likely to be popular locations for pickups or drop-offs, such as business districts, transportation hubs, or areas with a high concentration of entertainment venues and restaurants.
# 
# ###### 3. Point Size Variation: The size of the points varies significantly, which indicates that the 'day' variable in the dataset (used to size the points) likely represents the number of trips. Larger points correspond to more trips, which may suggest busy days or times. However, without additional context about what 'day' represents, this is an assumption.
# 
# ###### 4. Outliers: There are several points scattered further away from the main cluster of activity. These could represent trips to or from more remote areas, less frequently visited neighborhoods, or possibly data anomalies.
# 
# ###### 5. Coverage Area: The span of the plot suggests that Uber services are used across a wide geographic area, with a few points even further out from the dense central region, indicating the occasional long-distance trip.
# 
# ###### 6. Potential Data Issues: The plot does not seem to show any clear artifacts or anomalies that would suggest data issues, and the distribution appears to be typical for urban trip data.

# #### Now we can check the density of rides according to days, hours, and weekdays

# In[29]:


#Visualize the Density of rides per Day of month
fig,ax = plt.subplots(figsize = (12,6))
plt.hist(uber_data.day, width= 0.6, bins= 30)
plt.title("Density of trips per Day", fontsize=16)
plt.xlabel("Day", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)


# ##### From the above plot we can notice that the highest number of rides are during working days (Monday to Friday), while the least number of rides are in weekends.

# In[30]:


#Visualize the Density of rides per Weekday
fig,ax = plt.subplots(figsize = (12,6))
plt.hist(uber_data.weekday, width= 0.6, range= (0, 6.5), bins=7, color= "green")
plt.title("Density of trips per Weekday", fontsize=16)
plt.xlabel("Weekday", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)


# ##### The busiest day in the week for Uber is Monday. On the other hand, Saturday is the day with the least number of rides.

# In[31]:


#Visualize the Density of rides per hour
fig,ax = plt.subplots(figsize = (12,6))
plt.hist(uber_data.hour, width= 0.6, bins=24, color= "orange")
plt.title("Density of trips per Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)


# ##### It seems like the number of rides decrease gradually from 1 AM to 4 PM and then increases starting from 5 AM onward till it reaches 6 PM which is the hour with the highest number of rides.

# In[32]:


#Visualize the Density of rides per location
fig,ax = plt.subplots(figsize = (12,6))
x= uber_data.Lon
y= uber_data.Lat
plt.scatter(x, y, color= "purple")
plt.title("Density of trips per Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)


# ##### The region with the highest density of rides is near Manhattan and Newburgh. While the region with the lowest density is near New Jersey.

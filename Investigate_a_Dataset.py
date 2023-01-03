#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset - [tmdb-movies.csv]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# <li><a href="#limitations">Limitations</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# >This data set contains information
# about 10,000 movies collected from
# The Movie Database (TMDb),
# including user ratings and revenue.
# 
# >● Certain columns, like ‘cast’
# and ‘genres’, contain multiple
# values separated by pipe (|)
# characters.
# 
# >● There are some odd characters
# in the ‘cast’ column. Don’t worry
# about cleaning them. You can
# leave them as is.
# 
# >● The final two columns ending
# with “_adj” show the budget and
# revenue of the associated movie
# in terms of 2010 dollars,
# accounting for inflation over
# time.
# 
# 
# ### Dataset Description 
# 
# > This is a dataset showing the ratings of movies from 1960 to 2015 on Imdb, it also shows some more information like, revenue, budjet and cast, etc
# 
# 
# ### Question(s) for Analysis
# >1- Is there a change in revenues of movies over years and what are the years with minimum and maximum revenues?
# >2- Is there a change in popularity of movies over years and what are the years with minimum and maximum movies popularity?
# >3- What are the top 10 movies acheiving revenues from 1960 till 2015?
# ?4- What are the top 10 movies with highest budget from 1960 till 2015?

# In[3]:


# Use this cell to set up import statements for all of the packages that you
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', '')

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html


# <a id='wrangling'></a>
# ### Data Wrangling
# 
# In this section of the report, I will load in the data, check for cleanliness, and then trim and clean my dataset for analysis. 

# In[4]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df=pd.read_csv('tmdb-movies.csv')
df.head()


# In[5]:


# exploring the shape of the dataset
df.shape


# >The dataset comtains 10866 rows and 21 coloumns

# In[6]:


# exploring data types of each coloumn
df.dtypes


# In[7]:


# defining the number of duplicated rows
df.duplicated().sum()


# >we have 1 duplicated rows in the dataset will be removed in the data cleaning section

# In[8]:


# defining info of the dataset
df.info()


# >no missing data

# In[9]:


df.describe()


# >Minimum runtime is 0 which is non logical, so I am going to remove these rows because it will affect my analysis

# In[10]:


# Counting the number of occurence of each item in the budget column
df['budget'].value_counts()


# >Since there are alot of non logical budgets, I am going to remove non logical budgets with the effective counts so I am going to remove rows with 0 budget.

# In[11]:


# Counting the number of occurence of each item in the revenue column
df['revenue'].value_counts()


# >Since there are alot of non logical revenues, I am going to remove non logical revenues with the effective counts so I am going to remove rows with 0 revenue since they weights alot regarding the total sample.

# In[12]:


# defining number of nulls in each column
df.isna().sum()


# >There are some information missing, and should be added by investigating the right information, the missing information can't be added via python, we have to search for the correct information and add it manually.

# In[13]:


# removing the duplicate row
df.drop_duplicates(keep = 'first', inplace = True)
df.shape


# >The duplicated row was removed succefully and number of rows decreased to be 10865

# In[14]:


# removing rows which contain budget = 0
df=df.loc[df["budget"]!=0]
df.shape


#  >rows which contains movies budget = 0 were removed succefully and number of rows became 5169

# In[15]:


# removing rows which contain revenue = 0
df=df.loc[df["revenue"]!=0]
df.shape


# >rows which contains movies revenue = 0 were removed succefully and number of rows became 3854

# In[16]:


# removing rows which contain runtime = 0
df=df.loc[df["runtime"]!=0]
df.shape


# >All rows which contain non logical data were removed succefully and number of rows became 3854

# In[17]:


# checking if there is duplicated rows with the same imdb id
df["imdb_id"].duplicated().sum()


# >There is no rows with duplicated Imdb ids

# In[18]:


# checking if there is duplicated rows with the same id
df["id"].duplicated().sum()


# >There is no rows with duplicated id

# In[19]:


# checking if there is duplicated rows with the same originaltitle
df["original_title"].duplicated().sum()


# >There is 46 rows with duplicated original tittle

# In[20]:


#duplicated rows which have the same original tittle
df.drop_duplicates(['original_title'], keep ='first', inplace = True)
df.shape


# >Cleaning data is completed now, we have 3808 final rows to be analysed

# In[21]:


#Removing unnecessary columns
df.drop(['id','imdb_id','budget','revenue','cast','homepage','director','tagline','keywords','overview','production_companies','release_date','vote_count','vote_average'], axis=1, inplace=True)
df.head()


# In[22]:


#converting popularity, runtime, budget_adj, and revenue_adj into integers
df[['popularity','runtime','budget_adj', 'revenue_adj']] = df[['popularity','runtime','budget_adj', 'revenue_adj']].applymap(np.int64)
df.info()


# >all needed data were changed successfully into integers

# In[23]:


#Save a new csv file after cleaning data
df.to_csv('clean_tmdb_data.csv', index=False)


# <a id='eda'></a>
# ## Exploratory Data Analysis

# ### General look on the data

# In[24]:


# Having an overall look on the data
df.hist(figsize=(10,8));


# ### Research Question 1  (Is there a change in revenues of movies over years and what are the years with minimum and maximum revenues?)

# In[25]:


# showing a line graph to show change of movies revenues over time
def find_trend(column_x,column_y):
    #load clean data
    df = pd.read_csv('clean_tmdb_data.csv')
    #set graph size
    plt.figure(figsize=(15,5), dpi =120)
    #plotting the graph
    plt.plot(df.groupby(column_x)[column_y].sum())
    df.groupby(column_x)[column_y].sum().describe()
    max_value = df.groupby(column_x)[column_y].sum().idxmax()
    min_value = df.groupby(column_x)[column_y].sum().idxmin()
    return max_value,min_value,plt

maxval,minval,plt=find_trend('release_year','revenue_adj')
#x-axis label
plt.xlabel('Release Year', fontsize = 8)
#y-axis label
plt.ylabel('revenue', fontsize = 8)
#title of the graph
plt.title('revenues of movies')
plt.show()
print('Maximum revenue was in', maxval)
print('Minimum revenue was in',minval)


# ### Research Question 2  (Is there a change in popularity of movies over years and what are the years with minimum and maximum movies popularity?)

# In[26]:


maxval,minval,plt=find_trend('release_year','popularity')
#x-axis label
plt.xlabel('Release Year', fontsize = 8)
#y-axis label
plt.ylabel('popularity', fontsize = 8)
#title of the graph
plt.title('popularity of movies over years')
plt.show()
print('Maximum revenue was in', maxval)
print('Minimum revenue was in',minval)


# ### Research Question 3  (What are the top 10 movies acheiving revenues from 1960 till 2015?)

# In[27]:


df=df.sort_values(by = 'revenue_adj', ascending=False)
Top_10 =df.head(10)
Top_10


# In[28]:


#Setting the size of the chart

plt.figure(figsize=[20,5])

#Sorting datafreame descendingly according to the revenue acheived
df=df.sort_values(by = 'revenue_adj',ascending=False)

#Showing only top 10 movies acheiving revenue
Top_10 =df.head(10)

#Group original title by revenue in order to be able to graph it
sort=Top_10.groupby('original_title').sum()['revenue_adj']

#Sort values according to the grouped data and plot it on a bar graph
sort=sort.sort_values(ascending=False)
sort=sort.plot(kind = 'bar', color = 'green')

#determining plot details
plt.legend();
plt.title('Top 10 movies making revenue')
plt.xlabel('Movie name')
plt.ylabel('revenue');


# >Avatar is the movie with highest revenue from 1960 till 2015

# ### Research Question 4  (What are the top 10 movies with highest budget from 1960 till 2015?)

# In[29]:


#Setting the size of the chart

plt.figure(figsize=[20,5])

#Sorting datafreame descendingly according to the budget acheived
df=df.sort_values(by = 'budget_adj',ascending=False)

#Showing only top 10 movies according to budget
Top_10 =df.head(10)

#Group original title by budget in order to be able to graph it
sort=Top_10.groupby('original_title').sum()['budget_adj']

#Sort values according to the grouped data and plot it on a bar graph
sort=sort.sort_values(ascending=False)
sort=sort.plot(kind = 'bar', color = 'black')

#determining plot details
plt.legend();
plt.title('Top 10 movies according to budget')
plt.xlabel('Movie name')
plt.ylabel('budget');


# > The warrior's way is the movie with the highest budget 

# <a id='conclusions'></a>
# ## Conclusions
# 
# > Movies revenues has witnessed a huge increase over years, 2015 is the years in which movies had acgeived the highest revenues and 1960 is the lowest
# 
# > Movies' popularity had witnessed a massive increase over years to reach it's peak in 2015, while it was the lowest in 1966
# 
# > Top 10 movies generating revenues from 1960 till 2015 are:
# 
# 1-Avatar
# 2-Star wars
# 3-Titanic
# 4-The excorsist
# 5-Jaws
# 6-Star Wars: The Force Awakens
# 7-E.T. the Extra-Terrestrial
# 8-The Net
# 9-One Hundred and One Dalmatians	
# 10-The Avengers
# 
# >Top 10 movies according to budget from 1960 till 2015 are:
# 
# 1-The Warrior's Way
# 2-Pirates of the Caribbean: On Stranger Tides
# 3-Pirates of the Caribbean: At World's End
# 4-Superman Returns
# 5-Titanic
# 6-Spider-Man 3
# 7-Tangled
# 8-Avengers: Age of Ultron
# 9-Harry Potter and the Half-Blood Prince	
# 10-Waterworld
# 
# >Only one movie was from the top 10 movies with highest budget and was as well from the top 10 generating revenues, this movie is Titanic, this means that paying high budget is not a causation of generating high revenues
# 

# <a id='limitations'></a>
# ## Limitations

# >Budget and Revenues contain some illogical data, some budges are 120,300,400, etc, this is illogical
# 
# >There was alot of missing data
# 
# >There was some important data could have been added to the dataset as the number of awards won by each movie

# In[1]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:





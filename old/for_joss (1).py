
# coding: utf-8

# # Markdown Cell Example 
# markdown can be readibly interleaved and dispersed between code in notebooks 
# ## Explanation of code below
# The histogram (x-axis) binned readability score, (y-axis) counts of papers that occupy that readability score. 
# 
# The histogram is initially populated exclusively by the ART corpus, but the idea was every time a new author got scraped from scholar, it would be added in, such that with each persons new search our big picture of science readability would be better informed.
# 
# So the histogram changes a little modestly perceptible amount with the author scrape, but three dots pertaining to the authors easiest read, hardest read, and mean read where added.
# 
# These used to be ```[mean-standard dev, mean, mean+standard dev]``` but there was a flaw implementing that. It could just be that the plot looked way too busy around the mean, and it was harder to look at.
# 
# There is an issue with getting the dots to appear in the centre of histogram bins. I was working under the assumption, that if I knew the ```[min,mean,max]```, readability scores for Rick Gerkin, I could add to them half the bin width, and dot's would be centred. That is almost correct. I forgot that these calculations are not performed on pre binned data, so the x-coordinates of ```[min,mean,max]``` need to be slightly shifted to the nearest histogram bin start first.
# 
# Think of it as a bit like snapping something to a grid in photoshop.

# It should be easy to hack this code to run off a local machine, using sudo.
# Set up the Environment. This is now done in requirements, and the postBuild script.
# ```python
# !pip install matplotlib
# !pip install pandas
# !pip install seaborn
# 
# if os.path.exists('traingDats.p?dl=0'):
#     pass
# 
# else:
#     !wget https://www.dropbox.com/s/3h12l5y2pn49c80/traingDats.p?dl=0
#     !wget https://www.dropbox.com/s/crarli3772rf3lj/more_authors_results.p?dl=0
#     !wget https://www.dropbox.com/s/x66zf52himmp5ox/benchmarks.p?dl=0
# ```

# In[1]:


import pickle
import copy
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly as py


# In[2]:


try:
    with open('scraped_new.p?dl=0','rb') as f:
        texts = pickle.load(f) 
except:
    get_ipython().system('wget https://www.dropbox.com/s/1kc7alp79h701hx/scraped_new.p?dl=0')
    with open('scraped_new.p?dl=0','rb') as f:
        texts = pickle.load(f) 

queries = set([t['query'] for t in texts ])
temp = [t for t in texts if 'standard' in t.keys() and 'wikipedia' in t['link']]
science = ['cancer','Vaccines','evolution','climate change','Transgenic','photosysnthesis','evolution','GMO']
res = [t['standard'] for t in temp if t['query'] in science]
#res = [t['standard'] for t in temp]# if t['query'] in science]

mwp = np.mean(res)  
abstract_wiki = {'standard':mwp}


# In[3]:



with open('traingDats.p?dl=0','rb') as f:
        trainingDats = pickle.load(f) 

bmark = pickle.load(open('benchmarks.p?dl=0','rb'))
#bmark.append(abstract_wiki)
NAME,ar = pickle.load(open('more_authors_results.p?dl=0','rb'))
NAME = NAME[0]
trainingDats.extend(bmark)


# In[4]:



def get_heights(stats_items,histogram_content,x_sub_set):
    vertical_postions_indexs = []
    for i in stats_items:
        vertical_postions_indexs.append(find_nearest(histogram_content, i))
    bin_width_offset = (xys[1][0] - xys[0][0])/2.0
    x_sub_set = [ i+bin_width_offset for i in x_sub_set ]


    heights = []
    for i in vertical_postions_indexs:
        heights.append(xys[i][1])
    return heights, bin_width_offset


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def snap_to_grid(author_stats,bin_centers):
    author_stats_grid = []
    for as_ in author_stats:
        as_ = find_nearest(bin_centers,as_)
        author_stats_grid.append(bin_centers[as_])
    return author_stats_grid


# In[5]:


max_ART = np.max([ t['standard'] for t in trainingDats ])
publication = [ t['publication'] for t in trainingDats if t['standard'] == max_ART ]
keys = [ t.keys() for t in trainingDats if t['standard'] == max_ART ]

fname = [ t['file_name'] for t in trainingDats if t['standard'] == max_ART ]
bmark_max_art = {'standard':max_ART}
#max_ART


# In[6]:




standard_sci = [ t['standard'] for t in trainingDats ]
ar = [ t for t in ar if type(t) is type({})]
ar = [ t for t in ar if 'standard' in t.keys()]
xys = [ (h.get_x(),h.get_height()) for h in sns.distplot(standard_sci).patches ]

x_grid = [ h.get_x() for h in sns.distplot(standard_sci).patches ]
offset = float((x_grid[1] - x_grid[0])/2.0)
bin_centers = [gr+offset for gr in x_grid]
# this plot not used yet.

fig = plt.figure(figsize=(10, 8), dpi=80)
ax1 = fig.add_subplot(111)#)
mean_ = np.mean([a['standard'] for a in ar])
min_ = np.min([a['standard'] for a in ar])
max_ = np.max([a['standard'] for a in ar])
std_ = np.std([a['standard'] for a in ar])
stats_items = [mean_,min_,max_]

g = sns.distplot(standard_sci, label="Readability Index")


histogram_content = [x[0] for x in xys]
height_content = np.array([x[1] for x in xys])

hc = np.array(histogram_content)

x_sub_set=histogram_content


other_name=str('Phytochromobilin C15-Z,syn - C15-E,anti isomerization: concerted or stepwise?')
worst_height,_ = get_heights([max_ART],hc,x_sub_set)
mwp_height,_ = get_heights([mwp],hc,x_sub_set)

#bmark_max_art
worst_height = worst_height[0]
#bmark_stats_items_grid = snap_to_grid(bmark_stats_items,bin_centers)

#worst_distamnce = snap_to_grid(max_ART,bin_centers)
worst_distance = snap_to_grid([max_ART],bin_centers)
mwp_distance = snap_to_grid([mwp],bin_centers)


bmark_stats_items = [ b['standard'] for b in bmark ]
#bmark_stats_items.append(max_ART)
categories = ["upgoer 5", "Readibility Declining Over Time","Science of Writing","Post Modern Essay Generator"]#"Mean Wikipedia"]#,other_name]# "wikipedia science"]
bmark_heights, _ = get_heights(bmark_stats_items,histogram_content,x_sub_set)
heights, bwo = get_heights(stats_items,histogram_content,x_sub_set)
#bmark_heights.append(worst_height)
bmark_stats_items = [i+bwo for i in bmark_stats_items]
mean_a = mean_
min_a = min_ 
max_a = max_ 
xticks = list(range(0,45,5))

#print(xticks)
bmark_stats_items


# In[7]:



bmark_stats_items_grid = snap_to_grid(bmark_stats_items,bin_centers)
author_stats =[i for i in [mean_,min_,max_]]
author_stats_grid = snap_to_grid(author_stats,bin_centers)
mean_a_grid = snap_to_grid([mean_a],bin_centers)
x_sub_set_grid = snap_to_grid(x_sub_set,bin_centers)


# In[8]:


recalibrate_heights,b = get_heights(author_stats_grid,hc,x_sub_set)

heights[0] = np.max(recalibrate_heights)
heights[2] = recalibrate_heights[2]


# In[9]:


np.max(height_content)
heights


# In[10]:


bmark_heights
print(len(bmark_heights))
print(len(bin_centers))
print(len(bmark_stats_items))


# In[11]:


print(len(bmark_stats_items_grid))


# In[12]:


fig = plt.figure(figsize=(9, 9), dpi=100)
ax1 = fig.add_subplot(111)#)
g = sns.distplot(standard_sci, label="Readability Index")

benchmarks = pd.DataFrame({
'benchmarks': bmark_stats_items_grid,
    'CDF': bmark_heights
    })


data0 = pd.DataFrame({
'mean, min, maximum': author_stats_grid,
    'CDF': heights
    })


data2 = pd.DataFrame({
'Standard Reading Level': mean_a_grid,
    'CDF': np.max(height_content)
    })




legend_properties = {'weight':'bold','size':8}

ax = sns.regplot(data=benchmarks, x="benchmarks", y="CDF", fit_reg=False, marker="o", color="green")


ax = sns.regplot(data=data0, x="mean, min, maximum", y="CDF", fit_reg=False, marker="o", color="blue")
#ax = sns.regplot(x='Standard Reading Level', y='CDF',data=data1, fit_reg=False, marker="o", color="green")#, data=fmri)
ax = sns.regplot(data=data2, x="Standard Reading Level", y="CDF", fit_reg=False, marker="o", color="red")

legendMain=ax.legend(labels=[str("std deviation")], prop=legend_properties,loc='upper right')

legendSide0=ax.legend(labels=[NAME],prop=legend_properties,loc='center right')
legendSide1=ax.legend(labels=[str('Number of Documents: '+str(len(ar)))],prop=legend_properties,loc='upper left')


legendMain=ax.legend(labels=[str("Google scholar author relative to ART Corpus distribution. Total docs: ")+str(len(trainingDats))], prop=legend_properties,loc='upper left')


#for i,j,k in zip(author_stats_grid,heights,[str(NAME)+' mean',str(NAME)+' min',str(NAME)+' max']):

xinterval = author_stats_grid
xinterval.extend(bmark_stats_items_grid)
#xinterval.append(other)
#coords = copy.copy([item for item in ax.get_xticklabels()])
x1,y1,z1 = (mwp_distance[0],mwp_height[0],str('mean wikipedia'))
xinterval.insert(4,x1)
ax.set_xticks(xinterval)

new_categories = []
new_categories.extend([str(NAME)+' mean',str(NAME)+' min',str(NAME)+' max'])
new_categories.extend(categories)
new_categories.insert(4,'Mean Wikipedia Biological Science')
x,y,z = (worst_distance[0],worst_height,other_name)

data3 = pd.DataFrame({
'Standard Reading Level': [x1],
    'CDF': [y1]
    })
ax = sns.regplot(data=data3, x='Standard Reading Level', y="CDF", fit_reg=False, marker="o", color="green")

#new_categories.append(other_name)

ax.set_xticklabels(new_categories, minor=False, rotation=90)
ax.set_xticklabels(new_categories, minor=True, rotation=0)

ax = sns.regplot(data=benchmarks, x="benchmarks", y="CDF", fit_reg=False, marker="o", color="green")

# Get the two lines from the axes to generate shading
l1 = ax.lines[0]
#l2 = ax.lines[1]

# Get the xy data from the lines so that we can shade
x1_ = l1.get_xydata()[:,0]
y1 = l1.get_xydata()[:,1]
x1=[mean_-std_,mean_+std_]
index_0 = find_nearest(x1[0],x1_)
index_1 = find_nearest(x1[1],x1_)

reducedx = x1_[index_0:index_1]
reducedy = y1[index_0:index_1]

print(x1_)
ax.fill_between(reducedx,reducedy, color="red", alpha=0.1)
#ax.fill_between(x2,y2, color="blue", alpha=0.3)
ax2 = plt.twiny()
xticks = list(range(0,45,5))

plt.xticks(xticks)
#ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
plt.tick_params(axis="y", labelcolor="r", pad=8)




#ax.text(x-0.25,y+0.005,z, rotation=90)  


#ax.text(x1,y1,z1, rotation=90)  

    
plt.savefig(str(NAME)+'_author_readability.png')
plt.show()


# In[13]:


plt.savefig(str(NAME)+'_author_readability.png')

new_categories
x,y,z = (worst_distance[0],worst_height,other_name)
x
y
z


# In[14]:


x,y,z = (mwp_distance[0],mwp_height[0],str('mean wikipedia'))


# In[15]:


xticks
xinterval


# In[16]:


data2


# In[17]:


data0


# In[18]:


bmark_stats_items


# In[19]:


#bmark


# In[20]:


trainingDats[-1]


#tuple1= (10,20,30,40,50)
#print(tuple1[:2]) 
#print(tuple1[2:])  
#print(tuple1[1:4])

#name ="     student     "
#print(name.strip()) #removes the whitespace - strip method

#name="person\ndog\ncat" # /n puts each on separate lines
#print(name)

#afterSplitting= name.split('\n') # data is in a list
#print(afterSplitting)

#data = "Hello?how are you? I'm doing good!"
#print(data.split('.'))

#labels = open("coco.names").read().strip().split('\n')
#print(labels)

import numpy as np

#array=[100,300,120,540,600] 
#print(np.argmax(array)) #prints the index number of the maximum values inside the given array ^^

# [p,x,y,w,h,c1,c2] = probability of object, x-cor,y-cor, width of object, height of object, scores 1 , scores 2
#detection=[0,20,40,80,80,0,1]
#a = detection[0:4] #getting value from 0 to 4
#print(type(a))
#array=np.array(a) #no commas between data in arrays

#print(array)

name= "bicycle"
label="bicycle"
confidences=0.923456
print('{}:{:.2f}'.format(label,confidences*100)) # .2f and *100 are for percentages


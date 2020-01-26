import numpy as np

def readFromFile(filename):
	#file = np.fromfile("dataset.json")
	#X = np.array(file)
	#print(X)
	import json
	with open(filename) as f:
		data = json.load(f)
	X = np.asarray(data["x"])
	X = X/ 255
	Y = np.asarray(data["y"])
	Y = Y.astype(int)
	
	return X,Y




def save_weights(W1, W2, b1, b2):
	f=open('net_weights_W1.txt','wb')
	np.savetxt(f,W1, delimiter=',',fmt="%0.3f", newline="],\n[")	
	f.close()
	f=open('net_weights_b1.txt','wb')
	np.savetxt(f,b1, delimiter=',',fmt="%0.3f", newline=",\n")	
	f.close()
	f=open('net_weights_W2.txt','wb')
	np.savetxt(f,W2, delimiter=',',fmt="%0.3f", newline="],\n[")	
	f.close()
	f=open('net_weights_b2.txt','wb')
	np.savetxt(f,b2, delimiter=',',fmt="%0.3f", newline=",\n")	
	f.close()


def load_weights(n_x,n_h,n_y):
	
	tempW1 = []
	with open('net_weights_W1.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			temparr = x.replace("],","").replace("[","").replace("\n","").replace("'","").split(",")
			for y in temparr:
				#check for empty string
				if not y:
					continue
				tempW1.append(float(y))	
	W1 = np.array(tempW1).reshape(n_h,n_x)
	
	tempW2 = []
	with open('net_weights_W2.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			temparr = x.replace("],","").replace("[","").replace("\n","").replace("'","").split(",")
			for y in temparr:
				#check for empty string
				if not y:
					continue
				tempW2.append(float(y))
			
	W2 = np.array(tempW2).reshape(n_y,n_h)
	
	tempb1 = []
	with open('net_weights_b1.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			tempb1.append(float(x.replace(",\n","")))
	b1 = np.expand_dims(np.array(tempb1), axis=1)
	
	tempb2 = []
	with open('net_weights_b2.txt','r') as f: 
		contents = f.readlines()
		for x in contents:
			tempb2.append(float(x.replace(",\n","")))
	b2 = np.expand_dims(np.array(tempb2), axis=1)
	
	return W1, W2, b1, b2


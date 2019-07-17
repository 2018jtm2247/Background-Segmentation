import cv2
import numpy as np

K = 3
alpha = 0.01
#img_array = []

video = cv2.VideoCapture('umcp.mpg')

if not video.isOpened():
	print ('File did not open')
	#return

column = int(video.get(3)) # number of column from video
row = int(video.get(4))	  # number of rows from video
num = int(row * column)
fps = video.get(5)   # frames per second
sizes = (column,row)

mean = [40,120,200]         # Initial Mean of all the Gaussians 
var = [100,100,100]   # Initial Variance of all the Gaussians
weight = [0.33,0.33,0.34]   # Weight of each Gaussian
gauss_mean = []
gauss_var = []
gauss_weight = []

#out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (column, row ))

for i in range(num):
	gauss_mean.append(mean)	
	gauss_var.append(var)
	gauss_weight.append(weight)
#print(gauss_weight)
	


#########################################################################################################

def update_gauss(i,gauss_no,pixel):

	a = 1 / (np.sqrt(2*np.pi * gauss_var[i][gauss_no]))
	b = (pixel - gauss_mean[i][gauss_no]) * (pixel - gauss_mean[i][gauss_no])
	c = b / (2*gauss_var[i][gauss_no])
	pdf = a * np.exp(-c)
	
	ro = alpha * pdf
	gauss_mean[i][gauss_no] = (1 - ro)*gauss_mean[i][gauss_no] + (ro*pixel)
	#gauss_var[i][gauss_no] = np.sqrt((1-ro)*gauss_var[i][gauss_no]*gauss_var[i][gauss_no] + ro*(pixel-gauss_mean[i][gauss_no])*(pixel-gauss_mean[i][gauss_no]))
	gauss_var[i][gauss_no] = ((1-ro)*gauss_var[i][gauss_no]+ ro*(pixel-gauss_mean[i][gauss_no])*(pixel-gauss_mean[i][gauss_no]))
	
	for z in range(K):
		if z == gauss_no:
			gauss_weight[i][z] = ((1-alpha)*gauss_weight[i][z]) + alpha
		else:
			gauss_weight[i][z] = ((1-alpha)*gauss_weight[i][z])
	

########################################

def new_gauss(i,pixel):
	tryy = 10 
	for z in range(K):
		a = gauss_weight[i][z] / np.sqrt(gauss_var[i][z])
		if a < tryy :
			tryy=a
			index = z
	gauss_mean[i][index] = pixel
	gauss_var[i][index] = 200
	gauss_weight[i][index] = 0.01 
#########################################


while True:
	check,frame = video.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
	#print(gray.shape)
	#gray.reshape(gray,10)
	gray = gray.ravel()    # convert 2D numpy array to 1D array
	#print(gray.shape)

	for i in range(num):	
		pixel = gray[i]
		#print(pixel)

			# initializing lists
		mean_pixel_dis = [0,0,0]      # distnace of pixel from mean of gaussians	
		min_matching = [0,0,0]        # minimum threshold for detection of particular Gaussian
		max_matching = [0,0,0]	      # maximum threshold for detection of particular Gaussian
		none_matched = 0		
		        
			
		for z in range(K):
			min_matching[z] = gauss_mean[i][z] - (2.5 * np.sqrt(gauss_var[i][z]))
			max_matching[z] = gauss_mean[i][z] + (2.5 * np.sqrt(gauss_var[i][z]))
		
	
			if (pixel > min_matching[z] ) and (pixel < max_matching[z] ):
				if gauss_weight[i][z] > 0.95:
					gauss_no = z
					gray[i] = mean[z]
					update_gauss(i,gauss_no,pixel)
				#else:
					#gray[i] = 255
					
 		
			else:
				none_matched = none_matched + 1
				if none_matched ==K:
					new_gauss(i,pixel)
					gray[i] = mean[2]

		
			
		'''			
		R=[0,0,0]
		W=[]
		Rsort=[]
		for z in range(K):
			R[z] = gauss_weight[i][z] / np.sqrt(gauss_var[i][z])
			#print(gauss_weight[i][z])
						
		min_index = R.index(min(R))
		max_index = R.index(max(R))
		middle_index = 3-(min_index + max_index)

	
		W.append(gauss_weight[i][max_index])
		W.append(gauss_weight[i][middle_index])
		W.append(gauss_weight[i][min_index])

		#print(Rsort,W)
		#time.sleep(0.05)

	
		if W[0] < 0.95:
			gray[i] = 0
			

		else:
			gray[i] = 255'''
				
		#print(gray[i])	
	#convert into 2D array
	gray_new = gray.reshape(row, column)    # converting !d array to original numpy array
	#print(gray_new.shape)
	cv2.imshow('This is BGS', gray_new)
	#out.write(gray_new) 
	#img_array.append(gray_new)
	if(cv2.waitKey(30) == 27) & 0xff : 
		break


video.release()
#out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, sizes)
#for i in range(len(img_array)):
#	out.write(img_array[i])
	

#out.release()
cv2.destroyAllWindows()











""""
Disclaimer:

No guarantees of performance accompany this software,
nor is any responsibility assumed on the part of the author(s).
The software has been tested extensively and every effort has been
made to insure its reliability.

This software is provided by the provider ``as is'' and
any express or implied warranties, including, but not limited to, the
implied warranties of merchantability and fitness for a particular purpose
are disclaimed.  In no event shall the provider or the contributor(s) be liable
for any direct, indirect, incidental, special, exemplary, or consequential
damages (including, but not limited to, procurement of substitute goods
or services; loss of use, data, or profits; or business interruption),
however caused and on any theory of liability, whether in contract, strict
liability, or tort (including negligence or otherwise) arising in any way
out of the use of this software, even if advised of the possibility of
such damage. There are inherent dangers in the use of any software, and
you are solely responsible for determining whether this software product
is compatible with your equipment and other software installed on your
equipment. You are also solely responsible for the protection of your equipment
and backup of your data, and the provider will not be liable for any damages
you may suffer in connection with using, modifying, or distributing this
software product.

"""

#Examples on how to use the functions provided in toolSAR.py

import numpy as np
import toolSAR

#Loading data, in slc format
[slchh,slchv,slcvh,slcvv]=np.load('mydata.npy')
slccross=(slchv+slcvh)/2


#Pauli decomposition
[paulisingle,paulidouble,paulivol]=toolSAR.slctopauli(slchh,slcvv,slccross)

#RGB plot of the Pauli decomposition: Depending on your data, you might want to change
#the values for the RGB plotting, by using histograms
toolSAR.rgbplotrout(np.log(paulidouble),-8,2,1.2*np.log(paulivol),-10,2,np.log(paulisingle),-8,2,'Pauli decomposition')



#Circular polarization
[srr,sll,srl]=toolSAR.slchvtoslcrl(slchh,slcvv,slccross)

#Krogager decomposition and plot
srrsq=np.real(srr*np.conjugate(srr))
sllsq=np.real(sll*np.conjugate(sll))
srlsq=np.real(srl*np.conjugate(srl))  #=ks^2 for the Krogager decomposition

[ks,kh,kd]=toolSAR.slctokrogager(srrsq,sllsq,srlsq)

#In some cases, some parameters can be null, which is problematic when we want 
#to use the logarithm to take a look at the data, so we give them a very small 
#value instead
q=np.where(ks<=0)
ks[q]=10**(-10)
q=np.where(kh<=0)
kh[q]=10**(-10)
q=np.where(kd<=0)
kd[q]=10**(-10)

#Plot of the Krogager decomposition
#Once again, you can change the numerical values depending on your data
toolSAR.rgbplotrout(np.log(kh),-5,2,np.log(kd),-10,2,np.log(ks),-10,2,'Krogager decomposition')




#Multilooking
#incoherent formulation (covariance matrices)
#nb controls the area used to make the multi-looking
nb=2
covmats=toolSAR.slctocov(slchh,slcvv,slccross,nb)



#CLoude and Pottier decomposition
CPdec=toolSAR.CP_dec(covmats)



#Freeman and Durden decomposition
DFdec=toolSAR.DF_dec(covmats)
#Some powers here could be negative or null, so we just give them a very small value instead
qfd=np.where(DFdec<=0)
DFdec[qfd]=10**(-10)
#Plot the result
toolSAR.rgbplotrout(np.log(DFdec[:,:,2]),-8,2,np.log(DFdec[:,:,0]),-8,2,np.log(DFdec[:,:,1]),-8,2,'Durden and Freeman decomposition')

#The same can be done for the other decompositions, and to visualize them
#Van Zyl's decomposition
VZdec=toolSAR.VZ_dec(covmats)
#NNED 
NNEDdec=toolSAR.NNED_dec(covmats)

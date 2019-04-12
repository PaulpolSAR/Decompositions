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



#This script contains a list of functions to make polarimetric decompositions with fully polarimetric SAR data.
#It contains, in this order:
#Pauli decomposition (slc data)
#Basis change from HV to RL polarization (slc data)
#Krogager decomposition (slc data, in RL basis)
#Multilooking on an image (from slc data to covariance matrix). Retains the same ground resolution.
#Cloude and Pottier decomposition (covariance matrix)
#Durden and Freeman decomposition (covariance matrix)
#Van Zyl decomposition (covariance matrix)
#Plotting procedure to create RGB images based on results from decompositions.

import numpy as np
from numba import jit
from matplotlib import pyplot as plt



#######################################################################################################
#######################################################################################################
#Pauli decomposition
#This function makes the Pauli decomposition of the full image
#It takes as input the three full matrices of single look complex data, and gives back the matrices of the three squared elements
#Inputs:    -slchh: a m*n array containing the slc data in the HH polarization
#           -slcvv: a m*n array containing the slc data in the VV polarization
#           -slccross: a m*n array containing the slc data in the HV polarization   
#Outputs:   -paulisignle: a m*n array containing the power of the Pauli single bounce component
#           -paulidouble: a m*n array containing the power of the Pauli double bounce component
#           -paulivol: a m*n array containing the power of the Pauli volume component

#######################################################################################################
#######################################################################################################
def slctopauli(slchh,slcvv,slccross):
    paulisingle=((slchh+slcvv)/np.sqrt(2))
    paulidouble=((slchh-slcvv)/np.sqrt(2))
    paulivol=np.sqrt(2)*slccross

    paulisingle=(paulisingle*np.conjugate(paulisingle)).real
    paulidouble=(paulidouble*np.conjugate(paulidouble)).real
    paulivol=(paulivol*np.conjugate(paulivol)).real

    return paulisingle ,paulidouble, paulivol




#######################################################################################################
#######################################################################################################
#From HV to RL polarizations
#This functions transforms the slc image in the HV polarization to the circular RL polarization
#Takes as an input the three full images of slc data in HV polarization, and returns circular polarizations

#Inputs:    -slchh: a m*n array containing the slc data in the HH polarization
#           -slcvv: a m*n array containing the slc data in the VV polarization
#           -slccross: a m*n array containing the slc data in the HV polarization   
#Outputs:   -srr: a m*n array containing the slc data in the RR polarization
#           -sll: a m*n array containing the slc data in the LL polarization
#           -slr: a m*n array containing the slc data in the RL polarization   

#######################################################################################################
#######################################################################################################
def slchvtoslcrl(slchh,slcvv,slccross):
    
    srr=(1j*slccross+(slchh-slcvv)/2)
    sll=(1j*slccross-(slchh-slcvv)/2)
    srl=1j*(slchh+slcvv)/2
    
    return srr, sll, srl




#######################################################################################################
#######################################################################################################
#Krogager decomposition
#This function makes the Krogager decomposition
#Paper: Krogager, Ernst. "Decomposition of the radar target scattering matrix with application to high resolution target imaging.
#" NTC'91-National Telesystems Conference Proceedings. IEEE, 1991.
#Takes as input the three squared modules of slc elements in circular polarization of a whole scene,
#and returns its krogager decomposition in three m*n arrays
#Inputs:    -srrsq: a m*n array containing the squared module of the slc data in the RR polarization
#           -sllsq: a m*n array containing the squared module of the slc data in the LL polarization
#           -srlsq: a m*n array containing the squared module of the slc data in the RL polarization   
#Outputs:   -ks: a m*n array containing the sphere component power of the Krogager decomposition
#           -kd: a m*n array containing the diplane component power of the Krogager decomposition
#           -kh: a m*n array containing the helix component power of the Krogager decomposition   

#######################################################################################################
#######################################################################################################
@jit
def slctokrogager(srrsq,sllsq,srlsq):
    
    [m,n]=np.shape(srrsq)
    ks=srlsq
    kd=np.zeros((m,n))
    kh=np.zeros((m,n))
     
    for i in np.arange(0,m):
        for j in np.arange(0,n):

    
            if srrsq[i,j]>sllsq[i,j]:
                kd[i,j]=sllsq[i,j]
                kh[i,j]=(np.sqrt(srrsq[i,j])-np.sqrt(sllsq[i,j]))**2
            else:
                kd[i,j]=srrsq[i,j]
                kh[i,j]=(np.sqrt(sllsq[i,j])-np.sqrt(srrsq[i,j]))**2
        
    return ks, kd, kh



#######################################################################################################
#######################################################################################################
#From slc to covariance matrix (in the 3*3 format)
#It retains the same ground resolution as the slc data
#Inputs:
#       -slchh: m*n array containing the slc data in HH polarization
#       -slcvv: m*n array containing the slc data in VV polarization
#       -slccross: m*n array containing the slc data in HV polarization
#       -nb: integer, defines size of the square window use for covariance claculation. The window size is (2*nb+1)*(2*nb+1).
#Output:
#       -result: m*n*9 array, containing the covariance matrix elements in the third dimension
#######################################################################################################
#######################################################################################################
@jit(nopython=True)
def slctocov(slchh,slcvv,slccross,nb):
    
    [m,n]=slchh.shape
    result=np.zeros((m,n,9),dtype=np.complex64)
    slccross=slccross*np.sqrt(2)
    


    for i in np.arange(0,m):
        for j in np.arange(0,n):
            
            lilone=np.array((0,i-nb))
            liltwo=np.array((i+nb+1,m))
            
            ljlone=np.array((0,j-nb))
            ljltwo=np.array((j+nb+1,n))

            
    
            temphh=np.copy(slchh[np.max(lilone):np.min(liltwo),np.max(ljlone):np.min(ljltwo)])
            tempvv=np.copy(slcvv[np.max(lilone):np.min(liltwo),np.max(ljlone):np.min(ljltwo)])
            tempcross=np.copy(slccross[np.max(lilone):np.min(liltwo),np.max(ljlone):np.min(ljltwo)])

            [mbis,nbis]=temphh.shape
            
            temphh=temphh.reshape((mbis*nbis,1))
            tempvv=tempvv.reshape((mbis*nbis,1))
            tempcross=tempcross.reshape((mbis*nbis,1))


            data=np.hstack((temphh, tempcross, tempvv))
            
            covmat=np.dot(data.T,np.conjugate(data))/(mbis*nbis)
            result[i,j,:]=covmat.reshape(9,)
            
    return result





#######################################################################################################
#######################################################################################################
#Incoherent decompositions: Cloude and Pottier h-alpha
    
#Paper: Pottier, Eric, and J-S. Lee. "Application of the «H/A/alpha» polarimetric decomposition theorem
#for unsupervised classification of fully polarimetric SAR data based on the wishart distribution." 
#SAR workshop: CEOS Committee on Earth Observation Satellites. Vol. 450. 2000.
    
#Input:    -data: The covariance matrices in a m*n*9 format
#Output:   -CPdec: a m*n*3 array. In the third dimension are contained in this order: H, alpha, Anisotropy
#######################################################################################################
#######################################################################################################
@jit(nopython=True)
def CP_dec(data):
    
    [m,n,p]=data.shape
    

    #Initialization of the output matrices
    CPdec=np.zeros((m,n,3))
    
    for i in np.arange(0,m):
        for j in np.arange(0,n):
            
            covdata=(np.copy(data[i,j,:])).reshape((3,3))
            
            
            ##############################################################
            #Cloude and Potier decomposition
            #Going from the covariance matrix to the coherency matrix
            ##############################################################
            
            #Switch from covariance to coherency matrix
            N=np.zeros((3,3),dtype=np.complex64)
            N[0,0]=1
            N[0,2]=1
            N[1,0]=1
            N[1,2]=-1
            N[2,1]=np.sqrt(2)
            cohedata=0.5*np.dot(np.dot(N,covdata),N.T)
            
            #Eigenvalue/eigenvectors
            [V,D]=np.linalg.eig(cohedata)
            V2=V.real

            
            #Reordering of the eigenvalues and eigenvectors
            
            firstmax=np.max(V2)
            cpmaxpos=((np.where(V2==firstmax))[0])[0]
            firstmin=np.min(V2)
            cpminpos=((np.where(V2==firstmin))[0])[0]
            V2[cpmaxpos]=firstmin
            
            middle=np.max(V2)
            cpmiddlepos=((np.where(V2==middle))[0])[0]
            cpmiddlepos=cpmiddlepos
            
            #Eigenvalues and eigenvectors in decreasing order
            V2[0]=firstmax
            V2[1]=middle
            V2[2]=firstmin
            
            ev=np.zeros((3,3),dtype=np.complex128)
            ev[:,0]=D[:,cpmaxpos]
            ev[:,1]=D[:,cpmiddlepos]
            ev[:,2]=D[:,cpminpos]            
            D=ev
                        
            #Entropy
            Po=np.sum(V2)
            ent1=V2/Po
            H=-np.sum(ent1*np.log(ent1))/np.log(3)
            
            #Anisotropy
            Anis=(V2[1]-V2[2])/(V2[1]+V2[2])
            
            #Alpha angle
            Dal=(np.sqrt(D[0,:]*np.conjugate(D[0,:]))).real
            Dal=np.arccos(Dal)
            alpha=np.sum(ent1*Dal)
            
            CPdec[i,j,:]=[H, alpha, Anis]

    return CPdec



#######################################################################################################
#######################################################################################################
#Incoherent decompositions: Freeman and Durden's decomposition
    
#Paper: Freeman, Anthony, and Stephen L. Durden. "A three-component scattering model for polarimetric SAR data."
#IEEE Transactions on Geoscience and Remote Sensing 36.3 (1998): 963-973.
    
#Input:    -data: The covariance matrices in a m*n*9 format 
#Outputs:  -DFdec: a m*n*3 array. In the third dimension are contained in this order: pv,ps,pd
#######################################################################################################
#######################################################################################################
@jit(nopython=True)
def DF_dec(data):
    
    [m,n,p]=data.shape
    
    #Initialization of the output matrices
    DFdec=np.zeros((m,n,3))
    
    for i in np.arange(0,m):
        for j in np.arange(0,n):
            
            covdata=(np.copy(data[i,j,:])).reshape((3,3))
            
            ##############################################################
            #Freeman and Durden decomposition
            ##############################################################
            
            fvfreeman=(3*covdata[1,1]/2).real
            
            if (covdata[0,2]).real<=0:
                betafreeman=1
                
                Af=(covdata[0,0]-fvfreeman).real
                Bf=(covdata[2,2]-fvfreeman).real
                Cf=((covdata[0,2]).real-fvfreeman/3).real
                Df=(covdata[0,2]).imag
                
                if (Af+Bf-2*Cf)!=0:
                    fdfreeman=(Df**2+(Cf-Bf)**2)/(Af+Bf-2*Cf)
                else:
                    fdfreeman=Bf
                
                
                fsfreeman=Bf-fdfreeman
            
                if fdfreeman!=0:
                    alphafreeman=np.complex((Cf-Bf)/fdfreeman+1, Df/fdfreeman)
                else:
                    alphafreeman=np.complex(1,1)
               
                
                
            else:           
                alphafreeman=-1
                
                Af=(covdata[0,0]-fvfreeman).real
                Bf=(covdata[2,2]-fvfreeman).real
                Cf=(covdata[0,2]).real-fvfreeman/3
                
                if (Af+Bf+2*Cf)!=0:                    
                    fsfreeman=(Cf+Bf)**2/(Af+Bf+2*Cf)
                else:
                    fsfreeman=Bf
                    
                fdfreeman=Bf-fsfreeman
                betafreeman=(Af+Cf)/(Cf+Bf)
                
                
            pvfreeman=(8*fvfreeman/3)
            psfreeman=(fsfreeman*(1+betafreeman*np.conjugate(betafreeman))).real
            pdfreeman=(fdfreeman*(1+alphafreeman*np.conjugate(alphafreeman))).real
            
            
            DFdec[i,j,:]=[pvfreeman, psfreeman, pdfreeman]
            
            
    return DFdec

#######################################################################################################
#######################################################################################################
#Incoherent VanZyl decomposition

#van Zyl, Jakob J. "Application of Cloude's target decomposition theorem to polarimetric imaging radar data."    
#Radar polarimetry. Vol. 1748. International Society for Optics and Photonics, 1993.
    
#Input:    -data: The covariance matrices in a m*n*9 format
#Outputs:  -VZdec: a m*n*3 array. In the third dimension are contained in this order: pv,ps,pd
#######################################################################################################
#######################################################################################################
@jit(nopython=True)
def VZ_dec(data):
    
    [m,n,p]=data.shape
    
    #Initialization of the output matrices
    VZdec=np.zeros((m,n,3))
    
    for i in np.arange(0,m):
        for j in np.arange(0,n):
            
            covdata=(np.copy(data[i,j,:])).reshape((3,3))
                      
            ##############################################################
            #Van Zyl decomposition
            ##############################################################
            
            Kvan=(covdata[0,0]).real
            
            matvanzyl=covdata/Kvan
            
            rhovan=matvanzyl[0,2]
            etavan=(matvanzyl[1,1]).real
            zetavan=(matvanzyl[2,2]).real
            
            deltavan=(zetavan-1)**2+4*((rhovan*np.conjugate(rhovan)).real)
            lambda1van=(Kvan/2)*(zetavan+1+np.sqrt(deltavan))
            lambda2van=(Kvan/2)*(zetavan+1-np.sqrt(deltavan))
            lambdacanop=Kvan*etavan
    
            Avan=(zetavan-1+np.sqrt(deltavan))
            Bvan=(zetavan-1-np.sqrt(deltavan))
        
            alphavan=(2*rhovan)/Avan
            betavan=(2*rhovan)/Bvan
            
            if (alphavan).real>0:
                angleodd=alphavan
                lambdaodd=lambda1van*((Avan**2)/(Avan**2+4*((rhovan*np.conjugate(rhovan)).real)))
                angleeven=betavan
                lambdaeven=lambda2van*((Bvan**2)/(Bvan**2+4*((rhovan*np.conjugate(rhovan)).real)))
            else:
                angleeven=alphavan
                lambdaeven=lambda1van*((Avan**2)/(Avan**2+4*((rhovan*np.conjugate(rhovan)).real)))
                angleodd=betavan
                lambdaodd=lambda2van*((Bvan**2)/(Bvan**2+4*((rhovan*np.conjugate(rhovan)).real)))
            
            
            pcanopvz=lambdacanop
            poddvz=lambdaodd*(1+(angleodd*np.conjugate(angleodd)).real)
            pevenvz=lambdaeven*(1+(angleeven*np.conjugate(angleeven)).real)
            
            VZdec[i,j,:]=[pcanopvz, poddvz, pevenvz]
            
            
    return  VZdec


#######################################################################################################
#######################################################################################################
#This function makes a RGB plot, and creates a linear colorscale for each component.
#Inputs:   -mat1: m*n array, values should be in dbs, controls the red colour
#          -low1: lower bound for the mat1 values
#          -up1: upper bound for the mat1 values
#          -mat2: m*n array, values should be in dbs, controls the green colour
#          -mat3: m*n array, values should be in dbs, controls the blue colour
#          -low2,up2,low3,up3: lower and upper bounds for respectively the green and blue colours
#Output:   Just plots the RGB image, doesn't actively return anything    
#######################################################################################################
#######################################################################################################
def rgbplotrout(mat1,low1,up1,mat2,low2,up2,mat3,low3,up3,titlestr):
    
    
    q=np.where(mat1<low1)
    mat1[q[0],q[1]]=low1
    q=np.where(mat1>up1)
    mat1[q[0],q[1]]=up1
    
    
    q=np.where(mat2<low2)
    mat2[q[0],q[1]]=low2
    q=np.where(mat2>up2)
    mat2[q[0],q[1]]=up2
    
    
    q=np.where(mat3<low3)
    mat3[q[0],q[1]]=low3
    q=np.where(mat3>up3)
    mat3[q[0],q[1]]=up3
    
    
    
    #totalmin=np.min([np.min(paulisingleplt), np.min(paulidoubleplt), np.min(paulivolplt)])
    mat1=mat1-np.min(mat1)
    mat2=mat2-np.min(mat2)
    mat3=mat3-np.min(mat3)
    
    #totalmax=np.max([np.max(paulisingleplt), np.max(paulidoubleplt), np.max(paulivolplt)])
    mat1=(mat1/(np.max(mat1)))
    mat2=(mat2/(np.max(mat2)))
    mat3=(mat3/(np.max(mat3)))
    
    
    
    plttot=(np.dstack([mat1, mat2, mat3]))  #for plotting
    
    plt.figure()
    plt.imshow(plttot)
    plt.title(titlestr)
    plt.show
    
    return




import numpy as np

##############################################################################
##############################################################################
## Prove that 1D estimation of multidimensional interactions is biased 
##############################################################################
##############################################################################

def biased_2_dim_2_attr_Dim1_ER_DIR_homog_h1(
    h1str, ## Homophily value that is equal in all entries of h1 (unbiased interaction)
    h0_mtrx, ## Homophily matrix h0
    k ## Consolidation
    ):
    ## THIS IS ONLY VALID FOR HOMOGENEOUS POPULATION FRACTIONS: [[0.5,0.5],[0.5,0.5]]
    ## The following try-except is to verify that h1str is not a matrix or anything
    ## like that
    try:
        len(h1str)
        raise Exception(f"h1str must be a float, not {type(h1str)}")
    except TypeError: ## len() should not work I use this instead of checking the type in case there is some errors with np.float64, python's float, and other floats
        pass

    h0_mtrx = np.array(h0_mtrx)

    F_mtrx_effective = np.array([[k,(1-k)],[(1-k),k]])
    h1_estimated = np.zeros((2,2)) + np.nan
    for i in range(2):
        for j in range(2):

            h1_estimated[i,j] = 0
            for ii in range(2):
                for jj in range(2):
                    h1_estimated[i,j] += F_mtrx_effective[ii,i] * F_mtrx_effective[jj,j] * h0_mtrx[ii,jj]
    return h1str*h1_estimated
    
def biased_2_dim_2_attr_Dim1_ER_DIR_homog_h1_h1_00(h1str,h0_mtrx,k):
    ## THIS IS ONLY VALID FOR HOMOGENEOUS POPULATION FRACTIONS: [[0.5,0.5],[0.5,0.5]]
    ## Only to have an example of the closed expression
    return h1str*(k**2*h0_mtrx[0,0] + k*(1-k)*h0_mtrx[0,1] + k*(1-k)*h0_mtrx[1,0] + (1-k)**2*h0_mtrx[1,1])